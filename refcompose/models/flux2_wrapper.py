"""
FLUX.2 Dev Backbone Wrapper, Position-Aware Latent Packing, Rectified-Flow Euler Sampler,
and Lightweight MMDiT Test Backbone (Paper Section 3.1, 3.4, 4.1-4.3).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from refcompose.config import RefComposeConfig
from refcompose.models.dsa_lora import (
    DSAConditionInjectionProcessor,
    clear_kv_cache,
    inject_dsa_lora,
)
from refcompose.utils.image_ops import pil_to_tensor, tensor_to_pil


# ---------------------------------------------------------------------------
# Lightweight Standalone MMDiT + VAE (for unit tests, CI, and local debugging)
# Mirrors FLUX.2's exact double-stream + single-stream interface & tensor shapes
# ---------------------------------------------------------------------------


class _MiniDoubleStreamAttention(nn.Module):
    """MMDiT joint attention module matching `diffusers` FLUX double-stream interface."""

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])

        self.add_q_proj = nn.Linear(dim, dim, bias=True)
        self.add_k_proj = nn.Linear(dim, dim, bias=True)
        self.add_v_proj = nn.Linear(dim, dim, bias=True)
        self.to_add_out = nn.Linear(dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)
        self.norm_added_q = nn.RMSNorm(self.head_dim)
        self.norm_added_k = nn.RMSNorm(self.head_dim)
        self.__dict__["processor"] = None

    def set_processor(self, processor: Any) -> None:
        self.__dict__["processor"] = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Any:
        if self.processor is not None:
            return self.processor(
                self,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )
        bsz, img_len, dim = hidden_states.shape
        q_i = self.norm_q(self.to_q(hidden_states).view(bsz, img_len, self.heads, self.head_dim).transpose(1, 2))
        k_i = self.norm_k(self.to_k(hidden_states).view(bsz, img_len, self.heads, self.head_dim).transpose(1, 2))
        v_i = self.to_v(hidden_states).view(bsz, img_len, self.heads, self.head_dim).transpose(1, 2)

        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1]
            q_t = self.norm_added_q(
                self.add_q_proj(encoder_hidden_states).view(bsz, txt_len, self.heads, self.head_dim).transpose(1, 2)
            )
            k_t = self.norm_added_k(
                self.add_k_proj(encoder_hidden_states).view(bsz, txt_len, self.heads, self.head_dim).transpose(1, 2)
            )
            v_t = self.add_v_proj(encoder_hidden_states).view(bsz, txt_len, self.heads, self.head_dim).transpose(1, 2)
            q = torch.cat([q_t, q_i], dim=2)
            k = torch.cat([k_t, k_i], dim=2)
            v = torch.cat([v_t, v_i], dim=2)
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, txt_len + img_len, dim)
            txt_out = self.to_add_out(out[:, :txt_len])
            img_out = self.to_out[1](self.to_out[0](out[:, txt_len:]))
            return img_out, txt_out

        out = F.scaled_dot_product_attention(q_i, k_i, v_i).transpose(1, 2).reshape(bsz, img_len, dim)
        return self.to_out[1](self.to_out[0](out))


class _MiniDoubleStreamBlock(nn.Module):
    """Double-stream MMDiT block with AdaLN-Zero timestep modulation and FFN."""

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_context = nn.LayerNorm(dim)
        self.attn = _MiniDoubleStreamAttention(dim=dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2_context = nn.LayerNorm(dim)
        self.ff_context = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.time_mod = nn.Linear(dim, dim * 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_appearance_states: Optional[torch.Tensor] = None,
        cond_structure_states: Optional[torch.Tensor] = None,
        cond_temb: Optional[torch.Tensor] = None,
        cond_appearance_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_structure_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        appearance_scale: Optional[float] = None,
        structure_scale: Optional[float] = None,
        cache_mode: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        shift, scale = self.time_mod(F.silu(temb)).chunk(2, dim=-1)
        norm_img = self.norm1(hidden_states) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        norm_txt = self.norm1_context(encoder_hidden_states)

        norm_cond_a: Optional[torch.Tensor] = None
        norm_cond_s: Optional[torch.Tensor] = None
        if cache_mode != "read":
            c_shift, c_scale = (
                self.time_mod(F.silu(cond_temb)).chunk(2, dim=-1)
                if cond_temb is not None
                else (torch.zeros_like(shift), torch.zeros_like(scale))
            )
            if cond_appearance_states is not None:
                norm_cond_a = self.norm1(cond_appearance_states) * (1.0 + c_scale.unsqueeze(1)) + c_shift.unsqueeze(1)
            if cond_structure_states is not None:
                norm_cond_s = self.norm1(cond_structure_states) * (1.0 + c_scale.unsqueeze(1)) + c_shift.unsqueeze(1)

        want_cond_updates = cache_mode != "read" and (norm_cond_a is not None or norm_cond_s is not None)
        attn_res = self.attn(
            hidden_states=norm_img,
            encoder_hidden_states=norm_txt,
            image_rotary_emb=image_rotary_emb,
            cond_appearance_states=norm_cond_a,
            cond_structure_states=norm_cond_s,
            cond_appearance_rotary_emb=cond_appearance_rotary_emb,
            cond_structure_rotary_emb=cond_structure_rotary_emb,
            appearance_scale=appearance_scale,
            structure_scale=structure_scale,
            cache_mode=cache_mode,
            return_cond_outputs=want_cond_updates,
        )

        updated_conds: Dict[str, torch.Tensor] = {}
        if want_cond_updates and isinstance(attn_res, tuple) and len(attn_res) == 3:
            img_attn, txt_attn, updated_conds = attn_res
        else:
            img_attn, txt_attn = attn_res

        hidden_states = hidden_states + img_attn
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))

        encoder_hidden_states = encoder_hidden_states + txt_attn
        encoder_hidden_states = encoder_hidden_states + self.ff_context(self.norm2_context(encoder_hidden_states))

        next_cond_a = cond_appearance_states
        next_cond_s = cond_structure_states
        if "appearance" in updated_conds and cond_appearance_states is not None:
            next_cond_a = cond_appearance_states + updated_conds["appearance"]
            next_cond_a = next_cond_a + self.ff(self.norm2(next_cond_a))
        if "structure" in updated_conds and cond_structure_states is not None:
            next_cond_s = cond_structure_states + updated_conds["structure"]
            next_cond_s = next_cond_s + self.ff(self.norm2(next_cond_s))
        if "merged" in updated_conds and cond_appearance_states is not None:
            merged_next = cond_appearance_states + updated_conds["merged"]
            merged_next = merged_next + self.ff(self.norm2(merged_next))
            next_cond_a = merged_next
            next_cond_s = merged_next

        return encoder_hidden_states, hidden_states, next_cond_a, next_cond_s


class _MiniSingleStreamBlock(nn.Module):
    """Single-stream DiT block operating on concatenated [txt, img] tokens."""

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return x + self.proj_out(attn) + self.mlp(h)


class StandaloneMMDiT(nn.Module):
    """
    Self-contained FLUX.2-compatible MMDiT architecture with `transformer_blocks` (double-stream)
    and `single_transformer_blocks` (single-stream) for fast local execution and testing.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 256,
        joint_attention_dim: int = 256,
        num_heads: int = 8,
        num_double_blocks: int = 4,
        num_single_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.joint_attention_dim = joint_attention_dim
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_channels, hidden_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transformer_blocks = nn.ModuleList(
            [_MiniDoubleStreamBlock(dim=hidden_dim, heads=num_heads) for _ in range(num_double_blocks)]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [_MiniSingleStreamBlock(dim=hidden_dim, heads=num_heads) for _ in range(num_single_blocks)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, in_channels)

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(1, half)
        )
        args = t.float().unsqueeze(-1) * 1000.0 * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.x_embedder.weight.dtype)
        return self.time_mlp(emb)

    def _build_rope(
        self,
        ids: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.hidden_dim // self.num_heads
        if ids is None:
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        else:
            if ids.ndim == 3:
                ids = ids[0]
            pos = ids[:, 1].float() * 128.0 + ids[:, 2].float()
        half = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(1, half)))
        angles = torch.outer(pos, inv_freq)
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(dtype)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(dtype)
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        cond_appearance_latents: Optional[torch.Tensor] = None,
        cond_structure_latents: Optional[torch.Tensor] = None,
        cond_img_ids: Optional[torch.Tensor] = None,
        appearance_scale: Optional[float] = None,
        structure_scale: Optional[float] = None,
        cache_mode: str = "none",
        **kwargs: Any,
    ) -> torch.Tensor:
        dtype = self.x_embedder.weight.dtype
        h_img = self.x_embedder(hidden_states.to(dtype))
        h_txt = self.context_embedder(encoder_hidden_states.to(dtype))
        temb = self._timestep_embedding(timestep)

        cond_a = (
            self.x_embedder(cond_appearance_latents.to(dtype))
            if (cond_appearance_latents is not None and cache_mode != "read")
            else None
        )
        cond_s = (
            self.x_embedder(cond_structure_latents.to(dtype))
            if (cond_structure_latents is not None and cache_mode != "read")
            else None
        )
        cond_temb = self._timestep_embedding(torch.zeros_like(timestep)) if (cond_a is not None or cond_s is not None) else None

        txt_len = h_txt.shape[1]
        img_len = h_img.shape[1]
        main_ids = (
            torch.cat([txt_ids, img_ids], dim=-2)
            if (txt_ids is not None and img_ids is not None)
            else None
        )
        main_rope = self._build_rope(main_ids, txt_len + img_len, hidden_states.device, dtype)
        cond_rope = (
            self._build_rope(cond_img_ids if cond_img_ids is not None else img_ids, img_len, hidden_states.device, dtype)
            if (cond_a is not None or cond_s is not None)
            else None
        )

        for block in self.transformer_blocks:
            h_txt, h_img, cond_a, cond_s = block(
                hidden_states=h_img,
                encoder_hidden_states=h_txt,
                temb=temb,
                image_rotary_emb=main_rope,
                cond_appearance_states=cond_a,
                cond_structure_states=cond_s,
                cond_temb=cond_temb,
                cond_appearance_rotary_emb=cond_rope,
                cond_structure_rotary_emb=cond_rope,
                appearance_scale=appearance_scale,
                structure_scale=structure_scale,
                cache_mode=cache_mode,
            )

        joint = torch.cat([h_txt, h_img], dim=1)
        for s_block in self.single_transformer_blocks:
            joint = s_block(joint)

        h_out = joint[:, txt_len:]
        return self.proj_out(self.norm_out(h_out))


class StandaloneVAE(nn.Module):
    """Lightweight f=8 convolutional VAE matching AutoencoderKLFlux2 latent interface."""

    def __init__(self, latent_channels: int = 16, vae_scale_factor: int = 8) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.vae_scale_factor = vae_scale_factor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.to(self.encoder[0].weight.dtype))

    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z.to(self.decoder[0].weight.dtype))


# ---------------------------------------------------------------------------
# Latent Packing & RoPE Position ID Utilities (FLUX / FLUX.2 2x2 Patchify)
# ---------------------------------------------------------------------------


def pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Pack spatial VAE latents (B, C, H_lat, W_lat) into sequence tokens
    (B, (H_lat/p)*(W_lat/p), C * p * p).
    At 1280x720 with f=8 and p=2: H_lat=90, W_lat=160 -> 45 * 80 = 3,600 tokens!
    """
    bsz, ch, h, w = latents.shape
    p = patch_size
    latents = latents.view(bsz, ch, h // p, p, w // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    return latents.view(bsz, (h // p) * (w // p), ch * p * p)


def unpack_latents(
    packed: torch.Tensor,
    latent_height: int,
    latent_width: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """Unpack sequence tokens (B, L, C * p * p) back into (B, C, H_lat, W_lat)."""
    bsz, _, packed_ch = packed.shape
    p = patch_size
    ch = packed_ch // (p * p)
    hp, wp = latent_height // p, latent_width // p
    latents = packed.view(bsz, hp, wp, ch, p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
    return latents.view(bsz, ch, latent_height, latent_width)


def prepare_latent_image_ids(
    batch_size: int,
    packed_height: int,
    packed_width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    stream_id: float = 0.0,
) -> torch.Tensor:
    """
    Generate 3D RoPE position IDs `(stream_id, row_y, col_x)` of shape `(B, H_p * W_p, 3)`.
    Notice that noisy latent z_t, canvas z_C, and depth z_{D_s} share the EXACT `(row_y, col_x)`
    grid coordinates so geometric and appearance conditioning align pixel-for-pixel!
    """
    ids = torch.zeros((packed_height, packed_width, 3), device=device, dtype=dtype)
    ids[..., 0] = stream_id
    ids[..., 1] = torch.arange(packed_height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(packed_width, device=device, dtype=dtype)[None, :]
    ids = ids.reshape(1, packed_height * packed_width, 3).repeat(batch_size, 1, 1)
    return ids


# ---------------------------------------------------------------------------
# Production FLUX.2 Dev Backbone Wrapper
# ---------------------------------------------------------------------------


class Flux2BackboneWrapper(nn.Module):
    """
    Unified FLUX.2 Dev Diffusion Backbone Wrapper supporting:
      - Real `black-forest-labs/FLUX.2-dev` via HuggingFace `diffusers` (`Flux2Pipeline`,
        `Flux2Transformer2DModel`, `AutoencoderKLFlux2`).
      - Standalone `StandaloneMMDiT` mode (`use_mock_backbone=True`) for rapid CPU/GPU testing.
      - Frozen base weights + injected `DSAConditionInjectionProcessor` (rank r=32).
      - T=40 Rectified-Flow Euler integration with Step-0 `bank_kv` KV caching.
    """

    def __init__(
        self,
        cfg: RefComposeConfig,
        device: Optional[torch.device] = None,
        use_mock_backbone: bool = False,
        inject_lora: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device_obj = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock_backbone = use_mock_backbone
        self.dtype = cfg.model.torch_dtype if self.device_obj.type == "cuda" else torch.float32

        self.pipe: Optional[Any] = None
        if use_mock_backbone:
            latent_ch = 16
            patch_sq = cfg.resolution.patch_size ** 2
            self.vae = StandaloneVAE(
                latent_channels=latent_ch,
                vae_scale_factor=cfg.resolution.vae_scale_factor,
            ).to(device=self.device_obj, dtype=self.dtype)
            self.transformer = StandaloneMMDiT(
                in_channels=latent_ch * patch_sq,
                hidden_dim=256,
                joint_attention_dim=256,
                num_heads=8,
                num_double_blocks=4,
                num_single_blocks=2,
            ).to(device=self.device_obj, dtype=self.dtype)
            self.latent_channels = latent_ch
        else:
            from diffusers import AutoencoderKLFlux2, Flux2Pipeline, Flux2Transformer2DModel

            repo_id = cfg.model.flux2_repo_id
            self.transformer = Flux2Transformer2DModel.from_pretrained(
                repo_id,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            self.vae = AutoencoderKLFlux2.from_pretrained(
                repo_id,
                subfolder="vae",
                torch_dtype=self.dtype,
            )
            self.pipe = Flux2Pipeline.from_pretrained(
                repo_id,
                transformer=self.transformer,
                vae=self.vae,
                torch_dtype=self.dtype,
            )
            if cfg.model.enable_cpu_offload and self.device_obj.type == "cuda":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device_obj)

            in_ch = getattr(self.transformer.config, "in_channels", 128)
            self.latent_channels = in_ch // (cfg.resolution.patch_size ** 2)

        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False

        if inject_lora:
            inject_dsa_lora(self.transformer, self.cfg.lora)
            self.transformer.to(device=self.device_obj, dtype=self.dtype)

    @torch.no_grad()
    def encode_image_to_packed_latents(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode an image/canvas/depth tensor in R^{B x 3 x H x W} (normalized to [-1, 1])
        via frozen VAE E(.) and pack into patchified sequence tokens (Eq. 4 & Eq. 7).
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        x = image_tensor.to(device=self.device_obj, dtype=self.dtype)

        if self.use_mock_backbone:
            z = self.vae.encode_tensor(x)
        else:
            posterior = self.vae.encode(x)
            z = posterior.latent_dist.sample() if hasattr(posterior, "latent_dist") else posterior.latents
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0
            z = (z - shift_factor) * scaling_factor

        return pack_latents(z, patch_size=self.cfg.resolution.patch_size)

    @torch.no_grad()
    def decode_packed_latents_to_pil(
        self,
        packed_latents: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> List[Image.Image]:
        """Decode packed latents z_hat_0 via E^{-1}(z_hat_0) into RGB PIL Images (Eq. 11)."""
        h = height or self.cfg.resolution.height
        w = width or self.cfg.resolution.width
        lat_h = h // self.cfg.resolution.vae_scale_factor
        lat_w = w // self.cfg.resolution.vae_scale_factor

        z = unpack_latents(
            packed_latents,
            latent_height=lat_h,
            latent_width=lat_w,
            patch_size=self.cfg.resolution.patch_size,
        )

        if self.use_mock_backbone:
            decoded = self.vae.decode_tensor(z)
        else:
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0
            z = (z / scaling_factor) + shift_factor
            dec_out = self.vae.decode(z, return_dict=False)
            decoded = dec_out[0] if isinstance(dec_out, tuple) else dec_out

        return [tensor_to_pil(decoded[b], is_minus1_to_1=True) for b in range(decoded.shape[0])]

    @torch.no_grad()
    def encode_prompt(
        self,
        prompts: str | List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt(s) `p` into `(prompt_embeds, txt_ids)`."""
        if isinstance(prompts, str):
            prompts = [prompts]
        bsz = len(prompts)

        if self.use_mock_backbone:
            # Deterministic token-hash text embedding for standalone testing
            txt_len = 32
            dim = self.transformer.joint_attention_dim
            embeds = torch.zeros((bsz, txt_len, dim), device=self.device_obj, dtype=self.dtype)
            for b_idx, text in enumerate(prompts):
                words = text.split()[:txt_len]
                for t_idx, word in enumerate(words):
                    seed_val = sum(ord(ch) * (i + 1) for i, ch in enumerate(word)) % 9973
                    gen = torch.Generator(device="cpu").manual_seed(seed_val)
                    embeds[b_idx, t_idx] = torch.randn((dim,), generator=gen).to(
                        device=self.device_obj, dtype=self.dtype
                    ) * 0.2
            txt_ids = torch.zeros((bsz, txt_len, 3), device=self.device_obj, dtype=torch.float32)
            txt_ids[..., 2] = torch.arange(txt_len, device=self.device_obj, dtype=torch.float32)
            return embeds, txt_ids

        assert self.pipe is not None
        prompt_out = self.pipe.encode_prompt(
            prompt=prompts,
            device=self.device_obj,
            num_images_per_prompt=1,
        )
        prompt_embeds = prompt_out[0]
        txt_ids = (
            prompt_out[1]
            if len(prompt_out) > 1 and isinstance(prompt_out[1], torch.Tensor)
            else torch.zeros(
                (bsz, prompt_embeds.shape[1], 3), device=self.device_obj, dtype=torch.float32
            )
        )
        return prompt_embeds, txt_ids

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        z_canvas: Optional[torch.Tensor] = None,
        z_depth: Optional[torch.Tensor] = None,
        appearance_scale: Optional[float] = None,
        structure_scale: Optional[float] = None,
        cache_mode: str = "none",
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Evaluate v_theta(z_t, t, p, z_C, z_{D_s}; Delta W_s + Delta W_a) (Paper Eq. 10).
        """
        bsz = z_t.shape[0]
        g_val = guidance_scale if guidance_scale is not None else self.cfg.sampling.guidance_scale
        guidance = torch.full((bsz,), g_val, device=z_t.device, dtype=self.dtype)

        if self.use_mock_backbone:
            return self.transformer(
                hidden_states=z_t,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                cond_appearance_latents=z_canvas,
                cond_structure_latents=z_depth,
                cond_img_ids=img_ids,
                appearance_scale=appearance_scale,
                structure_scale=structure_scale,
                cache_mode=cache_mode,
            )

        # Production FLUX.2 Transformer forward pass with DSA joint_attention_kwargs
        # On step 0 / training (`cache_mode != "read"`), project z_canvas & z_depth via frozen x_embedder
        cond_a_states: Optional[torch.Tensor] = None
        cond_s_states: Optional[torch.Tensor] = None
        cond_rope = None
        if cache_mode != "read":
            if z_canvas is not None:
                cond_a_states = self.transformer.x_embedder(z_canvas.to(self.dtype))
            if z_depth is not None:
                cond_s_states = self.transformer.x_embedder(z_depth.to(self.dtype))
            if hasattr(self.transformer, "pos_embed") and (cond_a_states is not None or cond_s_states is not None):
                cond_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
                cond_rope = self.transformer.pos_embed(cond_ids)

        joint_kwargs = {
            "cond_appearance_states": cond_a_states,
            "cond_structure_states": cond_s_states,
            "cond_appearance_rotary_emb": cond_rope,
            "cond_structure_rotary_emb": cond_rope,
            "appearance_scale": appearance_scale,
            "structure_scale": structure_scale,
            "cache_mode": cache_mode,
        }

        out = self.transformer(
            hidden_states=z_t.to(self.dtype),
            encoder_hidden_states=prompt_embeds.to(self.dtype),
            timestep=timestep.to(self.dtype),
            img_ids=img_ids[0] if img_ids.ndim == 3 else img_ids,
            txt_ids=txt_ids[0] if txt_ids.ndim == 3 else txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_kwargs,
            return_dict=False,
        )
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def sample_flow_matching(
        self,
        prompt: str | List[str],
        z_canvas: Optional[torch.Tensor] = None,
        z_depth: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        appearance_scale: Optional[float] = None,
        structure_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Rectified-Flow Euler integration over T=40 steps starting from z_T ~ N(0, I) (Eq. 9-11):
          - Step 0 populates `bank_kv` (`cache_mode="write"`)
          - Steps 1..T-1 read cached condition K, V (`cache_mode="read"`)
          - Clears `bank_kv` upon completion.
        """
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        bsz = len(prompts)
        h = height or self.cfg.resolution.height
        w = width or self.cfg.resolution.width
        steps = num_steps or self.cfg.sampling.num_inference_steps

        lat_h = h // self.cfg.resolution.vae_scale_factor
        lat_w = w // self.cfg.resolution.vae_scale_factor
        pack_h = lat_h // self.cfg.resolution.patch_size
        pack_w = lat_w // self.cfg.resolution.patch_size
        seq_len = pack_h * pack_w
        packed_ch = self.latent_channels * (self.cfg.resolution.patch_size ** 2)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.cfg.sampling.seed if seed is None else seed)

        z_t = torch.randn((bsz, seq_len, packed_ch), generator=gen, dtype=torch.float32).to(
            device=self.device_obj, dtype=self.dtype
        )

        prompt_embeds, txt_ids = self.encode_prompt(prompts)
        img_ids = prepare_latent_image_ids(
            batch_size=bsz,
            packed_height=pack_h,
            packed_width=pack_w,
            device=self.device_obj,
        )

        # Schedule t from 1.0 -> 0.0 across T steps (z_t = (1 - t) z_0 + t z_1)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device_obj, dtype=torch.float32)

        has_conds = (
            (z_canvas is not None and (appearance_scale is None or appearance_scale != 0.0))
            or (z_depth is not None and (structure_scale is None or structure_scale != 0.0))
        )
        clear_kv_cache(self.transformer)

        try:
            for step_idx in range(steps):
                t_curr = timesteps[step_idx]
                t_next = timesteps[step_idx + 1]
                dt = t_curr - t_next  # positive step towards t=0

                t_batch = torch.full((bsz,), float(t_curr), device=self.device_obj, dtype=self.dtype)
                if has_conds and self.cfg.lora.use_kv_cache:
                    cache_mode = "write" if step_idx == 0 else "read"
                else:
                    cache_mode = "none"

                v_pred = self.predict_velocity(
                    z_t=z_t,
                    timestep=t_batch,
                    prompt_embeds=prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    z_canvas=z_canvas,
                    z_depth=z_depth,
                    appearance_scale=appearance_scale,
                    structure_scale=structure_scale,
                    cache_mode=cache_mode,
                    guidance_scale=guidance_scale,
                )
                # Since z_t = (1 - t) z_0 + t z_1 and v = z_1 - z_0 = dz_t / dt:
                # z_{t - dt} = z_t - dt * v_theta
                z_t = z_t - dt * v_pred
        finally:
            clear_kv_cache(self.transformer)

        return self.decode_packed_latents_to_pil(z_t, height=h, width=w)

    @torch.no_grad()
    def generate_layout_image(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Stage 1 (Section 3.1 Eq. 1): Prompt-induced coarse layout generation L = F_theta(p)
        over T=40 denoising steps using the frozen base transformer (LoRA scales = 0).
        """
        imgs = self.sample_flow_matching(
            prompt=prompt,
            z_canvas=None,
            z_depth=None,
            height=height,
            width=width,
            num_steps=self.cfg.sampling.layout_inference_steps,
            appearance_scale=0.0,
            structure_scale=0.0,
            seed=seed,
        )
        return imgs[0]

    @torch.no_grad()
    def canonicalize_crop(
        self,
        crop: Image.Image,
        object_description: str,
        seed: int = 42,
    ) -> Image.Image:
        """
        Synthesize a neutral-pose, neutral-lighting canonicalization of a detected object crop
        using FLUX.2 Dev (Section 4.1 Stage 3 & Section 4.4 Evaluation Protocol).
        """
        target_w, target_h = self.cfg.curation.canonical_resolution
        prompt = self.cfg.curation.canonical_prompt_template.format(
            object_description=object_description
        )
        if self.use_mock_backbone or self.pipe is None:
            # Deterministic clean studio-padded crop canonicalization in standalone mode
            rgb = crop.convert("RGB")
            bg = Image.new("RGB", (target_w, target_h), color=(235, 235, 235))
            scale = min(target_w * 0.8 / max(1, rgb.width), target_h * 0.8 / max(1, rgb.height))
            nw, nh = max(8, int(round(rgb.width * scale))), max(8, int(round(rgb.height * scale)))
            resized = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
            bg.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
            return bg

        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Flux2Pipeline supports image-conditioned editing via `image=`
        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "height": target_h,
            "width": target_w,
            "num_inference_steps": self.cfg.curation.canonical_steps,
            "guidance_scale": self.cfg.curation.canonical_guidance_scale,
            "generator": gen,
        }
        if "image" in self.pipe.__call__.__code__.co_varnames:
            call_kwargs["image"] = crop.convert("RGB").resize((target_w, target_h), Image.Resampling.BICUBIC)

        return self.pipe(**call_kwargs).images[0]
