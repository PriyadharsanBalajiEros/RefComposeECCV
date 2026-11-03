"""
Decoupled Structure-Appearance (DSA) LoRA & EasyControl Condition Injection
(Paper Section 3.4 Eq. 8-12 & Section 4.3).

Key Properties:
  1. Frozen base MMDiT weights W in R^{d_out x d_in}.
  2. Two orthogonal rank-r (r=32) LoRA branches in double-stream attention blocks:
       Delta W_s = B_s A_s  (Structure / Depth z_{D_s})
       Delta W_a = B_a A_a  (Appearance / Reference Canvas z_C)
  3. Subject/Appearance LoRA ('appearance') is ordered before Depth/Structure LoRA ('structure').
  4. Block-causal attention mask isolates `appearance` and `structure` condition streams from
     each other and from time-varying noisy latents z_t, preventing depth/appearance leakage.
  5. `bank_kv` KV caching stores condition (K_a, V_a) and (K_s, V_s) on denoising step 0 and
     reuses them across remaining T-1 flow-matching steps at zero extra condition cost.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
except ImportError:  # pragma: no cover
    safetensors_load = None
    safetensors_save = None

from refcompose.config import DSALoRAConfig


class LoRABranch(nn.Module):
    """
    Single low-rank update branch Delta W_m = (alpha / r) * B_m @ A_m (Paper Eq. 8):
        A_m in R^{r x d_in},  B_m in R^{d_out x r}
    Initialized with Kaiming-uniform A_m and zero B_m so Delta W_m = 0 at step 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(max(1, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        lora_dtype = self.lora_A.weight.dtype
        out = self.lora_B(self.lora_A(self.dropout(x.to(lora_dtype)))) * self.scaling
        return out.to(orig_dtype)


class DSALoRALinear(nn.Module):
    """
    Frozen base linear layer W augmented with decoupled Structure (s) and Appearance (a)
    LoRA branches:
      - Base stream (text / noisy image z_t):         y = W x
      - Structure stream (depth z_{D_s}):             y = (W + scale_s * Delta W_s) x
      - Appearance stream (reference canvas z_C):     y = (W + scale_a * Delta W_a) x
      - Combined projection / merged-stream ablation: y = (W + scale_s * Delta W_s + scale_a * Delta W_a) x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        self.appearance_lora = LoRABranch(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=dtype,
            device=device,
        )
        self.structure_lora = LoRABranch(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=dtype,
            device=device,
        )

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen base projection W x for noisy latent z_t or text tokens."""
        return self.base_layer(x)

    def forward_branch(
        self,
        x: torch.Tensor,
        branch: str,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Condition-specific projection (W + scale * Delta W_m) x for m in {'appearance', 'structure'}."""
        base_out = self.base_layer(x)
        if scale == 0.0:
            return base_out
        if branch == "appearance":
            return base_out + scale * self.appearance_lora(x)
        if branch == "structure":
            return base_out + scale * self.structure_lora(x)
        raise ValueError(f"Unknown DSA LoRA branch '{branch}'. Expected 'appearance' or 'structure'.")

    def forward_combined(
        self,
        x: torch.Tensor,
        structure_scale: float = 1.0,
        appearance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Effective combined projection (W + scale_s * Delta W_s + scale_a * Delta W_a) x (Eq. 8)."""
        out = self.base_layer(x)
        if appearance_scale != 0.0:
            out = out + appearance_scale * self.appearance_lora(x)
        if structure_scale != 0.0:
            out = out + structure_scale * self.structure_lora(x)
        return out


def build_dsa_causal_attention_mask(
    txt_len: int,
    img_len: int,
    cond_lengths: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build the EasyControl block-causal boolean attention mask for SDPA (`True` = allowed to attend):
      Sequence layout: [txt (L_txt), img (L_img), cond_0 (L_0), cond_1 (L_1), ...]
      - Main generation queries (txt + img) attend to ALL tokens (txt, img, and all active conditions).
      - Each condition stream `cond_i` attends ONLY to itself (diagonal block), preventing:
          (a) cross-condition leakage between depth z_{D_s} and canvas z_C, and
          (b) dependence on noisy z_t, enabling exact KV caching across all denoising steps!
    Returns boolean mask of shape (1, 1, total_len, total_len).
    """
    main_len = txt_len + img_len
    total_len = main_len + sum(cond_lengths)
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

    # Main generation tokens (txt + img) attend everywhere
    mask[:main_len, :total_len] = True

    # Each condition stream attends strictly within its own block
    offset = main_len
    for c_len in cond_lengths:
        if c_len > 0:
            mask[offset : offset + c_len, offset : offset + c_len] = True
            offset += c_len

    return mask.unsqueeze(0).unsqueeze(0)


def _apply_rotary_emb_compat(
    tensor: torch.Tensor,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Apply FLUX / FLUX.2 rotary position embeddings (cos, sin) to Q or K of shape (B, H, L, D).
    Compatible with `diffusers.models.embeddings.apply_rotary_emb` conventions.
    """
    if freqs_cis is None:
        return tensor
    cos, sin = freqs_cis
    # Broadcast cos, sin across batch and head axes -> (1, 1, L, D) or (B, 1, L, D)
    while cos.ndim < tensor.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(device=tensor.device, dtype=tensor.dtype)
    sin = sin.to(device=tensor.device, dtype=tensor.dtype)

    x_real, x_imag = tensor.reshape(*tensor.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (tensor.float() * cos.float() + x_rotated.float() * sin.float()).to(tensor.dtype)


class DSAConditionInjectionProcessor(nn.Module):
    """
    EasyControl-extended Decoupled Structure-Appearance (DSA) Attention Processor for
    FLUX.2 Double-Stream MMDiT blocks (`transformer_blocks.*.attn`).

    Supports:
      1. Training (`use_kv_cache=False` or `cache_mode="none"`):
         Joint block-causal attention over [txt, img, cond_appearance, cond_structure]
         updating condition streams cleanly for downstream double-stream blocks.
      2. Inference Step 0 (`cache_mode="write"`):
         Computes condition (K_a, V_a) and (K_s, V_s), stores them in `self.bank_kv`,
         and returns updated condition hidden states for deeper blocks.
      3. Inference Steps 1..T-1 (`cache_mode="read"`):
         Reads cached (K_a, V_a, K_s, V_s) directly from `self.bank_kv` and runs
         attention only for [Q_txt, Q_img], keeping multi-reference conditioning fast and constant-memory.
      4. Merged single-stream ablation (`merged_single_stream=True`):
         Routes summed/merged condition through combined (Delta W_a + Delta W_s) to reproduce
         the Section 4.5 / 4.6 cross-signal overwriting ablation.
    """

    def __init__(
        self,
        attn_module: nn.Module,
        lora_cfg: Optional[DSALoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = lora_cfg or DSALoRAConfig()
        self.bank_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Wrap image-stream Q, K, V, Out projections with DSALoRALinear
        self.q_dsa = DSALoRALinear(
            attn_module.to_q,
            rank=self.cfg.rank,
            alpha=self.cfg.alpha,
            dropout=self.cfg.dropout,
        )
        self.k_dsa = DSALoRALinear(
            attn_module.to_k,
            rank=self.cfg.rank,
            alpha=self.cfg.alpha,
            dropout=self.cfg.dropout,
        )
        self.v_dsa = DSALoRALinear(
            attn_module.to_v,
            rank=self.cfg.rank,
            alpha=self.cfg.alpha,
            dropout=self.cfg.dropout,
        )

        out_linear = (
            attn_module.to_out[0]
            if isinstance(attn_module.to_out, (nn.ModuleList, nn.Sequential))
            else attn_module.to_out
        )
        self.out_dsa = DSALoRALinear(
            out_linear,
            rank=self.cfg.rank,
            alpha=self.cfg.alpha,
            dropout=self.cfg.dropout,
        )

    def clear_cache(self) -> None:
        """Clear cached conditional (K, V) tensors in `bank_kv`."""
        self.bank_kv.clear()

    @staticmethod
    def _reshape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        head_dim = dim // heads
        return x.view(bsz, seq_len, heads, head_dim).transpose(1, 2)

    @staticmethod
    def _apply_qk_norm(
        q: torch.Tensor,
        k: torch.Tensor,
        norm_q: Optional[nn.Module],
        norm_k: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if norm_q is not None:
            q = norm_q(q)
        if norm_k is not None:
            k = norm_k(k)
        return q, k

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_appearance_states: Optional[torch.Tensor] = None,
        cond_structure_states: Optional[torch.Tensor] = None,
        cond_appearance_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_structure_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        appearance_scale: Optional[float] = None,
        structure_scale: Optional[float] = None,
        cache_mode: str = "none",  # "none" | "write" | "read"
        **kwargs: Any,
    ) -> Any:
        bsz, img_len, _ = hidden_states.shape
        heads = attn.heads

        scale_a = self.cfg.appearance_scale if appearance_scale is None else appearance_scale
        scale_s = self.cfg.structure_scale if structure_scale is None else structure_scale

        # 1. Noisy image latent projections (frozen W)
        q_img = self._reshape_heads(self.q_dsa(hidden_states), heads)
        k_img = self._reshape_heads(self.k_dsa(hidden_states), heads)
        v_img = self._reshape_heads(self.v_dsa(hidden_states), heads)
        q_img, k_img = self._apply_qk_norm(
            q_img,
            k_img,
            getattr(attn, "norm_q", None),
            getattr(attn, "norm_k", None),
        )

        # 2. Text conditioning projections (frozen W_add)
        txt_len = 0
        if encoder_hidden_states is not None and hasattr(attn, "add_q_proj") and attn.add_q_proj is not None:
            txt_len = encoder_hidden_states.shape[1]
            q_txt = self._reshape_heads(attn.add_q_proj(encoder_hidden_states), heads)
            k_txt = self._reshape_heads(attn.add_k_proj(encoder_hidden_states), heads)
            v_txt = self._reshape_heads(attn.add_v_proj(encoder_hidden_states), heads)
            q_txt, k_txt = self._apply_qk_norm(
                q_txt,
                k_txt,
                getattr(attn, "norm_added_q", None),
                getattr(attn, "norm_added_k", None),
            )
            q_main = torch.cat([q_txt, q_img], dim=2)
            k_main = torch.cat([k_txt, k_img], dim=2)
            v_main = torch.cat([v_txt, v_img], dim=2)
        else:
            q_main, k_main, v_main = q_img, k_img, v_img

        # Apply RoPE to [txt, img]
        if image_rotary_emb is not None:
            q_main = _apply_rotary_emb_compat(q_main, image_rotary_emb)
            k_main = _apply_rotary_emb_compat(k_main, image_rotary_emb)

        # 3. Fast path: Inference steps 1..T-1 reading pre-cached condition K, V from `bank_kv`
        if cache_mode == "read" and self.cfg.use_kv_cache and len(self.bank_kv) > 0:
            k_list = [k_main]
            v_list = [v_main]
            for stream_name in self.cfg.stream_order:
                if stream_name == "appearance" and scale_a == 0.0:
                    continue
                if stream_name == "structure" and scale_s == 0.0:
                    continue
                if stream_name in self.bank_kv:
                    cached_k, cached_v = self.bank_kv[stream_name]
                    k_list.append(cached_k)
                    v_list.append(cached_v)

            k_all = torch.cat(k_list, dim=2)
            v_all = torch.cat(v_list, dim=2)

            attn_out = F.scaled_dot_product_attention(
                q_main, k_all, v_all, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).reshape(bsz, txt_len + img_len, -1).to(hidden_states.dtype)

            if txt_len > 0:
                txt_out, img_out = attn_out[:, :txt_len], attn_out[:, txt_len:]
                img_out = self.out_dsa(img_out)
                if len(attn.to_out) > 1:
                    img_out = attn.to_out[1](img_out)
                txt_out = attn.to_add_out(txt_out)
                return img_out, txt_out
            img_out = self.out_dsa(attn_out)
            return img_out

        # 4. Full condition stream processing (Training or Step-0 Cache Write)
        # Paper Section 4.3: Subject/Appearance stream ordered before Depth/Structure stream
        active_streams: List[Tuple[str, torch.Tensor, float, Optional[Tuple[torch.Tensor, torch.Tensor]]]] = []

        if self.cfg.merged_single_stream_ablation:
            # Ablation mode (Paper p. 12): merge depth + canvas into a single combined LoRA stream
            merged_cond: Optional[torch.Tensor] = None
            if cond_appearance_states is not None and scale_a != 0.0:
                merged_cond = cond_appearance_states
            if cond_structure_states is not None and scale_s != 0.0:
                merged_cond = (
                    cond_structure_states if merged_cond is None else 0.5 * (merged_cond + cond_structure_states)
                )
            if merged_cond is not None:
                active_streams.append(
                    ("merged", merged_cond, 1.0, cond_appearance_rotary_emb or cond_structure_rotary_emb)
                )
        else:
            for stream_name in self.cfg.stream_order:
                if stream_name == "appearance" and cond_appearance_states is not None and scale_a != 0.0:
                    active_streams.append(
                        ("appearance", cond_appearance_states, scale_a, cond_appearance_rotary_emb)
                    )
                elif stream_name == "structure" and cond_structure_states is not None and scale_s != 0.0:
                    active_streams.append(
                        ("structure", cond_structure_states, scale_s, cond_structure_rotary_emb)
                    )

        if not active_streams:
            attn_out = F.scaled_dot_product_attention(
                q_main, k_main, v_main, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).reshape(bsz, txt_len + img_len, -1).to(hidden_states.dtype)
            if txt_len > 0:
                txt_out, img_out = attn_out[:, :txt_len], attn_out[:, txt_len:]
                img_out = self.out_dsa(img_out)
                if isinstance(attn.to_out, (nn.ModuleList, nn.Sequential)) and len(attn.to_out) > 1:
                    img_out = attn.to_out[1](img_out)
                txt_out = attn.to_add_out(txt_out)
                return img_out, txt_out
            return self.out_dsa(attn_out)

        q_parts = [q_main]
        k_parts = [k_main]
        v_parts = [v_main]
        cond_lengths: List[int] = []

        for stream_name, cond_states, stream_scale, cond_rope in active_streams:
            c_len = cond_states.shape[1]
            cond_lengths.append(c_len)

            if stream_name == "merged":
                qc = self._reshape_heads(
                    self.q_dsa.forward_combined(cond_states, scale_s, scale_a), heads
                )
                kc = self._reshape_heads(
                    self.k_dsa.forward_combined(cond_states, scale_s, scale_a), heads
                )
                vc = self._reshape_heads(
                    self.v_dsa.forward_combined(cond_states, scale_s, scale_a), heads
                )
            else:
                qc = self._reshape_heads(
                    self.q_dsa.forward_branch(cond_states, stream_name, stream_scale), heads
                )
                kc = self._reshape_heads(
                    self.k_dsa.forward_branch(cond_states, stream_name, stream_scale), heads
                )
                vc = self._reshape_heads(
                    self.v_dsa.forward_branch(cond_states, stream_name, stream_scale), heads
                )

            qc, kc = self._apply_qk_norm(
                qc,
                kc,
                getattr(attn, "norm_q", None),
                getattr(attn, "norm_k", None),
            )
            if cond_rope is not None:
                qc = _apply_rotary_emb_compat(qc, cond_rope)
                kc = _apply_rotary_emb_compat(kc, cond_rope)

            if cache_mode == "write" and self.cfg.use_kv_cache and stream_name in ("appearance", "structure"):
                self.bank_kv[stream_name] = (kc.detach(), vc.detach())

            q_parts.append(qc)
            k_parts.append(kc)
            v_parts.append(vc)

        q_full = torch.cat(q_parts, dim=2)
        k_full = torch.cat(k_parts, dim=2)
        v_full = torch.cat(v_parts, dim=2)

        dsa_mask: Optional[torch.Tensor] = attention_mask
        if self.cfg.use_causal_mask:
            dsa_mask = build_dsa_causal_attention_mask(
                txt_len=txt_len,
                img_len=img_len,
                cond_lengths=cond_lengths,
                device=hidden_states.device,
            )

        full_out = F.scaled_dot_product_attention(
            q_full, k_full, v_full, attn_mask=dsa_mask, dropout_p=0.0, is_causal=False
        )
        full_out = full_out.transpose(1, 2).reshape(bsz, q_full.shape[2], -1).to(hidden_states.dtype)

        main_len = txt_len + img_len
        main_out = full_out[:, :main_len]

        # Project each condition stream output through its dedicated LoRA output projection
        updated_conds: Dict[str, torch.Tensor] = {}
        offset = main_len
        for (stream_name, _, stream_scale, _), c_len in zip(active_streams, cond_lengths):
            cond_slice = full_out[:, offset : offset + c_len]
            offset += c_len
            if stream_name == "merged":
                proj_cond = self.out_dsa.forward_combined(cond_slice, scale_s, scale_a)
            else:
                proj_cond = self.out_dsa.forward_branch(cond_slice, stream_name, stream_scale)
            if isinstance(attn.to_out, (nn.ModuleList, nn.Sequential)) and len(attn.to_out) > 1:
                proj_cond = attn.to_out[1](proj_cond)
            updated_conds[stream_name] = proj_cond

        if txt_len > 0:
            txt_out, img_out = main_out[:, :txt_len], main_out[:, txt_len:]
            img_out = self.out_dsa(img_out)
            if isinstance(attn.to_out, (nn.ModuleList, nn.Sequential)) and len(attn.to_out) > 1:
                img_out = attn.to_out[1](img_out)
            txt_out = attn.to_add_out(txt_out)
            if kwargs.get("return_cond_outputs", False):
                return img_out, txt_out, updated_conds
            return img_out, txt_out

        img_out = self.out_dsa(main_out)
        if kwargs.get("return_cond_outputs", False):
            return img_out, updated_conds
        return img_out


def inject_dsa_lora(
    transformer: nn.Module,
    lora_cfg: Optional[DSALoRAConfig] = None,
) -> Dict[str, DSAConditionInjectionProcessor]:
    """
    Freeze all base parameters of `transformer` and attach `DSAConditionInjectionProcessor`
    to every double-stream attention block (`transformer_blocks.*.attn`) with rank r=32.
    """
    cfg = lora_cfg or DSALoRAConfig()

    for param in transformer.parameters():
        param.requires_grad = False

    injected: Dict[str, DSAConditionInjectionProcessor] = {}
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise AttributeError(
            "Expected `transformer` to expose `transformer_blocks` (double-stream MMDiT blocks)."
        )

    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        proc = DSAConditionInjectionProcessor(attn_module=attn, lora_cfg=cfg)
        if hasattr(attn, "set_processor"):
            attn.set_processor(proc)
        else:
            attn.processor = proc
        # Register processor on the block as an nn.Module attribute so `.parameters()`, `.to()`,
        # and Accelerate DDP discover all trainable LoRA weights automatically.
        block.dsa_processor = proc
        injected[f"transformer_blocks.{idx}.dsa_processor"] = proc

    return injected


def iter_dsa_processors(transformer: nn.Module) -> Iterable[Tuple[str, DSAConditionInjectionProcessor]]:
    """Yield `(name, processor)` for every injected `DSAConditionInjectionProcessor`."""
    for name, module in transformer.named_modules():
        if isinstance(module, DSAConditionInjectionProcessor):
            yield name, module


def clear_kv_cache(transformer: nn.Module) -> None:
    """Clear `bank_kv` across all `DSAConditionInjectionProcessor` modules in `transformer`."""
    for _, proc in iter_dsa_processors(transformer):
        proc.clear_cache()


def set_dsa_trainable_branches(
    transformer: nn.Module,
    train_structure: bool = True,
    train_appearance: bool = True,
) -> None:
    """
    Toggle gradient updates per LoRA stream:
      - Stage 1 (0 - 10k steps): `train_structure=True, train_appearance=False`
      - Stage 2 & 3 (10k - 50k steps): `train_structure=True, train_appearance=True`
    """
    for _, proc in iter_dsa_processors(transformer):
        for dsa_linear in (proc.q_dsa, proc.k_dsa, proc.v_dsa, proc.out_dsa):
            for p in dsa_linear.structure_lora.parameters():
                p.requires_grad = train_structure
            for p in dsa_linear.appearance_lora.parameters():
                p.requires_grad = train_appearance


def get_dsa_lora_parameters(
    transformer: nn.Module,
    include_structure: bool = True,
    include_appearance: bool = True,
) -> List[nn.Parameter]:
    """Return the list of DSA LoRA parameters matching the requested branches."""
    params: List[nn.Parameter] = []
    seen = set()
    for _, proc in iter_dsa_processors(transformer):
        for dsa_linear in (proc.q_dsa, proc.k_dsa, proc.v_dsa, proc.out_dsa):
            if include_appearance:
                for p in dsa_linear.appearance_lora.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
            if include_structure:
                for p in dsa_linear.structure_lora.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
    return params


def save_dsa_lora_weights(
    transformer: nn.Module,
    save_dir: str | Path,
) -> Dict[str, Path]:
    """
    Save separate `appearance_lora.safetensors` (subject LoRA) and
    `structure_lora.safetensors` (depth LoRA) checkpoints, plus a combined state dict.
    Falls back to `torch.save` if `safetensors` is unavailable.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    appearance_sd: Dict[str, torch.Tensor] = {}
    structure_sd: Dict[str, torch.Tensor] = {}
    combined_sd: Dict[str, torch.Tensor] = {}

    for full_key, v in transformer.state_dict().items():
        if ".appearance_lora." in full_key:
            t = v.detach().cpu().contiguous()
            appearance_sd[full_key] = t
            combined_sd[full_key] = t
        elif ".structure_lora." in full_key:
            t = v.detach().cpu().contiguous()
            structure_sd[full_key] = t
            combined_sd[full_key] = t

    if safetensors_save is not None:
        app_path = out_dir / "appearance_lora.safetensors"
        struct_path = out_dir / "structure_lora.safetensors"
        comb_path = out_dir / "dsa_lora_combined.safetensors"
        safetensors_save(appearance_sd, str(app_path))
        safetensors_save(structure_sd, str(struct_path))
        safetensors_save(combined_sd, str(comb_path))
    else:
        app_path = out_dir / "appearance_lora.pt"
        struct_path = out_dir / "structure_lora.pt"
        comb_path = out_dir / "dsa_lora_combined.pt"
        torch.save(appearance_sd, app_path)
        torch.save(structure_sd, struct_path)
        torch.save(combined_sd, comb_path)

    return {
        "appearance": app_path,
        "structure": struct_path,
        "combined": comb_path,
    }


def load_dsa_lora_weights(
    transformer: nn.Module,
    load_dir: str | Path,
    load_appearance: bool = True,
    load_structure: bool = True,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load DSA LoRA weights into `transformer`.
    Follows Paper Section 4.3 ordering: loads Subject/Appearance LoRA first, then Depth/Structure LoRA.
    """
    in_dir = Path(load_dir)
    loaded_tensors: Dict[str, torch.Tensor] = {}

    def _read_checkpoint(stem: str) -> Dict[str, torch.Tensor]:
        st_path = in_dir / f"{stem}.safetensors"
        pt_path = in_dir / f"{stem}.pt"
        if st_path.exists() and safetensors_load is not None:
            return safetensors_load(str(st_path), device="cpu")
        if pt_path.exists():
            return torch.load(pt_path, map_location="cpu", weights_only=True)
        return {}

    # 1. Subject/Appearance LoRA loaded first (Section 4.3)
    if load_appearance:
        loaded_tensors.update(_read_checkpoint("appearance_lora"))
    # 2. Depth/Structure LoRA loaded second
    if load_structure:
        loaded_tensors.update(_read_checkpoint("structure_lora"))

    if not loaded_tensors:
        combined = _read_checkpoint("dsa_lora_combined")
        for k, v in combined.items():
            if not load_appearance and ".appearance_lora." in k:
                continue
            if not load_structure and ".structure_lora." in k:
                continue
            loaded_tensors[k] = v

    current_sd = transformer.state_dict()
    missing: List[str] = []
    unexpected: List[str] = []

    dsa_keys = {
        k for k in current_sd.keys()
        if (".appearance_lora." in k and load_appearance) or (".structure_lora." in k and load_structure)
    }

    for k in dsa_keys:
        if k not in loaded_tensors:
            missing.append(k)
    for k in loaded_tensors:
        if k not in current_sd:
            unexpected.append(k)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"DSA LoRA checkpoint mismatch from {in_dir}: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    transformer.load_state_dict(loaded_tensors, strict=False)
    return missing, unexpected
