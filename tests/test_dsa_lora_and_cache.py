"""Unit tests for DSA LoRA, Block-Causal Attention Isolation, KV Caching, and Flow-Matching Loss."""

from __future__ import annotations

import tempfile
import unittest
import torch

from refcompose.config import DSALoRAConfig
from refcompose.models.dsa_lora import (
    build_dsa_causal_attention_mask,
    clear_kv_cache,
    get_dsa_lora_parameters,
    inject_dsa_lora,
    iter_dsa_processors,
    load_dsa_lora_weights,
    save_dsa_lora_weights,
    set_dsa_trainable_branches,
)
from refcompose.models.flux2_wrapper import StandaloneMMDiT
from refcompose.training.losses import ConditionalFlowMatchingLoss


class TestDSALoRAAndCache(unittest.TestCase):
    def test_causal_mask_prevents_cross_condition_and_noisy_leakage(self) -> None:
        """
        Verify EasyControl / RefCompose block-causal mask:
          - Main tokens (txt=4, img=8) attend to all 4 + 8 + 8 + 8 = 28 keys.
          - Appearance condition tokens (8) attend ONLY to [12:20].
          - Structure condition tokens (8) attend ONLY to [20:28].
        """
        mask = build_dsa_causal_attention_mask(
            txt_len=4,
            img_len=8,
            cond_lengths=[8, 8],
            device=torch.device("cpu"),
        )[0, 0]
        self.assertEqual(tuple(mask.shape), (28, 28))
        # Main queries attend everywhere
        self.assertTrue(torch.all(mask[:12, :]))
        # Appearance queries isolated to [12:20]
        self.assertFalse(torch.any(mask[12:20, :12]))
        self.assertTrue(torch.all(mask[12:20, 12:20]))
        self.assertFalse(torch.any(mask[12:20, 20:28]))
        # Structure queries isolated to [20:28]
        self.assertFalse(torch.any(mask[20:28, :20]))
        self.assertTrue(torch.all(mask[20:28, 20:28]))

    def test_dsa_lora_injection_kv_cache_and_checkpoint_roundtrip(self) -> None:
        mmdit = StandaloneMMDiT(
            in_channels=32,
            hidden_dim=64,
            joint_attention_dim=64,
            num_heads=4,
            num_double_blocks=2,
            num_single_blocks=1,
        )
        cfg = DSALoRAConfig(rank=8, alpha=8.0, use_causal_mask=True, use_kv_cache=True)
        inject_dsa_lora(mmdit, cfg)

        # Base weights must be frozen; only LoRA A/B weights trainable
        for name, param in mmdit.named_parameters():
            if "lora_" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)

        # Perturb LoRA B slightly so non-zero low-rank updates are active
        for p in get_dsa_lora_parameters(mmdit):
            with torch.no_grad():
                p.add_(0.01 * torch.randn_like(p))

        z_t = torch.randn(1, 16, 32)
        txt = torch.randn(1, 6, 64)
        z_c = torch.randn(1, 16, 32)
        z_d = torch.randn(1, 16, 32)
        t = torch.tensor([0.5])

        # Step 0 write populates bank_kv
        clear_kv_cache(mmdit)
        _ = mmdit(
            hidden_states=z_t,
            encoder_hidden_states=txt,
            timestep=t,
            cond_appearance_latents=z_c,
            cond_structure_latents=z_d,
            cache_mode="write",
        )
        for _, proc in iter_dsa_processors(mmdit):
            self.assertIn("appearance", proc.bank_kv)
            self.assertIn("structure", proc.bank_kv)

        # Step 1 read succeeds WITHOUT passing z_c or z_d again
        v_cached = mmdit(
            hidden_states=z_t,
            encoder_hidden_states=txt,
            timestep=torch.tensor([0.4]),
            cond_appearance_latents=None,
            cond_structure_latents=None,
            cache_mode="read",
        )
        self.assertEqual(tuple(v_cached.shape), (1, 16, 32))

        clear_kv_cache(mmdit)
        for _, proc in iter_dsa_processors(mmdit):
            self.assertEqual(len(proc.bank_kv), 0)

        # Checkpoint save & load roundtrip
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dsa_lora_weights(mmdit, tmp_dir)
            mmdit2 = StandaloneMMDiT(
                in_channels=32,
                hidden_dim=64,
                joint_attention_dim=64,
                num_heads=4,
                num_double_blocks=2,
                num_single_blocks=1,
            )
            inject_dsa_lora(mmdit2, cfg)
            missing, unexpected = load_dsa_lora_weights(mmdit2, tmp_dir, strict=True)
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(unexpected), 0)

    def test_flow_matching_loss_and_branch_selective_gradients(self) -> None:
        mmdit = StandaloneMMDiT(
            in_channels=16,
            hidden_dim=64,
            joint_attention_dim=64,
            num_heads=4,
            num_double_blocks=2,
            num_single_blocks=1,
        )
        inject_dsa_lora(mmdit, DSALoRAConfig(rank=4, alpha=4.0))

        # Stage 1 mode: train structure ONLY, freeze appearance
        set_dsa_trainable_branches(mmdit, train_structure=True, train_appearance=False)
        criterion = ConditionalFlowMatchingLoss(timestep_sampling="uniform")

        z_0 = torch.randn(2, 12, 16)
        z_d = torch.randn(2, 12, 16)
        txt = torch.randn(2, 5, 64)
        z_t, t, v_target = criterion.prepare_noisy_latents(z_0)

        v_pred = mmdit(
            hidden_states=z_t,
            encoder_hidden_states=txt,
            timestep=t,
            cond_appearance_latents=None,
            cond_structure_latents=z_d,
            appearance_scale=0.0,
            structure_scale=1.0,
            cache_mode="none",
        )
        loss = criterion(v_pred, v_target)
        loss.backward()

        struct_grads = [
            p.grad for p in get_dsa_lora_parameters(mmdit, include_structure=True, include_appearance=False)
            if p.grad is not None
        ]
        app_grads = [
            p.grad for p in get_dsa_lora_parameters(mmdit, include_structure=False, include_appearance=True)
            if p.grad is not None
        ]
        self.assertGreater(len(struct_grads), 0)
        self.assertEqual(len(app_grads), 0)


if __name__ == "__main__":
    unittest.main()

