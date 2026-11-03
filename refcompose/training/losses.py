"""
Conditional Rectified Flow-Matching Training Objective (Paper Section 3.4 Eq. 12):
    z_t = (1 - t) z_0 + t z_1,   z_1 ~ N(0, I),   t ~ U[0, 1]
    L_FM = E_{t, z_0, z_1} [ || v_theta(z_t, t, p, z_C, z_{D_s}; Delta W_s + Delta W_a) - (z_1 - z_0) ||_2^2 ]
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFlowMatchingLoss(nn.Module):
    """Implements Paper Eq. (12) Conditional Rectified-Flow Velocity Regression."""

    def __init__(self, timestep_sampling: str = "uniform") -> None:
        super().__init__()
        if timestep_sampling not in ("uniform", "logit_normal"):
            raise ValueError(f"Unsupported `timestep_sampling`: {timestep_sampling}")
        self.timestep_sampling = timestep_sampling

    def sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.timestep_sampling == "logit_normal":
            u = torch.randn((batch_size,), device=device, dtype=torch.float32)
            t = torch.sigmoid(u)
        else:
            # Exact Paper Eq. (12): t ~ U[0, 1]
            t = torch.rand((batch_size,), device=device, dtype=torch.float32)
        return t.clamp(1e-4, 1.0 - 1e-4).to(dtype)

    def prepare_noisy_latents(
        self,
        z_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given clean target latent z_0 = E(I) of shape (B, L, D):
          1. Sample Gaussian terminal noise z_1 ~ N(0, I)
          2. Sample timestep t in [0, 1]
          3. Return `(z_t, t, target_velocity)` where `z_t = (1 - t) z_0 + t z_1`
             and `target_velocity = z_1 - z_0`.
        """
        bsz = z_0.shape[0]
        z_1 = torch.randn_like(z_0)
        t = self.sample_timesteps(bsz, device=z_0.device, dtype=z_0.dtype)
        t_view = t.view(bsz, *([1] * (z_0.ndim - 1)))
        z_t = (1.0 - t_view) * z_0 + t_view * z_1
        target_velocity = z_1 - z_0
        return z_t, t, target_velocity

    def forward(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error over velocity vector elements || v_theta - (z_1 - z_0) ||_2^2."""
        return F.mse_loss(predicted_velocity.float(), target_velocity.float(), reduction="mean")

