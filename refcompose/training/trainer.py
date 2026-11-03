"""
RefCompose DSA LoRA Trainer (Paper Section 3.4 & Section 4.3).
Supports both distributed/mixed-precision HuggingFace `accelerate` execution on H100 GPUs
and standalone PyTorch execution for single-GPU / CPU testing.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from refcompose.config import RefComposeConfig
from refcompose.models.dsa_lora import (
    get_dsa_lora_parameters,
    load_dsa_lora_weights,
    save_dsa_lora_weights,
)
from refcompose.models.flux2_wrapper import Flux2BackboneWrapper, prepare_latent_image_ids
from refcompose.training.curriculum import ThreeStageCurriculumScheduler
from refcompose.training.losses import ConditionalFlowMatchingLoss
from refcompose.utils.logging_utils import set_global_seed, setup_logger


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    max_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        if scheduler_type == "constant":
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _collate_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "target_tensor": torch.stack([x["target_tensor"] for x in items], dim=0),
        "canvas_tensor": torch.stack([x["canvas_tensor"] for x in items], dim=0),
        "depth_tensor": torch.stack([x["depth_tensor"] for x in items], dim=0),
        "prompt": [x["prompt"] for x in items],
        "stage": items[0]["stage"],
    }


class RefComposeTrainer:
    """
    Trains Decoupled Structure-Appearance (DSA) LoRA adapters on frozen FLUX.2 Dev
    across the 50,000-iteration 3-stage curriculum (Section 4.3).
    """

    def __init__(
        self,
        cfg: RefComposeConfig,
        dataset: Dataset,
        backbone: Optional[Flux2BackboneWrapper] = None,
        use_mock_backbone: bool = False,
        use_accelerate: bool = True,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        set_global_seed(cfg.training.seed)

        self.output_dir = Path(cfg.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            "refcompose.trainer",
            log_file=Path(cfg.training.logging_dir) / "train.log",
        )

        self.accelerator = None
        if use_accelerate:
            try:
                from accelerate import Accelerator

                mp = cfg.training.mixed_precision if torch.cuda.is_available() else "no"
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                    mixed_precision=mp,
                )
            except ImportError:
                self.accelerator = None

        device = (
            self.accelerator.device
            if self.accelerator is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.backbone = backbone or Flux2BackboneWrapper(
            cfg=cfg,
            device=device,
            use_mock_backbone=use_mock_backbone,
            inject_lora=True,
        )

        if cfg.training.gradient_checkpointing and hasattr(
            self.backbone.transformer, "enable_gradient_checkpointing"
        ):
            self.backbone.transformer.enable_gradient_checkpointing()

        # Verify base backbone is strictly frozen and collect ONLY DSA LoRA parameters
        self.lora_params = get_dsa_lora_parameters(
            self.backbone.transformer,
            include_structure=True,
            include_appearance=True,
        )
        if not self.lora_params:
            raise RuntimeError("No trainable DSA LoRA parameters found on transformer!")

        self.optimizer = AdamW(
            self.lora_params,
            lr=cfg.training.learning_rate,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            eps=cfg.training.adam_epsilon,
            weight_decay=cfg.training.weight_decay,
        )
        self.lr_scheduler = _build_lr_scheduler(
            self.optimizer,
            scheduler_type=cfg.training.lr_scheduler,
            warmup_steps=cfg.training.lr_warmup_steps,
            max_steps=cfg.training.max_train_steps,
        )

        self.curriculum = ThreeStageCurriculumScheduler(cfg.training.curriculum)
        self.criterion = ConditionalFlowMatchingLoss(cfg.training.timestep_sampling)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0 if use_mock_backbone else cfg.training.num_workers,
            collate_fn=_collate_batch,
            drop_last=True,
        )

        self.global_step = 0
        if cfg.training.resume_from_checkpoint:
            self.load_checkpoint(cfg.training.resume_from_checkpoint)

        if self.accelerator is not None:
            (
                self.backbone.transformer,
                self.optimizer,
                self.dataloader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.backbone.transformer,
                self.optimizer,
                self.dataloader,
                self.lr_scheduler,
            )

    def close_log_handlers(self) -> None:
        """Close file handlers on Windows so temporary log directories can be cleaned up cleanly."""
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)

    def _unwrap_transformer(self) -> torch.nn.Module:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.backbone.transformer)
        return self.backbone.transformer

    def save_checkpoint(self, tag: str) -> Path:
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self._unwrap_transformer()
        save_dsa_lora_weights(unwrapped, ckpt_dir)

        state = {
            "global_step": self.global_step,
            "stage": self.curriculum.get_stage_for_step(self.global_step),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(state, ckpt_dir / "trainer_state.pt")
        with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "stage": state["stage"],
                    "lora_rank": self.cfg.lora.rank,
                    "resolution": [self.cfg.resolution.width, self.cfg.resolution.height],
                },
                f,
                indent=2,
            )
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: str | Path) -> None:
        in_dir = Path(ckpt_dir)
        load_dsa_lora_weights(self._unwrap_transformer(), in_dir, strict=False)
        state_path = in_dir / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.global_step = int(state.get("global_step", 0))
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if "lr_scheduler" in state:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.logger.info("Resumed training from %s at step %d", in_dir, self.global_step)

    def train_step(self, batch: Dict[str, Any], stage_state: Any) -> float:
        """Run one forward + backward rectified-flow update step."""
        target_img = batch["target_tensor"]
        canvas_img = batch["canvas_tensor"]
        depth_img = batch["depth_tensor"]
        prompts: List[str] = batch["prompt"]

        # Encode latents with frozen VAE E(.)
        with torch.no_grad():
            z_0 = self.backbone.encode_image_to_packed_latents(target_img)
            z_depth = self.backbone.encode_image_to_packed_latents(depth_img)
            z_canvas = (
                None
                if stage_state.use_black_canvas
                else self.backbone.encode_image_to_packed_latents(canvas_img)
            )
            prompt_embeds, txt_ids = self.backbone.encode_prompt(prompts)
            img_ids = prepare_latent_image_ids(
                batch_size=z_0.shape[0],
                packed_height=self.cfg.resolution.packed_height,
                packed_width=self.cfg.resolution.packed_width,
                device=z_0.device,
            )

        # Eq. (12): z_t = (1 - t) z_0 + t z_1, target = z_1 - z_0
        z_t, t, v_target = self.criterion.prepare_noisy_latents(z_0)

        v_pred = self.backbone.predict_velocity(
            z_t=z_t,
            timestep=t,
            prompt_embeds=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            z_canvas=z_canvas,
            z_depth=z_depth,
            appearance_scale=stage_state.appearance_scale,
            structure_scale=stage_state.structure_scale,
            cache_mode="none",  # Training always runs full differentiable block-causal pass
        )
        loss = self.criterion(v_pred, v_target)
        return loss  # type: ignore[return-value]

    def train(self, max_steps_override: Optional[int] = None) -> Dict[str, Any]:
        """Execute the 3-stage training loop up to `max_train_steps` (default 50,000)."""
        target_steps = max_steps_override or self.cfg.training.max_train_steps
        grad_accum = self.cfg.training.gradient_accumulation_steps
        self.backbone.transformer.train()

        loss_history: List[float] = []
        stage_history: List[int] = []
        data_iter = iter(self.dataloader)

        self.logger.info(
            "Starting RefCompose 3-Stage DSA LoRA Training: steps %d -> %d (r=%d, %dx%d)",
            self.global_step,
            target_steps,
            self.cfg.lora.rank,
            self.cfg.resolution.width,
            self.cfg.resolution.height,
        )

        while self.global_step < target_steps:
            transitioned, stage_state = self.curriculum.maybe_transition(
                global_step=self.global_step,
                transformer=self._unwrap_transformer(),
                dataset=self.dataset,
            )
            if transitioned:
                self.logger.info(
                    "=== Curriculum Transition at step %d -> Stage %d (%s) ===",
                    self.global_step,
                    stage_state.stage,
                    stage_state.name,
                )
                # Refresh iterator so upcoming batches reflect updated dataset stage immediately
                data_iter = iter(self.dataloader)

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                loss = self.train_step(batch, stage_state)
                scaled_loss = loss / float(grad_accum)
                if self.accelerator is not None:
                    self.accelerator.backward(scaled_loss)
                else:
                    scaled_loss.backward()
                accum_loss += float(loss.detach().item()) / float(grad_accum)

            active_params = [p for p in self.lora_params if p.requires_grad and p.grad is not None]
            if active_params and self.cfg.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(active_params, self.cfg.training.max_grad_norm)

            self.optimizer.step()
            self.lr_scheduler.step()

            self.global_step += 1
            loss_history.append(accum_loss)
            stage_history.append(stage_state.stage)

            if self.global_step % self.cfg.training.log_every_steps == 0 or self.global_step == 1:
                curr_lr = self.lr_scheduler.get_last_lr()[0]
                self.logger.info(
                    "Step %d/%d | Stage %d | L_FM: %.6f | LR: %.2e",
                    self.global_step,
                    target_steps,
                    stage_state.stage,
                    accum_loss,
                    curr_lr,
                )

            if self.global_step % self.cfg.training.save_every_steps == 0:
                saved = self.save_checkpoint(f"step_{self.global_step:06d}")
                self.logger.info("Saved checkpoint to %s", saved)

        final_dir = self.save_checkpoint("final")
        self.logger.info("Completed training! Final DSA LoRA saved at %s", final_dir)
        self.close_log_handlers()

        return {
            "final_step": self.global_step,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "stage_history": stage_history,
            "final_checkpoint_dir": str(final_dir),
        }
