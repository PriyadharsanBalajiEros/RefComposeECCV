"""
Three-Stage RefCompose LoRA Training Curriculum (Paper Section 4.3, p. 9):
  - Stage 1 (steps 0..9,999; 10,000 iters):
      Depth D_s + black canvas C=0. Initializes structure LoRA branch (Delta W_s) before
      appearance cues are introduced so canvas stream cannot compensate for geometry.
  - Stage 2 (steps 10,000..29,999; 20,000 iters):
      Crops placed at clean Grounding DINO boxes on ground-truth image (depth-sorted).
      Jointly trains structure (Delta W_s) and appearance (Delta W_a).
  - Stage 3 (steps 30,000..49,999; 20,000 iters):
      Adds photometric jitter and mild spatial perturbations (affine warps, small box
      translations) to crop placement for robustness to inference-time layout noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import torch.nn as nn

from refcompose.config import CurriculumConfig
from refcompose.models.dsa_lora import set_dsa_trainable_branches


@dataclass(frozen=True)
class StageState:
    stage: int
    name: str
    structure_scale: float
    appearance_scale: float
    train_structure: bool
    train_appearance: bool
    use_black_canvas: bool
    apply_perturbations: bool


class ThreeStageCurriculumScheduler:
    """Manages stage transitions across the 50,000-iteration RefCompose training schedule."""

    def __init__(self, cfg: Optional[CurriculumConfig] = None) -> None:
        self.cfg = cfg or CurriculumConfig()
        self._active_stage: Optional[int] = None

    def get_stage_for_step(self, global_step: int) -> int:
        if global_step < self.cfg.stage1_end_step:
            return 1
        if global_step < self.cfg.stage2_end_step:
            return 2
        return 3

    def get_state(self, global_step: int) -> StageState:
        stage = self.get_stage_for_step(global_step)
        if stage == 1:
            return StageState(
                stage=1,
                name="stage1_structure_warmup_black_canvas",
                structure_scale=1.0,
                appearance_scale=0.0,
                train_structure=True,
                train_appearance=False,
                use_black_canvas=True,
                apply_perturbations=False,
            )
        if stage == 2:
            return StageState(
                stage=2,
                name="stage2_clean_appearance_grounding",
                structure_scale=1.0,
                appearance_scale=1.0,
                train_structure=True,
                train_appearance=True,
                use_black_canvas=False,
                apply_perturbations=False,
            )
        return StageState(
            stage=3,
            name="stage3_perturbed_photometric_affine_grounding",
            structure_scale=1.0,
            appearance_scale=1.0,
            train_structure=True,
            train_appearance=True,
            use_black_canvas=False,
            apply_perturbations=True,
        )

    def maybe_transition(
        self,
        global_step: int,
        transformer: nn.Module,
        dataset: Optional[Any] = None,
    ) -> TupleResult:
        """
        Apply trainable branch masks & dataset mode updates whenever entering a new stage.
        Returns `(transitioned: bool, state: StageState)`.
        """
        state = self.get_state(global_step)
        transitioned = state.stage != self._active_stage
        if transitioned:
            set_dsa_trainable_branches(
                transformer,
                train_structure=state.train_structure,
                train_appearance=state.train_appearance,
            )
            if dataset is not None and hasattr(dataset, "set_curriculum_stage"):
                dataset.set_curriculum_stage(state.stage)
            self._active_stage = state.stage
        return transitioned, state


TupleResult = tuple[bool, StageState]

