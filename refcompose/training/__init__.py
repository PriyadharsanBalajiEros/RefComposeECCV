"""RefCompose DSA LoRA training loop, 3-stage curriculum scheduler, and flow-matching objective."""

from refcompose.training.curriculum import StageState, ThreeStageCurriculumScheduler
from refcompose.training.losses import ConditionalFlowMatchingLoss
from refcompose.training.trainer import RefComposeTrainer

__all__ = [
    "StageState",
    "ThreeStageCurriculumScheduler",
    "ConditionalFlowMatchingLoss",
    "RefComposeTrainer",
]

