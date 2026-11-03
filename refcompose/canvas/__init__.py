"""Reference Canvas Construction and Stage-3 Spatial/Photometric Perturbations."""

from refcompose.canvas.augmentations import Stage3CropAugmentor
from refcompose.canvas.composer import (
    BoundingBox,
    CanvasCompositionOutput,
    ReferenceCanvasComposer,
    compute_box_iou,
    compute_mean_box_depths,
    sort_far_to_near,
)

__all__ = [
    "Stage3CropAugmentor",
    "BoundingBox",
    "CanvasCompositionOutput",
    "ReferenceCanvasComposer",
    "compute_box_iou",
    "compute_mean_box_depths",
    "sort_far_to_near",
]

