"""
Reference Canvas Representation & Depth-Ordered Sequential Overwrite Compositing
(Paper Section 3.2 Eq. 3-4 & Section 3.3 Eq. 5-7).

Key Equations:
    phi(I_k, b_k) = Resize(I_k, h_k x w_k) placed at b_k                       (Eq. 3)
    z_C = E(C) in R^{(H/f) x (W/f) x c}                                        (Eq. 4)
    D_s = Psi(L) in R^{H x W}                                                  (Eq. 5)
    d_bar_k = mean(D_s | b_k), sorted far-to-near as permutation pi
    C = Paste(phi(I_{pi(1)}, b_{pi(1)}), ..., phi(I_{pi(N)}, b_{pi(N)}))        (Eq. 6)
    z_{D_s} = E(D_s) in R^{(H/f) x (W/f) x c}                                  (Eq. 7)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

from refcompose.canvas.augmentations import Stage3CropAugmentor
from refcompose.config import AugmentationConfig, CanvasConfig
from refcompose.utils.image_ops import depth_to_rgb_tensor, pil_to_tensor


@dataclass
class BoundingBox:
    """
    Axis-aligned pixel-space bounding box b_k = (x_k, y_k, w_k, h_k) (Eq. 2).
    Coordinates are in pixel units with (x, y) as the top-left corner.
    """

    x: float
    y: float
    w: float
    h: float
    label: str = ""
    score: float = 1.0
    ref_index: int = -1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_int_xyxy(self, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
        x1 = int(np.clip(round(self.x), 0, max_w - 1))
        y1 = int(np.clip(round(self.y), 0, max_h - 1))
        x2 = int(np.clip(round(self.x + self.w), x1 + 1, max_w))
        y2 = int(np.clip(round(self.y + self.h), y1 + 1, max_h))
        return x1, y1, x2, y2

    def to_normalized_xyxy(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.to_xyxy()
        return (
            float(np.clip(x1 / max(1, width), 0.0, 1.0)),
            float(np.clip(y1 / max(1, height), 0.0, 1.0)),
            float(np.clip(x2 / max(1, width), 0.0, 1.0)),
            float(np.clip(y2 / max(1, height), 0.0, 1.0)),
        )

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = "",
        score: float = 1.0,
        ref_index: int = -1,
    ) -> "BoundingBox":
        return cls(
            x=min(x1, x2),
            y=min(y1, y2),
            w=abs(x2 - x1),
            h=abs(y2 - y1),
            label=label,
            score=score,
            ref_index=ref_index,
        )

    @classmethod
    def from_normalized_xyxy(
        cls,
        norm_xyxy: Sequence[float],
        width: int,
        height: int,
        label: str = "",
        score: float = 1.0,
        ref_index: int = -1,
    ) -> "BoundingBox":
        nx1, ny1, nx2, ny2 = norm_xyxy
        return cls.from_xyxy(
            nx1 * width,
            ny1 * height,
            nx2 * width,
            ny2 * height,
            label=label,
            score=score,
            ref_index=ref_index,
        )


def compute_box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two BoundingBoxes."""
    ax1, ay1, ax2, ay2 = box_a.to_xyxy()
    bx1, by1, bx2, by2 = box_b.to_xyxy()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = box_a.area + box_b.area - inter_area
    if union_area <= 1e-8:
        return 0.0
    return float(inter_area / union_area)


def compute_mean_box_depths(
    depth_map: np.ndarray,
    boxes: Sequence[BoundingBox],
) -> List[float]:
    """
    Compute mean depth d_bar_k = mean(D_s | b_k) for each reference k (Section 3.3).
    `depth_map` has shape (H, W).
    """
    if depth_map.ndim == 3:
        depth_map = depth_map[..., 0] if depth_map.shape[-1] == 1 else depth_map.mean(axis=-1)
    h, w = depth_map.shape[:2]

    mean_depths: List[float] = []
    for box in boxes:
        x1, y1, x2, y2 = box.to_int_xyxy(max_w=w, max_h=h)
        patch = depth_map[y1:y2, x1:x2]
        if patch.size == 0:
            mean_depths.append(0.5)
        else:
            mean_depths.append(float(np.mean(patch)))
    return mean_depths


def sort_far_to_near(
    mean_depths: Sequence[float],
    higher_is_farther: bool = True,
) -> List[int]:
    """
    Sort reference indices from FAR-to-NEAR as permutation pi (Eq. 6).
    - When `higher_is_farther=True` (metric/relative depth convention used by Depth Anything 3,
      where larger depth = farther from camera, smaller depth = nearer to camera):
      sorts descending so farthest objects are pasted first and nearer objects overwrite them.
    - When `higher_is_farther=False` (inverse depth / disparity convention):
      sorts ascending so smallest disparity (farthest) is pasted first.
    """
    indexed = list(enumerate(mean_depths))
    indexed.sort(key=lambda item: item[1], reverse=higher_is_farther)
    return [idx for idx, _ in indexed]


@dataclass
class CanvasCompositionOutput:
    """Output bundle produced by `ReferenceCanvasComposer.compose`."""

    canvas_pil: Image.Image
    canvas_tensor: torch.Tensor          # (3, H, W) in [-1, 1] for z_C = E(C)
    depth_tensor: Optional[torch.Tensor] # (3, H, W) in [-1, 1] for z_{D_s} = E(D_s)
    sorted_order: List[int]              # Permutation pi (far-to-near)
    mean_depths: List[float]             # d_bar_k per input reference
    placed_boxes: List[BoundingBox]      # Final placed boxes (after any Stage-3 perturbation)


class ReferenceCanvasComposer:
    """
    Constructs fixed-resolution pixel-space Reference Canvas C in R^{H x W x 3} (Section 3.2-3.3).
    Token count |z_C| stays strictly constant regardless of reference count N.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        canvas_cfg: Optional[CanvasConfig] = None,
        aug_cfg: Optional[AugmentationConfig] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.canvas_cfg = canvas_cfg or CanvasConfig()
        self.augmentor = Stage3CropAugmentor(aug_cfg)

        interp_map = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }
        self.resample = interp_map.get(
            self.canvas_cfg.interpolation.lower(),
            Image.Resampling.BICUBIC,
        )

    def create_black_canvas(self) -> Tuple[Image.Image, torch.Tensor]:
        """Create an empty/black canvas C = 0 used during Stage-1 warm-up and w/o-canvas ablations."""
        bg_uint8 = int(np.clip(round(self.canvas_cfg.background_value * 255.0), 0, 255))
        canvas_pil = Image.new("RGB", (self.width, self.height), color=(bg_uint8, bg_uint8, bg_uint8))
        canvas_tensor = pil_to_tensor(canvas_pil, normalize_minus1_to_1=True)
        return canvas_pil, canvas_tensor

    def compose(
        self,
        references: Sequence[Image.Image],
        boxes: Sequence[BoundingBox],
        depth_map: Optional[np.ndarray] = None,
        higher_depth_is_farther: bool = True,
        masks: Optional[Sequence[Optional[np.ndarray]]] = None,
        apply_stage3_perturbations: bool = False,
        black_canvas_override: bool = False,
        rng: Optional[random.Random] = None,
    ) -> CanvasCompositionOutput:
        """
        Assemble the fixed-resolution Reference Canvas C via depth-sorted sequential overwrite:
            C = Paste(phi(I_{pi(1)}, b_{pi(1)}), ..., phi(I_{pi(N)}, b_{pi(N)}))   (Eq. 6)
        """
        depth_tensor: Optional[torch.Tensor] = None
        if depth_map is not None:
            if depth_map.shape[:2] != (self.height, self.width):
                depth_map = cv2.resize(
                    depth_map.astype(np.float32),
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            depth_tensor = depth_to_rgb_tensor(depth_map, normalize_minus1_to_1=True)

        if black_canvas_override or len(references) == 0 or len(boxes) == 0:
            canvas_pil, canvas_tensor = self.create_black_canvas()
            return CanvasCompositionOutput(
                canvas_pil=canvas_pil,
                canvas_tensor=canvas_tensor,
                depth_tensor=depth_tensor,
                sorted_order=[],
                mean_depths=[],
                placed_boxes=[],
            )

        n = min(len(references), len(boxes))
        refs = list(references[:n])
        active_boxes = list(boxes[:n])

        if depth_map is not None:
            mean_depths = compute_mean_box_depths(depth_map, active_boxes)
            sorted_order = sort_far_to_near(
                mean_depths,
                higher_is_farther=higher_depth_is_farther,
            )
        else:
            # Fallback when depth prior is ablated: sort larger area boxes first (background -> foreground)
            mean_depths = [0.0] * n
            sorted_order = sorted(range(n), key=lambda k: active_boxes[k].area, reverse=True)

        bg_uint8 = int(np.clip(round(self.canvas_cfg.background_value * 255.0), 0, 255))
        canvas_np = np.full((self.height, self.width, 3), bg_uint8, dtype=np.uint8)
        placed_boxes: List[BoundingBox] = [b for b in active_boxes]

        for k in sorted_order:
            crop_img = refs[k]
            box = active_boxes[k]
            crop_mask: Optional[np.ndarray] = None
            if masks is not None and k < len(masks) and masks[k] is not None:
                crop_mask = masks[k]
            elif crop_img.mode == "RGBA":
                crop_mask = np.asarray(crop_img.split()[-1], dtype=np.float32) / 255.0

            crop_rgb = crop_img.convert("RGB")
            box_xywh = box.to_xywh()

            if apply_stage3_perturbations:
                crop_rgb, box_xywh, crop_mask = self.augmentor.augment_crop_and_box(
                    crop=crop_rgb,
                    box_xywh=box_xywh,
                    canvas_w=self.width,
                    canvas_h=self.height,
                    mask=crop_mask,
                    rng=rng,
                )

            perturbed_box = BoundingBox(
                x=box_xywh[0],
                y=box_xywh[1],
                w=box_xywh[2],
                h=box_xywh[3],
                label=box.label,
                score=box.score,
                ref_index=box.ref_index if box.ref_index >= 0 else k,
            )
            placed_boxes[k] = perturbed_box

            x1, y1, x2, y2 = perturbed_box.to_int_xyxy(max_w=self.width, max_h=self.height)
            target_w = x2 - x1
            target_h = y2 - y1
            if target_w <= 0 or target_h <= 0:
                continue

            # Eq. (3): phi(I_k, b_k) = Resize(I_k, h_k x w_k) placed at b_k
            resized_crop = np.asarray(
                crop_rgb.resize((target_w, target_h), self.resample),
                dtype=np.uint8,
            )

            if crop_mask is not None:
                resized_alpha = cv2.resize(
                    crop_mask.astype(np.float32),
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_alpha = np.clip(resized_alpha, 0.0, 1.0)[..., None]
                dst_patch = canvas_np[y1:y2, x1:x2].astype(np.float32)
                blended = resized_alpha * resized_crop.astype(np.float32) + (1.0 - resized_alpha) * dst_patch
                canvas_np[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
            else:
                if self.canvas_cfg.feather_border_px > 0 and min(target_w, target_h) > 4 * self.canvas_cfg.feather_border_px:
                    fp = self.canvas_cfg.feather_border_px
                    alpha = np.ones((target_h, target_w), dtype=np.float32)
                    ramp = np.linspace(0.0, 1.0, fp, dtype=np.float32)
                    alpha[:fp, :] *= ramp[:, None]
                    alpha[-fp:, :] *= ramp[::-1, None]
                    alpha[:, :fp] *= ramp[None, :]
                    alpha[:, -fp:] *= ramp[None, ::-1]
                    alpha = alpha[..., None]
                    dst_patch = canvas_np[y1:y2, x1:x2].astype(np.float32)
                    blended = alpha * resized_crop.astype(np.float32) + (1.0 - alpha) * dst_patch
                    canvas_np[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
                else:
                    # Eq. (6): Direct pixel overwrite compositing (nearer overwrites farther)
                    canvas_np[y1:y2, x1:x2] = resized_crop

        canvas_pil = Image.fromarray(canvas_np, mode="RGB")
        canvas_tensor = pil_to_tensor(canvas_pil, normalize_minus1_to_1=True)

        return CanvasCompositionOutput(
            canvas_pil=canvas_pil,
            canvas_tensor=canvas_tensor,
            depth_tensor=depth_tensor,
            sorted_order=sorted_order,
            mean_depths=mean_depths,
            placed_boxes=placed_boxes,
        )

