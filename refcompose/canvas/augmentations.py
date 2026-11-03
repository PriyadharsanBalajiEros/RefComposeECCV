"""
Stage-3 Training Augmentations (Paper Section 4.3):
    "the remaining iterations add photometric jitter and mild spatial perturbations
     (affine warps, small translations) to crop placement, improving robustness to
     noisier boxes from the inference-time layout pipeline."
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageEnhance

from refcompose.config import AugmentationConfig


class Stage3CropAugmentor:
    """Photometric jitter + mild affine warps + small box translation perturbations."""

    def __init__(self, cfg: Optional[AugmentationConfig] = None) -> None:
        self.cfg = cfg or AugmentationConfig()

    def apply_photometric_jitter(
        self,
        crop: Image.Image,
        rng: Optional[random.Random] = None,
    ) -> Image.Image:
        """Apply mild brightness, contrast, saturation, and hue shifts to a reference crop."""
        r = rng or random
        img = crop.convert("RGB")

        if self.cfg.brightness_jitter > 0:
            factor = 1.0 + r.uniform(-self.cfg.brightness_jitter, self.cfg.brightness_jitter)
            img = ImageEnhance.Brightness(img).enhance(max(0.1, factor))

        if self.cfg.contrast_jitter > 0:
            factor = 1.0 + r.uniform(-self.cfg.contrast_jitter, self.cfg.contrast_jitter)
            img = ImageEnhance.Contrast(img).enhance(max(0.1, factor))

        if self.cfg.saturation_jitter > 0:
            factor = 1.0 + r.uniform(-self.cfg.saturation_jitter, self.cfg.saturation_jitter)
            img = ImageEnhance.Color(img).enhance(max(0.0, factor))

        if self.cfg.hue_jitter > 0:
            hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
            delta_h = int(round(r.uniform(-self.cfg.hue_jitter, self.cfg.hue_jitter) * 255.0))
            hsv[..., 0] = (hsv[..., 0] + delta_h) % 256
            img = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

        return img

    def apply_affine_warp(
        self,
        crop: Image.Image,
        mask: Optional[np.ndarray] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Apply a mild 2D affine warp (small rotation, isotropic/anisotropic scale, shear)
        around the crop center, preserving spatial dimensions.
        """
        r = rng or random
        arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]
        if h < 4 or w < 4:
            return crop, mask

        angle_deg = r.uniform(-self.cfg.max_rotation_deg, self.cfg.max_rotation_deg)
        scale = r.uniform(self.cfg.scale_range[0], self.cfg.scale_range[1])
        shear_x = math.radians(r.uniform(-self.cfg.max_shear_deg, self.cfg.max_shear_deg))
        shear_y = math.radians(r.uniform(-self.cfg.max_shear_deg, self.cfg.max_shear_deg))

        cx, cy = w * 0.5, h * 0.5
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)  # (2, 3)
        rot_3x3 = np.vstack([rot_mat, [0.0, 0.0, 1.0]])

        shear_3x3 = np.array(
            [
                [1.0, math.tan(shear_x), -cx * math.tan(shear_x)],
                [math.tan(shear_y), 1.0, -cy * math.tan(shear_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        affine = (shear_3x3 @ rot_3x3)[:2, :]

        warped = cv2.warpAffine(
            arr,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask: Optional[np.ndarray] = None
        if mask is not None:
            warped_mask = cv2.warpAffine(
                mask.astype(np.float32),
                affine,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        return Image.fromarray(warped, mode="RGB"), warped_mask

    def perturb_box_xywh(
        self,
        box_xywh: Tuple[float, float, float, float],
        canvas_w: int,
        canvas_h: int,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, float, float, float]:
        """Apply small random box translation and mild scale jitter within canvas bounds."""
        r = rng or random
        x, y, w, h = box_xywh
        dx = r.uniform(-self.cfg.max_box_translate_frac, self.cfg.max_box_translate_frac) * w
        dy = r.uniform(-self.cfg.max_box_translate_frac, self.cfg.max_box_translate_frac) * h
        sw = r.uniform(self.cfg.scale_range[0], self.cfg.scale_range[1])
        sh = r.uniform(self.cfg.scale_range[0], self.cfg.scale_range[1])

        cx = x + 0.5 * w + dx
        cy = y + 0.5 * h + dy
        new_w = max(4.0, min(float(canvas_w), w * sw))
        new_h = max(4.0, min(float(canvas_h), h * sh))
        new_x = max(0.0, min(float(canvas_w) - new_w, cx - 0.5 * new_w))
        new_y = max(0.0, min(float(canvas_h) - new_h, cy - 0.5 * new_h))
        return (new_x, new_y, new_w, new_h)

    def augment_crop_and_box(
        self,
        crop: Image.Image,
        box_xywh: Tuple[float, float, float, float],
        canvas_w: int,
        canvas_h: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Image.Image, Tuple[float, float, float, float], Optional[np.ndarray]]:
        """Jointly apply Stage-3 photometric and spatial perturbations with configured probability."""
        r = rng or random
        if r.random() > self.cfg.perturbation_prob:
            return crop, box_xywh, mask

        aug_crop = self.apply_photometric_jitter(crop, rng=r)
        aug_crop, aug_mask = self.apply_affine_warp(aug_crop, mask=mask, rng=r)
        aug_box = self.perturb_box_xywh(box_xywh, canvas_w=canvas_w, canvas_h=canvas_h, rng=r)
        return aug_crop, aug_box, aug_mask

