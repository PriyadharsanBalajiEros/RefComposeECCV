"""
Region-wise Quality Metrics following LayoutSAM-Eval (Paper Section 4.4, Table 2):
  - Spatial ↑ : Placement accuracy using Grounding DINO localizations vs target layout/scene
  - Color ↑   : Region-level chromaticity & histogram fidelity + VLM attribute VQA
  - Texture ↑ : Region-level multi-scale gradient/frequency fidelity + VLM attribute VQA
  - Shape ↑   : Region-level contour/orientation structural fidelity + VLM attribute VQA
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from PIL import Image

from refcompose.canvas.composer import BoundingBox, compute_box_iou
from refcompose.models.detector import GroundingDINOLocalizer
from refcompose.models.vlm_annotator import Qwen3VLAnnotator


class RegionMetricsSuite:
    """
    Evaluates LayoutSAM-Eval region-wise metrics (Spatial, Color, Texture, Shape)
    using Grounding DINO object localizations and hybrid visual-feature + Qwen3-VL VQA scoring.
    """

    def __init__(
        self,
        detector: GroundingDINOLocalizer,
        vlm: Optional[Qwen3VLAnnotator] = None,
        eval_mode: str = "hybrid",  # "hybrid" or "feature_only"
    ) -> None:
        self.detector = detector
        self.vlm = vlm
        self.eval_mode = eval_mode

    @staticmethod
    def _compute_color_similarity(gen_crop: Image.Image, ref_crop: Image.Image) -> float:
        """Compare 3D HSV + CIE-LAB color distributions between generated region and reference."""
        g_bgr = cv2.cvtColor(np.asarray(gen_crop.convert("RGB").resize((128, 128))), cv2.COLOR_RGB2BGR)
        r_bgr = cv2.cvtColor(np.asarray(ref_crop.convert("RGB").resize((128, 128))), cv2.COLOR_RGB2BGR)

        g_hsv = cv2.cvtColor(g_bgr, cv2.COLOR_BGR2HSV)
        r_hsv = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2HSV)

        hist_g = cv2.calcHist([g_hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
        hist_r = cv2.calcHist([r_hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_g, hist_g)
        cv2.normalize(hist_r, hist_r)
        bhat = cv2.compareHist(hist_g, hist_r, cv2.HISTCMP_BHATTACHARYYA)
        hist_sim = float(np.clip(1.0 - bhat, 0.0, 1.0))

        g_lab = cv2.cvtColor(g_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        r_lab = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        mean_diff = float(np.linalg.norm(g_lab.mean(axis=(0, 1)) - r_lab.mean(axis=(0, 1))))
        lab_sim = float(np.exp(-2.0 * mean_diff))
        return 0.5 * hist_sim + 0.5 * lab_sim

    @staticmethod
    def _compute_texture_similarity(gen_crop: Image.Image, ref_crop: Image.Image) -> float:
        """Compare multi-scale Sobel & Laplacian spatial-frequency texture statistics."""
        g_gray = np.asarray(gen_crop.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        r_gray = np.asarray(ref_crop.convert("L").resize((128, 128)), dtype=np.float32) / 255.0

        feats: List[Tuple[np.ndarray, np.ndarray]] = []
        for ksize in (3, 5):
            gx_g = cv2.Sobel(g_gray, cv2.CV_32F, 1, 0, ksize=ksize)
            gy_g = cv2.Sobel(g_gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag_g = np.hypot(gx_g, gy_g)

            gx_r = cv2.Sobel(r_gray, cv2.CV_32F, 1, 0, ksize=ksize)
            gy_r = cv2.Sobel(r_gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag_r = np.hypot(gx_r, gy_r)
            feats.append((mag_g, mag_r))

        lap_g = np.abs(cv2.Laplacian(g_gray, cv2.CV_32F, ksize=3))
        lap_r = np.abs(cv2.Laplacian(r_gray, cv2.CV_32F, ksize=3))
        feats.append((lap_g, lap_r))

        vec_g = np.array([f[0].mean() for f in feats] + [f[0].std() for f in feats], dtype=np.float32)
        vec_r = np.array([f[1].mean() for f in feats] + [f[1].std() for f in feats], dtype=np.float32)

        denom = float(np.linalg.norm(vec_g) * np.linalg.norm(vec_r))
        if denom < 1e-8:
            return 1.0 if float(np.linalg.norm(vec_g - vec_r)) < 1e-5 else 0.5
        cos = float(np.dot(vec_g, vec_r) / denom)
        mag_diff = float(np.abs(vec_g - vec_r).mean())
        return float(np.clip(0.6 * cos + 0.4 * np.exp(-3.0 * mag_diff), 0.0, 1.0))

    @staticmethod
    def _compute_shape_similarity(gen_crop: Image.Image, ref_crop: Image.Image) -> float:
        """Compare spatial edge-orientation grid histograms (HOG-style structural descriptor)."""
        g_gray = np.asarray(gen_crop.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        r_gray = np.asarray(ref_crop.convert("L").resize((128, 128)), dtype=np.float32) / 255.0

        def _orientation_grid_descriptor(img_gray: np.ndarray) -> np.ndarray:
            gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            ang = ang % 180.0
            cells: List[np.ndarray] = []
            # 4x4 spatial cells x 9 orientation bins = 144-D shape descriptor
            for cy in range(0, 128, 32):
                for cx in range(0, 128, 32):
                    m_cell = mag[cy : cy + 32, cx : cx + 32].reshape(-1)
                    a_cell = ang[cy : cy + 32, cx : cx + 32].reshape(-1)
                    hist, _ = np.histogram(a_cell, bins=9, range=(0.0, 180.0), weights=m_cell)
                    cells.append(hist.astype(np.float32))
            desc = np.concatenate(cells)
            norm = float(np.linalg.norm(desc))
            return desc / max(1e-6, norm)

        d_g = _orientation_grid_descriptor(g_gray)
        d_r = _orientation_grid_descriptor(r_gray)
        if float(np.linalg.norm(d_g)) < 1e-6 and float(np.linalg.norm(d_r)) < 1e-6:
            return 1.0
        return float(np.clip(np.dot(d_g, d_r), 0.0, 1.0))

    @staticmethod
    def _compute_spatial_score(
        detected_boxes: Sequence[BoundingBox],
        target_boxes: Sequence[BoundingBox],
        image_w: int,
        image_h: int,
    ) -> float:
        """
        Compute LayoutSAM-Eval Spatial accuracy combining:
          1. Grounding presence/confidence score of each localized object in I_hat, and
          2. Spatial center proximity + IoU alignment relative to reference layout regions.
        """
        if not target_boxes:
            return 0.0
        if not detected_boxes:
            return 0.0

        diag = float(np.hypot(image_w, image_h))
        scores: List[float] = []
        n = min(len(detected_boxes), len(target_boxes))

        for k in range(n):
            det = detected_boxes[k]
            tgt = target_boxes[k]
            iou = compute_box_iou(det, tgt)

            dcx, dcy = det.x + 0.5 * det.w, det.y + 0.5 * det.h
            tcx, tcy = tgt.x + 0.5 * tgt.w, tgt.y + 0.5 * tgt.h
            center_dist = float(np.hypot(dcx - tcx, dcy - tcy)) / max(1.0, diag)
            center_sim = float(np.clip(1.0 - center_dist, 0.0, 1.0))

            presence = float(np.clip(det.score, 0.0, 1.0))
            region_spatial = 0.40 * presence + 0.35 * center_sim + 0.25 * iou
            scores.append(region_spatial)

        # Penalize missing detections if fewer than target_boxes were localized
        coverage = float(n) / float(len(target_boxes))
        return float(np.mean(scores)) * coverage

    def evaluate_sample(
        self,
        generated: Image.Image,
        reference_crops: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        target_boxes: Sequence[BoundingBox],
        expected_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> Tuple[Dict[str, float], List[BoundingBox]]:
        """
        Localize objects in `generated` using Grounding DINO and compute region-wise
        `Spatial`, `Color`, `Texture`, and `Shape` scores.
        `expected_boxes` allows evaluating prompt-induced spatial consistency against the
        method's layout boxes (when provided) or DenseLayout ground-truth boxes.
        """
        w, h = generated.size
        localized = self.detector.localize_references(generated, reference_descriptions)
        ref_layout = expected_boxes if expected_boxes is not None else target_boxes

        spatial_val = self._compute_spatial_score(localized, ref_layout, image_w=w, image_h=h)

        color_scores: List[float] = []
        texture_scores: List[float] = []
        shape_scores: List[float] = []

        for k, ref_img in enumerate(reference_crops):
            if k >= len(localized):
                color_scores.append(0.0)
                texture_scores.append(0.0)
                shape_scores.append(0.0)
                continue

            x1, y1, x2, y2 = localized[k].to_int_xyxy(max_w=w, max_h=h)
            gen_crop = generated.crop((x1, y1, x2, y2))

            c_feat = self._compute_color_similarity(gen_crop, ref_img)
            t_feat = self._compute_texture_similarity(gen_crop, ref_img)
            s_feat = self._compute_shape_similarity(gen_crop, ref_img)

            if self.eval_mode == "hybrid" and self.vlm is not None:
                vqa = self.vlm.score_region_attributes_vqa(
                    generated_crop=gen_crop,
                    reference_crop=ref_img,
                    object_label=reference_descriptions[k] if k < len(reference_descriptions) else "object",
                )
                c_val = 0.5 * c_feat + 0.5 * vqa["color"]
                t_val = 0.5 * t_feat + 0.5 * vqa["texture"]
                s_val = 0.5 * s_feat + 0.5 * vqa["shape"]
            else:
                c_val, t_val, s_val = c_feat, t_feat, s_feat

            color_scores.append(c_val)
            texture_scores.append(t_val)
            shape_scores.append(s_val)

        return (
            {
                "Spatial": float(spatial_val),
                "Color": float(np.mean(color_scores)) if color_scores else 0.0,
                "Texture": float(np.mean(texture_scores)) if texture_scores else 0.0,
                "Shape": float(np.mean(shape_scores)) if shape_scores else 0.0,
            },
            localized,
        )

