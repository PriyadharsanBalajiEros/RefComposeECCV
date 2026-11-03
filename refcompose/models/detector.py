"""
Open-Vocabulary Object Localization via Grounding DINO (Paper Section 3.1 Eq. 2 & Section 4.1):
    B = G(L) = {b_1, ..., b_N},   b_k = (x_k, y_k, w_k, h_k)
Includes bipartite Hungarian matching (`scipy.optimize.linear_sum_assignment`) to tie
each reference I_k cleanly to its localized output region b_k in L.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from refcompose.canvas.composer import BoundingBox, compute_box_iou
from refcompose.config import DetectionConfig


class GroundingDINOLocalizer:
    """
    Wraps HuggingFace `IDEA-Research/grounding-dino-base` for open-vocabulary localization.
    Supports `use_mock_detector=True` for fast offline unit testing.
    """

    def __init__(
        self,
        repo_id: str = "IDEA-Research/grounding-dino-base",
        cfg: Optional[DetectionConfig] = None,
        device: Optional[torch.device] = None,
        use_mock_detector: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.cfg = cfg or DetectionConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock_detector = use_mock_detector
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self.use_mock_detector or self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.repo_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.repo_id).to(self.device)
        self._model.eval()

    @staticmethod
    def _nms_boxes(boxes: List[BoundingBox], iou_threshold: float = 0.50) -> List[BoundingBox]:
        sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
        kept: List[BoundingBox] = []
        for candidate in sorted_boxes:
            if all(compute_box_iou(candidate, existing) < iou_threshold for existing in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _generate_fallback_grid_boxes(
        num_refs: int,
        width: int,
        height: int,
        labels: Sequence[str],
    ) -> List[BoundingBox]:
        """Generate clean non-overlapping compositional layout boxes when a detection misses."""
        if num_refs <= 0:
            return []
        cols = max(1, int(math.ceil(math.sqrt(num_refs * (width / max(1.0, float(height)))))))
        rows = max(1, int(math.ceil(num_refs / cols)))
        cell_w = width / cols
        cell_h = height / rows
        pad_x = cell_w * 0.08
        pad_y = cell_h * 0.08

        boxes: List[BoundingBox] = []
        for idx in range(num_refs):
            r = idx // cols
            c = idx % cols
            bx = c * cell_w + pad_x
            by = r * cell_h + pad_y
            bw = max(16.0, cell_w - 2.0 * pad_x)
            bh = max(16.0, cell_h - 2.0 * pad_y)
            boxes.append(
                BoundingBox(
                    x=bx,
                    y=by,
                    w=bw,
                    h=bh,
                    label=labels[idx] if idx < len(labels) else f"object_{idx}",
                    score=0.5,
                    ref_index=idx,
                )
            )
        return boxes

    @torch.no_grad()
    def detect_open_vocabulary(
        self,
        image: Image.Image,
        queries: Sequence[str],
    ) -> List[BoundingBox]:
        """
        Run Grounding DINO on `image` with open-vocabulary text queries.
        Returns NMS-filtered `BoundingBox` detections in pixel `(x, y, w, h)` coordinates.
        """
        w, h = image.size
        clean_queries = [q.strip().rstrip(".") for q in queries if q.strip()]
        if not clean_queries:
            return []

        if self.use_mock_detector:
            return self._generate_fallback_grid_boxes(len(clean_queries), w, h, clean_queries)

        self._lazy_load()
        assert self._processor is not None and self._model is not None

        # Grounding DINO expects lowercase period-separated text queries
        text_prompt = ". ".join(q.lower() for q in clean_queries) + "."
        inputs = self._processor(images=image.convert("RGB"), text=text_prompt, return_tensors="pt").to(
            self.device
        )
        outputs = self._model(**inputs)

        post_kwargs = {
            "outputs": outputs,
            "input_ids": inputs.input_ids,
            "target_sizes": [(h, w)],
        }
        # Support both `threshold` (newer transformers) and `box_threshold`
        post_fn = self._processor.post_process_grounded_object_detection
        if "threshold" in post_fn.__code__.co_varnames:
            post_kwargs["threshold"] = self.cfg.box_threshold
            post_kwargs["text_threshold"] = self.cfg.text_threshold
        else:
            post_kwargs["box_threshold"] = self.cfg.box_threshold
            post_kwargs["text_threshold"] = self.cfg.text_threshold

        result = post_fn(**post_kwargs)[0]
        raw_boxes = result["boxes"].detach().cpu().numpy()
        raw_scores = result["scores"].detach().cpu().numpy()
        raw_labels = result.get("text_labels", result.get("labels", [""] * len(raw_boxes)))

        detections: List[BoundingBox] = []
        for (x1, y1, x2, y2), score, lbl in zip(raw_boxes, raw_scores, raw_labels):
            detections.append(
                BoundingBox.from_xyxy(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    label=str(lbl),
                    score=float(score),
                )
            )
        return self._nms_boxes(detections, iou_threshold=self.cfg.iou_nms_threshold)

    def localize_references(
        self,
        layout_image: Image.Image,
        reference_descriptions: Sequence[str],
    ) -> List[BoundingBox]:
        """
        Localize each of the N references {I_1, ..., I_N} in layout image L (Eq. 2),
        assigning every reference k an output region b_k = (x_k, y_k, w_k, h_k) via
        optimal bipartite Hungarian matching over phrase similarity + confidence score.
        """
        num_refs = len(reference_descriptions)
        w, h = layout_image.size
        if num_refs == 0:
            return []

        detections = self.detect_open_vocabulary(layout_image, reference_descriptions)
        fallback_boxes = self._generate_fallback_grid_boxes(num_refs, w, h, reference_descriptions)

        if not detections:
            return fallback_boxes if self.cfg.fallback_layout_on_miss else []

        # Build cost matrix (num_refs x num_detections) combining token overlap + detection confidence
        cost = np.zeros((num_refs, len(detections)), dtype=np.float32)
        for r_idx, ref_desc in enumerate(reference_descriptions):
            ref_tokens = set(ref_desc.lower().replace(",", " ").split())
            for d_idx, det in enumerate(detections):
                det_tokens = set(det.label.lower().replace(",", " ").split())
                overlap = (
                    len(ref_tokens & det_tokens) / max(1, len(ref_tokens | det_tokens))
                    if ref_tokens and det_tokens
                    else 0.0
                )
                # Higher overlap + higher score -> lower cost
                cost[r_idx, d_idx] = -(2.0 * overlap + det.score)

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned: List[Optional[BoundingBox]] = [None] * num_refs
        for r_i, c_i in zip(row_ind, col_ind):
            matched = detections[c_i]
            assigned[r_i] = BoundingBox(
                x=matched.x,
                y=matched.y,
                w=matched.w,
                h=matched.h,
                label=reference_descriptions[r_i],
                score=matched.score,
                ref_index=r_i,
            )

        final_boxes: List[BoundingBox] = []
        for idx in range(num_refs):
            if assigned[idx] is not None:
                final_boxes.append(assigned[idx])  # type: ignore[arg-type]
            elif self.cfg.fallback_layout_on_miss:
                final_boxes.append(fallback_boxes[idx])
        return final_boxes

