"""
Monocular Geometric Scaffolding via Depth Anything 3 (Paper Section 3.3 Eq. 5-7):
    D_s = Psi(L) in R^{H x W}
Normalizes output consistently to [0.0, 1.0] where:
    - smaller value (0.0) = nearer to camera (foreground)
    - larger value (1.0)  = farther from camera (background)
so `sort_far_to_near(..., higher_is_farther=True)` always pastes background first
and foreground last across all backends!
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from PIL import Image
import torch


class DepthEstimator:
    """
    Estimates monocular relative depth D_s = Psi(L) using:
      1. `depth_anything_3.api.DepthAnything3` (`depth-anything/da3-large`, Paper Ref [16])
      2. Automatic HuggingFace fallback (`depth-anything/Depth-Anything-V2-Large-hf`)
      3. Deterministic vertical-perspective + luminance geometric prior when `use_mock_depth=True`
    """

    def __init__(
        self,
        da3_repo_id: str = "depth-anything/da3-large",
        fallback_repo_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
        device: Optional[torch.device] = None,
        use_mock_depth: bool = False,
    ) -> None:
        self.da3_repo_id = da3_repo_id
        self.fallback_repo_id = fallback_repo_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock_depth = use_mock_depth
        self._da3_model = None
        self._hf_processor = None
        self._hf_model = None
        self._backend: Optional[str] = None

    def _lazy_load(self) -> None:
        if self.use_mock_depth or self._backend is not None:
            return
        # Preferred: Official Depth Anything 3 (DA3, arXiv:2511.10647)
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]

            self._da3_model = DepthAnything3.from_pretrained(self.da3_repo_id).to(device=self.device)
            self._backend = "da3"
            return
        except ImportError:
            pass

        # Fallback: HuggingFace Transformers Depth-Anything-V2
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._hf_processor = AutoImageProcessor.from_pretrained(self.fallback_repo_id)
        self._hf_model = AutoModelForDepthEstimation.from_pretrained(self.fallback_repo_id).to(self.device)
        self._hf_model.eval()
        self._backend = "hf_dav2"

    @staticmethod
    def _normalize_zero_one(arr: np.ndarray) -> np.ndarray:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi - lo < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    @torch.no_grad()
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate relative depth D_s in R^{H x W} normalized to [0.0, 1.0],
        with 0.0 = nearest (foreground) and 1.0 = farthest (background).
        """
        w, h = image.size
        rgb = image.convert("RGB")

        if self.use_mock_depth:
            # Smooth vertical perspective ramp (top=farther=1.0, bottom=nearer=0.0) + edge structure
            y_ramp = np.linspace(1.0, 0.1, h, dtype=np.float32)[:, None]
            gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
            mock = 0.75 * np.broadcast_to(y_ramp, (h, w)) + 0.25 * (1.0 - gray)
            return self._normalize_zero_one(mock)

        self._lazy_load()

        if self._backend == "da3" and self._da3_model is not None:
            pred = self._da3_model.inference([rgb])
            # DA3 prediction.depth has shape [1, H_pred, W_pred] where larger = farther metric depth
            depth_raw = np.asarray(pred.depth[0], dtype=np.float32)
            if depth_raw.shape != (h, w):
                depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_CUBIC)
            return self._normalize_zero_one(depth_raw)

        # HuggingFace Depth-Anything-V2 outputs inverse depth (disparity: larger = nearer)
        # Invert `1.0 - norm_disparity` so larger = farther matches DA3!
        assert self._hf_processor is not None and self._hf_model is not None
        inputs = self._hf_processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self._hf_model(**inputs)
        predicted_disp = outputs.predicted_depth
        interp = torch.nn.functional.interpolate(
            predicted_disp.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)[0]
        disp_norm = self._normalize_zero_one(interp.detach().cpu().numpy())
        return (1.0 - disp_norm).astype(np.float32)

