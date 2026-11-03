"""
Global-wise Quality Metrics (Paper Section 4.4, Table 2):
  - PICK ↑      : PickScore human preference score (Kirstain et al., 2023)
  - DINO ↑      : Global composition similarity between generated I_hat and ground-truth I
  - DINO_refs ↑ : Mean DINO similarity between localized regions in I_hat and canonicalized references R
  - CLIP-I ↑    : CLIP image-level cosine similarity between I_hat and ground-truth I
  - CLIP-T ↑    : CLIP text-image cosine similarity between I_hat and prompt p
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from refcompose.canvas.composer import BoundingBox
from refcompose.config import RefComposeConfig
from refcompose.models.detector import GroundingDINOLocalizer


class GlobalMetricsSuite:
    """Computes PICK, DINO, DINO_refs, CLIP-I, and CLIP-T on generated composites."""

    def __init__(
        self,
        cfg: RefComposeConfig,
        detector: Optional[GroundingDINOLocalizer] = None,
        device: Optional[torch.device] = None,
        use_mock_metrics: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock_metrics = use_mock_metrics
        self.detector = detector or GroundingDINOLocalizer(
            repo_id=cfg.model.grounding_dino_repo_id,
            cfg=cfg.detection,
            device=self.device,
            use_mock_detector=use_mock_metrics,
        )

        self._dino_proc = None
        self._dino_model = None
        self._clip_proc = None
        self._clip_model = None
        self._pick_proc = None
        self._pick_model = None

    def _load_dino(self) -> None:
        if self.use_mock_metrics or self._dino_model is not None:
            return
        from transformers import ViTImageProcessor, ViTModel

        self._dino_proc = ViTImageProcessor.from_pretrained(self.cfg.model.dino_repo_id)
        self._dino_model = ViTModel.from_pretrained(self.cfg.model.dino_repo_id).to(self.device)
        self._dino_model.eval()

    def _load_clip(self) -> None:
        if self.use_mock_metrics or self._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._clip_proc = CLIPProcessor.from_pretrained(self.cfg.model.clip_repo_id)
        self._clip_model = CLIPModel.from_pretrained(self.cfg.model.clip_repo_id).to(self.device)
        self._clip_model.eval()

    def _load_pickscore(self) -> None:
        if self.use_mock_metrics or self._pick_model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._pick_proc = AutoProcessor.from_pretrained(self.cfg.model.pickscore_clip_repo_id)
        self._pick_model = AutoModel.from_pretrained(self.cfg.model.pickscore_repo_id).to(self.device)
        self._pick_model.eval()

    @staticmethod
    def _mock_image_embedding(img: Image.Image, dim: int = 256) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        flat = torch.from_numpy(arr.reshape(-1))
        if flat.numel() >= dim:
            vec = flat[:dim]
        else:
            vec = F.pad(flat, (0, dim - flat.numel()))
        return F.normalize(vec, dim=0)

    @torch.no_grad()
    def extract_dino_feature(self, image: Image.Image) -> torch.Tensor:
        if self.use_mock_metrics:
            return self._mock_image_embedding(image, dim=384)
        self._load_dino()
        assert self._dino_proc is not None and self._dino_model is not None
        inputs = self._dino_proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        outputs = self._dino_model(**inputs)
        cls_feat = outputs.last_hidden_state[:, 0, :][0].float()
        return F.normalize(cls_feat, dim=0)

    @torch.no_grad()
    def compute_dino_global(self, generated: Image.Image, ground_truth: Image.Image) -> float:
        emb_gen = self.extract_dino_feature(generated)
        emb_gt = self.extract_dino_feature(ground_truth)
        return float(torch.dot(emb_gen, emb_gt).clamp(0.0, 1.0).item())

    @torch.no_grad()
    def compute_dino_refs(
        self,
        generated: Image.Image,
        reference_crops: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        localized_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> float:
        """
        Compute DINO_refs: localize each reference object in `generated` with Grounding DINO,
        crop the localized region, and average DINO cosine similarity against canonicalized R.
        """
        if not reference_crops:
            return 0.0

        w, h = generated.size
        boxes = (
            list(localized_boxes)
            if localized_boxes is not None
            else self.detector.localize_references(generated, reference_descriptions)
        )
        if not boxes:
            return 0.0

        sims: List[float] = []
        for idx, ref_img in enumerate(reference_crops):
            if idx >= len(boxes):
                break
            x1, y1, x2, y2 = boxes[idx].to_int_xyxy(max_w=w, max_h=h)
            gen_crop = generated.crop((x1, y1, x2, y2))
            sim = float(
                torch.dot(
                    self.extract_dino_feature(gen_crop),
                    self.extract_dino_feature(ref_img),
                )
                .clamp(0.0, 1.0)
                .item()
            )
            sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0

    @torch.no_grad()
    def compute_clip_scores(
        self,
        generated: Image.Image,
        ground_truth: Image.Image,
        prompt: str,
    ) -> Dict[str, float]:
        """Compute CLIP-I (image-image similarity) and CLIP-T (image-text alignment)."""
        if self.use_mock_metrics:
            emb_gen = self._mock_image_embedding(generated, dim=256)
            emb_gt = self._mock_image_embedding(ground_truth, dim=256)
            clip_i = float(torch.dot(emb_gen, emb_gt).clamp(0.0, 1.0).item())
            # Deterministic text hash vector scaled to realistic CLIP-T cosine range (~0.30-0.34)
            gen_rng = torch.Generator(device="cpu").manual_seed(sum(ord(c) for c in prompt) % 10007)
            txt_vec = F.normalize(torch.randn((256,), generator=gen_rng), dim=0)
            raw_t = abs(float(torch.dot(emb_gen, txt_vec).item()))
            clip_t = 0.28 + 0.06 * min(1.0, raw_t * 3.0)
            return {"CLIP-I": clip_i, "CLIP-T": clip_t}

        self._load_clip()
        assert self._clip_proc is not None and self._clip_model is not None

        img_inputs = self._clip_proc(
            images=[generated.convert("RGB"), ground_truth.convert("RGB")],
            return_tensors="pt",
        ).to(self.device)
        img_feats = self._clip_model.get_image_features(**img_inputs).float()
        img_feats = F.normalize(img_feats, dim=-1)
        clip_i = float(torch.dot(img_feats[0], img_feats[1]).clamp(0.0, 1.0).item())

        txt_inputs = self._clip_proc(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            max_length=77,
            padding=True,
        ).to(self.device)
        txt_feat = self._clip_model.get_text_features(**txt_inputs).float()
        txt_feat = F.normalize(txt_feat, dim=-1)
        clip_t = float(torch.dot(img_feats[0], txt_feat[0]).clamp(0.0, 1.0).item())

        return {"CLIP-I": clip_i, "CLIP-T": clip_t}

    @torch.no_grad()
    def compute_pickscore(self, generated: Image.Image, prompt: str) -> float:
        """
        Compute PickScore (Kirstain et al., NeurIPS 2023) unnormalized cosine preference
        (matches Table 2 range ~0.218 - 0.231).
        """
        if self.use_mock_metrics:
            emb = self._mock_image_embedding(generated, dim=128)
            contrast = float(emb.std().item())
            return float(np.clip(0.215 + 0.25 * contrast, 0.20, 0.25))

        self._load_pickscore()
        assert self._pick_proc is not None and self._pick_model is not None

        image_inputs = self._pick_proc(
            images=generated.convert("RGB"),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_inputs = self._pick_proc(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        img_emb = self._pick_model.get_image_features(**image_inputs).float()
        img_emb = F.normalize(img_emb, dim=-1)

        txt_emb = self._pick_model.get_text_features(**text_inputs).float()
        txt_emb = F.normalize(txt_emb, dim=-1)

        # Raw normalized dot product reported in PickScore / LayoutSAM-Eval (~0.22)
        return float(torch.sum(img_emb * txt_emb, dim=-1)[0].item())

    def evaluate_sample(
        self,
        generated: Image.Image,
        ground_truth: Image.Image,
        prompt: str,
        reference_crops: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        localized_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> Dict[str, float]:
        pick = self.compute_pickscore(generated, prompt)
        dino_global = self.compute_dino_global(generated, ground_truth)
        dino_refs = self.compute_dino_refs(
            generated=generated,
            reference_crops=reference_crops,
            reference_descriptions=reference_descriptions,
            localized_boxes=localized_boxes,
        )
        clip_dict = self.compute_clip_scores(generated, ground_truth, prompt)
        return {
            "PICK": pick,
            "DINO": dino_global,
            "DINO_refs": dino_refs,
            "CLIP-I": clip_dict["CLIP-I"],
            "CLIP-T": clip_dict["CLIP-T"],
        }

