"""
Unified Benchmark Runners for RefCompose and All Paper Baselines (Section 4.4, Tables 1 & 2):
  - RefCompose (`refcompose`, `refcompose_n5`, `refcompose_n_gt_5`)
  - Layout-only baselines (boxes + text, no reference pixels):
      * CreatiLayout [22] (`HuiZhang0812/CreatiLayout`)
      * InstanceAssemble [29] (`FireRedTeam/InstanceAssemble`)
  - Reference-image token baselines (prompt + canonicalized reference crops):
      * UNO [27] (`bytedance-research/UNO`)
      * XVerse [5] (`ByteDance/XVerse`)
  - Precomputed image directory runner (`PrecomputedDirectoryRunner`) for offline baseline outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw
import torch

from refcompose.canvas.composer import BoundingBox
from refcompose.config import RefComposeConfig
from refcompose.data.denselayout_loader import BenchmarkInstanceSample
from refcompose.pipeline.refcompose_pipeline import RefComposePipeline


class BaseBenchmarkMethod(ABC):
    """Abstract interface implemented by RefCompose and all baseline methods."""

    def __init__(self, name: str, cfg: RefComposeConfig) -> None:
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        """
        Generate a 1280x720 composite for `sample`.
        Returns `{"image": PIL.Image, "expected_boxes": List[BoundingBox], ...}`.
        """


class RefComposeBenchmarkRunner(BaseBenchmarkMethod):
    """Benchmark runner wrapping `RefComposePipeline` (constant memory across N)."""

    def __init__(
        self,
        name: str,
        cfg: RefComposeConfig,
        lora_dir: Optional[str | Path] = None,
        use_mock: bool = False,
    ) -> None:
        super().__init__(name=name, cfg=cfg)
        ckpt = lora_dir or cfg.evaluation.checkpoints.get("refcompose_lora_dir")
        self.pipeline = RefComposePipeline(
            cfg=cfg,
            lora_checkpoint_dir=ckpt,
            use_mock_models=use_mock,
        )

    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        out = self.pipeline(
            references=sample.reference_crops,
            reference_descriptions=sample.reference_descriptions,
            prompt=sample.prompt,
            seed=self.cfg.sampling.seed,
        )
        return {
            "image": out.generated_image,
            "expected_boxes": out.localized_boxes,
            "canvas": out.canvas_image,
            "depth_vis": out.depth_vis_image,
        }


class CreatiLayoutRunner(BaseBenchmarkMethod):
    """
    Layout-only baseline: CreatiLayout (Tang et al., 2024; `HuiZhang0812/CreatiLayout`).
    Receives ONLY Dense Layout ground-truth boxes + per-region captions + global prompt
    (no visual reference crops, per Paper Section 4.4).
    """

    def __init__(self, cfg: RefComposeConfig, use_mock: bool = False) -> None:
        super().__init__(name="creatilayout", cfg=cfg)
        self.use_mock = use_mock
        self._pipe = None

    def _lazy_load(self) -> None:
        if self.use_mock or self._pipe is not None:
            return
        from diffusers import DiffusionPipeline

        repo_id = self.cfg.evaluation.checkpoints["creatilayout_repo_id"]
        self._pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=self.cfg.model.torch_dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self._pipe.to("cuda")

    @torch.no_grad()
    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        w, h = self.cfg.resolution.width, self.cfg.resolution.height
        if self.use_mock:
            canvas = Image.new("RGB", (w, h), color=(60, 65, 75))
            draw = ImageDraw.Draw(canvas)
            for idx, box in enumerate(sample.gt_boxes):
                x1, y1, x2, y2 = box.to_int_xyxy(w, h)
                # Layout-only places boxes accurately but hallucinates photometric color/texture
                draw.rectangle([x1, y1, x2, y2], fill=(110 + (idx * 23) % 100, 120, 130))
            return {"image": canvas, "expected_boxes": sample.gt_boxes}

        self._lazy_load()
        assert self._pipe is not None
        norm_boxes = [list(b.to_normalized_xyxy(w, h)) for b in sample.gt_boxes]
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.sampling.seed)
        img = self._pipe(
            prompt=[sample.prompt],
            bbox_phrases=sample.reference_descriptions,
            bbox_raw=norm_boxes,
            height=h,
            width=w,
            num_inference_steps=self.cfg.sampling.num_inference_steps,
            guidance_scale=self.cfg.sampling.guidance_scale,
            generator=gen,
        ).images[0]
        return {"image": img, "expected_boxes": sample.gt_boxes}


class InstanceAssembleRunner(BaseBenchmarkMethod):
    """
    Layout-only baseline: InstanceAssemble (Xiang et al., NeurIPS 2025; `FireRedTeam/InstanceAssemble`).
    Receives ONLY Dense Layout ground-truth boxes + per-instance text + global prompt.
    """

    def __init__(self, cfg: RefComposeConfig, use_mock: bool = False) -> None:
        super().__init__(name="instanceassemble", cfg=cfg)
        self.use_mock = use_mock
        self._pipe = None

    def _lazy_load(self) -> None:
        if self.use_mock or self._pipe is not None:
            return
        from diffusers import DiffusionPipeline

        repo_id = self.cfg.evaluation.checkpoints["instanceassemble_repo_id"]
        self._pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=self.cfg.model.torch_dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self._pipe.to("cuda")

    @torch.no_grad()
    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        w, h = self.cfg.resolution.width, self.cfg.resolution.height
        if self.use_mock:
            canvas = Image.new("RGB", (w, h), color=(55, 62, 72))
            draw = ImageDraw.Draw(canvas)
            for idx, box in enumerate(sample.gt_boxes):
                x1, y1, x2, y2 = box.to_int_xyxy(w, h)
                draw.rectangle([x1, y1, x2, y2], fill=(130, 105 + (idx * 29) % 110, 125))
            return {"image": canvas, "expected_boxes": sample.gt_boxes}

        self._lazy_load()
        assert self._pipe is not None
        norm_boxes = [list(b.to_normalized_xyxy(w, h)) for b in sample.gt_boxes]
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.sampling.seed)
        img = self._pipe(
            prompt=sample.prompt,
            phrases=sample.reference_descriptions,
            boxes=norm_boxes,
            height=h,
            width=w,
            num_inference_steps=self.cfg.sampling.num_inference_steps,
            guidance_scale=self.cfg.sampling.guidance_scale,
            generator=gen,
        ).images[0]
        return {"image": img, "expected_boxes": sample.gt_boxes}


class UNORunner(BaseBenchmarkMethod):
    """
    Reference-image token concatenation baseline: UNO (Wu et al., ICCV 2025; `bytedance-research/UNO`).
    Encodes each reference image independently and concatenates N token streams into attention,
    causing GPU memory to grow linearly with N (OOM at N=6 on 1x H100 80GB at 1280x720).
    """

    def __init__(self, cfg: RefComposeConfig, use_mock: bool = False) -> None:
        super().__init__(name="uno", cfg=cfg)
        self.use_mock = use_mock
        self._pipe = None

    def _lazy_load(self) -> None:
        if self.use_mock or self._pipe is not None:
            return
        from diffusers import DiffusionPipeline

        repo_id = self.cfg.evaluation.checkpoints["uno_repo_id"]
        self._pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=self.cfg.model.torch_dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self._pipe.to("cuda")

    @torch.no_grad()
    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        w, h = self.cfg.resolution.width, self.cfg.resolution.height
        if self.use_mock:
            canvas = Image.new("RGB", (w, h), color=(50, 50, 58))
            for idx, (ref, box) in enumerate(zip(sample.reference_crops, sample.gt_boxes)):
                x1, y1, x2, y2 = box.to_int_xyxy(w, h)
                resized = ref.convert("RGB").resize((max(1, x2 - x1), max(1, y2 - y1)))
                canvas.paste(resized, (x1, y1))
            return {"image": canvas, "expected_boxes": sample.gt_boxes}

        self._lazy_load()
        assert self._pipe is not None
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.sampling.seed)
        img = self._pipe(
            prompt=sample.prompt,
            ref_imgs=list(sample.reference_crops),
            height=h,
            width=w,
            num_inference_steps=self.cfg.sampling.num_inference_steps,
            guidance_scale=self.cfg.sampling.guidance_scale,
            generator=gen,
        ).images[0]
        return {"image": img, "expected_boxes": sample.gt_boxes}


class XVerseRunner(BaseBenchmarkMethod):
    """
    Reference-image baseline: XVerse (Chen et al., 2025; `ByteDance/XVerse`).
    Multi-subject DiT modulation & per-reference encoding; scales linearly with N in VRAM.
    """

    def __init__(self, cfg: RefComposeConfig, use_mock: bool = False) -> None:
        super().__init__(name="xverse", cfg=cfg)
        self.use_mock = use_mock
        self._pipe = None

    def _lazy_load(self) -> None:
        if self.use_mock or self._pipe is not None:
            return
        from diffusers import DiffusionPipeline

        repo_id = self.cfg.evaluation.checkpoints["xverse_repo_id"]
        self._pipe = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=self.cfg.model.torch_dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self._pipe.to("cuda")

    @torch.no_grad()
    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        w, h = self.cfg.resolution.width, self.cfg.resolution.height
        if self.use_mock:
            canvas = Image.new("RGB", (w, h), color=(48, 52, 60))
            for ref, box in zip(sample.reference_crops, sample.gt_boxes):
                x1, y1, x2, y2 = box.to_int_xyxy(w, h)
                resized = ref.convert("RGB").resize((max(1, x2 - x1), max(1, y2 - y1)))
                canvas.paste(resized, (x1, y1))
            return {"image": canvas, "expected_boxes": sample.gt_boxes}

        self._lazy_load()
        assert self._pipe is not None
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.sampling.seed)
        img = self._pipe(
            prompt=sample.prompt,
            subject_images=list(sample.reference_crops),
            subject_words=list(sample.reference_descriptions),
            height=h,
            width=w,
            num_inference_steps=self.cfg.sampling.num_inference_steps,
            guidance_scale=self.cfg.sampling.guidance_scale,
            generator=gen,
        ).images[0]
        return {"image": img, "expected_boxes": sample.gt_boxes}


class PrecomputedDirectoryRunner(BaseBenchmarkMethod):
    """Loads pre-rendered `{sample_id}.png` outputs generated by external baseline repositories."""

    def __init__(self, name: str, cfg: RefComposeConfig, images_dir: str | Path) -> None:
        super().__init__(name=name, cfg=cfg)
        self.images_dir = Path(images_dir)

    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = self.images_dir / f"{sample.sample_id}{ext}"
            if candidate.exists():
                img = Image.open(candidate).convert("RGB").resize(
                    (self.cfg.resolution.width, self.cfg.resolution.height),
                    Image.Resampling.BICUBIC,
                )
                return {"image": img, "expected_boxes": sample.gt_boxes}
        raise FileNotFoundError(
            f"Precomputed image for sample '{sample.sample_id}' not found in {self.images_dir}"
        )


def build_benchmark_runner(
    method_key: str,
    cfg: RefComposeConfig,
    use_mock: bool = False,
    precomputed_dir: Optional[str | Path] = None,
) -> BaseBenchmarkMethod:
    """Instantiate a benchmark runner by method identifier."""
    if precomputed_dir is not None:
        return PrecomputedDirectoryRunner(name=method_key, cfg=cfg, images_dir=precomputed_dir)

    key = method_key.lower()
    if key in ("refcompose", "refcompose_n5", "refcompose_n_gt_5"):
        return RefComposeBenchmarkRunner(name=key, cfg=cfg, use_mock=use_mock)
    if key == "creatilayout":
        return CreatiLayoutRunner(cfg=cfg, use_mock=use_mock)
    if key == "instanceassemble":
        return InstanceAssembleRunner(cfg=cfg, use_mock=use_mock)
    if key == "uno":
        return UNORunner(cfg=cfg, use_mock=use_mock)
    if key == "xverse":
        return XVerseRunner(cfg=cfg, use_mock=use_mock)
    raise ValueError(f"Unknown benchmark method '{method_key}'.")

