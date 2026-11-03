"""
Qualitative & Quantitative Ablation Studies (Paper Section 4.5, Section 4.6, Figures 4, 5, 6, 7):
  1. Ablation Study (Figure 7 & Section 4.6):
       - `wo_canvas`:          w/o reference canvas (black canvas C=0, depth LoRA only)
       - `wo_depth`:           w/o depth map prior (reference canvas LoRA only)
       - `merged_single_lora`: merged depth + canvas through a single LoRA stream (p. 12)
       - `full`:               full RefCompose (disentangled dual-stream DSA LoRA)
  2. Multi-Reference Flexibility (Figure 5):
       - Shared prompt p, layout L, and depth map D_s across two distinct reference sets R_1, R_2.
  3. Depth-Controlled Orientation (Figure 6):
       - Identical reference assets R under a conventional static depth map vs a chaotic
         floating-debris depth map.
  4. Qualitative Multi-Method Grid (Figure 4):
       - Side-by-side comparison grid across Ground Truth, RefCompose, CreatiLayout,
         InstanceAssemble, UNO, and XVerse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from PIL import Image

from refcompose.benchmarks.baselines import build_benchmark_runner
from refcompose.config import RefComposeConfig
from refcompose.data.denselayout_loader import BenchmarkInstanceSample, DenseLayoutBenchmarkLoader
from refcompose.evaluation.evaluator import DenseLayoutBenchmarkEvaluator, TABLE2_METRIC_COLUMNS
from refcompose.pipeline.refcompose_pipeline import RefComposePipeline
from refcompose.utils.image_ops import make_comparison_grid


ABLATION_MODES = (
    "wo_canvas",
    "wo_depth",
    "merged_single_lora",
    "full",
)

ABLATION_DISPLAY_LABELS = {
    "wo_canvas": "w/o reference canvas",
    "wo_depth": "w/o depth map prior",
    "merged_single_lora": "single merged LoRA stream",
    "full": "with reference canvas and depth map prior",
}


class _AblationPipelineAdapter:
    """Adapts `RefComposePipeline` with a fixed `ablation_mode` to `BaseBenchmarkMethod`."""

    def __init__(self, pipeline: RefComposePipeline, ablation_mode: str, seed: int = 42) -> None:
        self.pipeline = pipeline
        self.ablation_mode = ablation_mode
        self.seed = seed

    def generate(self, sample: BenchmarkInstanceSample) -> Dict[str, Any]:
        out = self.pipeline(
            references=sample.reference_crops,
            reference_descriptions=sample.reference_descriptions,
            prompt=sample.prompt,
            ablation_mode=self.ablation_mode,
            seed=self.seed,
        )
        return {
            "image": out.generated_image,
            "expected_boxes": out.localized_boxes,
        }


class AblationAndQualitativeSuite:
    """Runs Section 4.5 qualitative experiments and Section 4.6 quantitative/visual ablations."""

    def __init__(
        self,
        cfg: RefComposeConfig,
        lora_checkpoint_dir: Optional[str | Path] = None,
        use_mock: bool = False,
    ) -> None:
        self.cfg = cfg
        self.use_mock = use_mock
        self.pipeline = RefComposePipeline(
            cfg=cfg,
            lora_checkpoint_dir=lora_checkpoint_dir,
            use_mock_models=use_mock,
        )

    def run_figure7_ablation_study(
        self,
        loader: DenseLayoutBenchmarkLoader,
        output_dir: str | Path,
        num_samples: int = 10,
        modes: Sequence[str] = ABLATION_MODES,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all ablation variants quantitatively on `num_samples` composites and save
        Figure 7 side-by-side comparison panels.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        evaluator = DenseLayoutBenchmarkEvaluator(
            cfg=self.cfg,
            loader=loader,
            use_mock_metrics=self.use_mock,
        )

        metrics_by_mode: Dict[str, Dict[str, float]] = {}
        for mode in modes:
            adapter = _AblationPipelineAdapter(
                pipeline=self.pipeline,
                ablation_mode=mode,
                seed=self.cfg.sampling.seed,
            )
            metrics_by_mode[mode] = evaluator.evaluate_method(
                method_key=mode,
                runner=adapter,
                max_samples=num_samples,
                save_images_dir=out_path / mode,
            )

        # Render visual Figure 7 panel for sample 0 ( sharing exact layout & boxes across modes )
        sample0 = loader.get_sample(0, max_refs=self.cfg.evaluation.baseline_max_refs)
        if sample0 is not None:
            full_ref = self.pipeline(
                references=sample0.reference_crops,
                reference_descriptions=sample0.reference_descriptions,
                prompt=sample0.prompt,
                ablation_mode="full",
                seed=self.cfg.sampling.seed,
            )
            panel_imgs: List[Image.Image] = []
            headers: List[str] = []
            for mode in modes:
                headers.append(ABLATION_DISPLAY_LABELS.get(mode, mode))
                if mode == "full":
                    panel_imgs.append(full_ref.generated_image)
                else:
                    res = self.pipeline(
                        references=sample0.reference_crops,
                        reference_descriptions=sample0.reference_descriptions,
                        prompt=sample0.prompt,
                        override_layout_image=full_ref.layout_image,
                        override_boxes=full_ref.localized_boxes,
                        override_depth_map=full_ref.depth_map,
                        ablation_mode=mode,
                        seed=self.cfg.sampling.seed,
                    )
                    panel_imgs.append(res.generated_image)

            fig7_grid = make_comparison_grid([panel_imgs], column_headers=headers)
            fig7_grid.save(out_path / "figure7_ablation_grid.png")

        with open(out_path / "ablation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_by_mode, f, indent=2)

        # Markdown summary
        md_lines = [
            "| Ablation Variant | " + " | ".join(TABLE2_METRIC_COLUMNS) + " |",
            "| :--- | " + " | ".join("---:" for _ in TABLE2_METRIC_COLUMNS) + " |",
        ]
        for mode, row in metrics_by_mode.items():
            lbl = ABLATION_DISPLAY_LABELS.get(mode, mode)
            vals = " | ".join(f"{row[m]:.3f}" for m in TABLE2_METRIC_COLUMNS)
            md_lines.append(f"| {lbl} | {vals} |")
        (out_path / "ablation_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        return metrics_by_mode

    def run_figure5_multi_reference_swap(
        self,
        ref_set_a: Sequence[Image.Image],
        ref_set_b: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        prompt: str,
        output_dir: str | Path,
    ) -> Path:
        """Run Paper Figure 5: two distinct reference sets sharing one prompt-induced layout & depth."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        outputs = self.pipeline.generate_multi_reference_swap(
            reference_sets=[ref_set_a, ref_set_b],
            reference_descriptions=reference_descriptions,
            prompt=prompt,
            seed=self.cfg.sampling.seed,
        )
        rows = [
            [
                out.layout_image or out.canvas_image,
                out.depth_vis_image or out.canvas_image,
                out.canvas_image,
                out.generated_image,
            ]
            for out in outputs
        ]
        grid = make_comparison_grid(
            rows,
            column_headers=["T2I Layout L", "Depth Map D_s", "Reference Canvas C", "Generated Image"],
        )
        grid_file = out_path / "figure5_multi_reference_flexibility.png"
        grid.save(grid_file)
        return grid_file

    def run_figure6_depth_controlled_orientation(
        self,
        references: Sequence[Image.Image],
        reference_descriptions: Sequence[str],
        static_prompt: str,
        dynamic_prompt: str,
        output_dir: str | Path,
    ) -> Path:
        """
        Run Paper Figure 6: identical reference assets under (1) a conventional static depth map
        and (2) a dynamic scene depth map with floating foreground debris/objects.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        w, h = self.cfg.resolution.width, self.cfg.resolution.height

        # 1. Conventional ground-plane depth gradient (background=1.0 top, foreground=0.1 bottom)
        static_depth = np.linspace(1.0, 0.1, h, dtype=np.float32)[:, None].repeat(w, axis=1)

        # 2. Dynamic floating-debris depth map (scattered near-camera floating patches across mid-air)
        dynamic_depth = static_depth.copy()
        rng = np.random.default_rng(seed=2026)
        for _ in range(12):
            ph = int(rng.integers(max(24, h // 12), max(48, h // 6)))
            pw = int(rng.integers(max(24, w // 14), max(48, w // 7)))
            py = int(rng.integers(0, max(1, h - ph)))
            px = int(rng.integers(0, max(1, w - pw)))
            dynamic_depth[py : py + ph, px : px + pw] = float(rng.uniform(0.05, 0.30))

        outputs = self.pipeline.generate_depth_controlled(
            references=references,
            reference_descriptions=reference_descriptions,
            prompts=[static_prompt, dynamic_prompt],
            depth_maps=[static_depth, dynamic_depth],
            seed=self.cfg.sampling.seed,
        )

        rows = [
            [out.depth_vis_image or out.canvas_image, out.canvas_image, out.generated_image]
            for out in outputs
        ]
        grid = make_comparison_grid(
            rows,
            column_headers=["Depth Map D_s", "Reference Canvas C", "Generated Image"],
        )
        grid_file = out_path / "figure6_depth_controlled_orientation.png"
        grid.save(grid_file)
        return grid_file

    def run_figure4_qualitative_grid(
        self,
        loader: DenseLayoutBenchmarkLoader,
        output_dir: str | Path,
        num_scenes: int = 7,
    ) -> Path:
        """
        Render Paper Figure 4 qualitative multi-scene comparison across:
        Ground Truth | RefCompose | CreatiLayout | InstanceAssemble | UNO | XVerse.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        method_keys = ["creatilayout", "instanceassemble", "uno", "xverse"]
        runners = {k: build_benchmark_runner(k, self.cfg, use_mock=self.use_mock) for k in method_keys}

        rows: List[List[Image.Image]] = []
        count = min(num_scenes, len(loader))
        for idx in range(count):
            sample = loader.get_sample(idx, max_refs=self.cfg.evaluation.baseline_max_refs)
            if sample is None:
                continue
            rc_out = self.pipeline(
                references=sample.reference_crops,
                reference_descriptions=sample.reference_descriptions,
                prompt=sample.prompt,
                seed=self.cfg.sampling.seed + idx,
            )
            row = [
                sample.ground_truth_image,
                rc_out.generated_image,
                runners["creatilayout"].generate(sample)["image"],
                runners["instanceassemble"].generate(sample)["image"],
                runners["uno"].generate(sample)["image"],
                runners["xverse"].generate(sample)["image"],
            ]
            rows.append(row)

        grid = make_comparison_grid(
            rows,
            column_headers=[
                "Ground Truth",
                "RefCompose",
                "CreatiLayout",
                "InstanceAssemble",
                "UNO",
                "XVerse",
            ],
        )
        grid_file = out_path / "figure4_qualitative_comparison.png"
        grid.save(grid_file)
        return grid_file

