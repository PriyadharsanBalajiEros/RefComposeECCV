"""
Quantitative Benchmark Evaluator on Dense Layout (Paper Section 4.4, Table 2).
Evaluates RefCompose (N=5 and N>5) alongside CreatiLayout, InstanceAssemble, UNO, and XVerse,
computing all 9 region-wise and global-wise metrics and exporting JSON, CSV, Markdown, and LaTeX tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from refcompose.utils.logging_utils import progress_bar as tqdm

from refcompose.config import RefComposeConfig
from refcompose.data.denselayout_loader import DenseLayoutBenchmarkLoader
from refcompose.evaluation.metrics_global import GlobalMetricsSuite
from refcompose.evaluation.metrics_region import RegionMetricsSuite
from refcompose.models.detector import GroundingDINOLocalizer
from refcompose.models.vlm_annotator import Qwen3VLAnnotator


TABLE2_METRIC_COLUMNS = [
    "Spatial",
    "Color",
    "Texture",
    "Shape",
    "PICK",
    "DINO",
    "DINO_refs",
    "CLIP-I",
    "CLIP-T",
]

METHOD_DISPLAY_NAMES = {
    "creatilayout": "CreatiLayout [22]",
    "instanceassemble": "InstanceAssemble [29]",
    "uno": "UNO [27]",
    "xverse": "XVerse [5]",
    "refcompose_n5": "RefCompose (N=5)",
    "refcompose_n_gt_5": "RefCompose (N>5)",
}


class DenseLayoutBenchmarkEvaluator:
    """Runs the complete Section 4.4 / Table 2 evaluation protocol across methods."""

    def __init__(
        self,
        cfg: RefComposeConfig,
        loader: DenseLayoutBenchmarkLoader,
        use_mock_metrics: bool = False,
    ) -> None:
        self.cfg = cfg
        self.loader = loader
        self.detector = GroundingDINOLocalizer(
            repo_id=cfg.model.grounding_dino_repo_id,
            cfg=cfg.detection,
            use_mock_detector=use_mock_metrics,
        )
        self.vlm = (
            Qwen3VLAnnotator(
                repo_id=cfg.model.qwen3_vl_repo_id,
                dtype=cfg.model.torch_dtype,
                use_mock_vlm=use_mock_metrics,
            )
            if cfg.evaluation.region_eval_mode == "hybrid"
            else None
        )
        self.region_suite = RegionMetricsSuite(
            detector=self.detector,
            vlm=self.vlm,
            eval_mode=cfg.evaluation.region_eval_mode,
        )
        self.global_suite = GlobalMetricsSuite(
            cfg=cfg,
            detector=self.detector,
            use_mock_metrics=use_mock_metrics,
        )

    def evaluate_method(
        self,
        method_key: str,
        runner: Any,
        max_samples: Optional[int] = None,
        save_images_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluate one `BaseBenchmarkMethod` runner over the DenseLayout test split."""
        is_scalable_n_gt_5 = method_key == "refcompose_n_gt_5"
        max_refs = (
            self.cfg.evaluation.scalable_max_refs
            if is_scalable_n_gt_5
            else self.cfg.evaluation.baseline_max_refs
        )
        total = min(len(self.loader), max_samples or self.cfg.evaluation.num_eval_samples)

        per_metric_values: Dict[str, List[float]] = {m: [] for m in TABLE2_METRIC_COLUMNS}

        if save_images_dir is not None:
            save_images_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(total), desc=f"Evaluating {method_key}"):
            sample = self.loader.get_sample(
                index=idx,
                max_refs=max_refs,
                require_gt_more_than_5=is_scalable_n_gt_5,
            )
            if sample is None:
                continue

            gen_result = runner.generate(sample)
            gen_img = gen_result["image"]
            expected_boxes = gen_result.get("expected_boxes", sample.gt_boxes)

            if save_images_dir is not None:
                gen_img.save(save_images_dir / f"{sample.sample_id}.png")

            reg_scores, localized_boxes = self.region_suite.evaluate_sample(
                generated=gen_img,
                reference_crops=sample.reference_crops,
                reference_descriptions=sample.reference_descriptions,
                target_boxes=sample.gt_boxes,
                expected_boxes=expected_boxes,
            )
            glob_scores = self.global_suite.evaluate_sample(
                generated=gen_img,
                ground_truth=sample.ground_truth_image,
                prompt=sample.prompt,
                reference_crops=sample.reference_crops,
                reference_descriptions=sample.reference_descriptions,
                localized_boxes=localized_boxes,
            )

            merged = {**reg_scores, **glob_scores}
            for m in TABLE2_METRIC_COLUMNS:
                per_metric_values[m].append(float(merged[m]))

        return {
            m: (float(np.mean(vals)) if vals else 0.0)
            for m, vals in per_metric_values.items()
        }

    @staticmethod
    def compute_improvement_row(
        results: Dict[str, Dict[str, float]],
        target_key: str = "refcompose_n_gt_5",
        reference_baseline_keys: Sequence[str] = ("uno", "xverse"),
    ) -> Optional[Dict[str, float]]:
        """
        Compute the `vs. best reference baseline` row in Paper Table 2:
        compares `RefCompose (N>5)` (or `RefCompose (N=5)` if N>5 wasn't run)
        against the strongest reference-image baseline (UNO / XVerse) per metric.
        """
        tgt = results.get(target_key) or results.get("refcompose_n5")
        avail_refs = [results[k] for k in reference_baseline_keys if k in results]
        if tgt is None or not avail_refs:
            return None

        deltas: Dict[str, float] = {}
        for m in TABLE2_METRIC_COLUMNS:
            best_ref = max(r[m] for r in avail_refs)
            deltas[m] = tgt[m] - best_ref
        return deltas

    @staticmethod
    def format_markdown_table(results: Dict[str, Dict[str, float]]) -> str:
        """Render Table 2 in clean GitHub-Flavored Markdown."""
        header = "| Method | " + " | ".join(f"{m} ↑" for m in TABLE2_METRIC_COLUMNS) + " |"
        sep = "| :--- | " + " | ".join("---:" for _ in TABLE2_METRIC_COLUMNS) + " |"
        lines = [header, sep]

        deltas = DenseLayoutBenchmarkEvaluator.compute_improvement_row(results)

        ordered_keys = [
            "creatilayout",
            "instanceassemble",
            "uno",
            "xverse",
            "refcompose_n5",
            "refcompose_n_gt_5",
        ]
        present_keys = [k for k in ordered_keys if k in results] + [
            k for k in results.keys() if k not in ordered_keys
        ]

        # Find column maxima for bolding
        col_max = {
            m: max((results[k][m] for k in present_keys), default=0.0)
            for m in TABLE2_METRIC_COLUMNS
        }

        for k in present_keys:
            if k == "refcompose_n_gt_5" and deltas is not None:
                delta_cells = " | ".join(f"{deltas[m]:+.3f}" for m in TABLE2_METRIC_COLUMNS)
                lines.append(f"| *vs. best reference baseline* | {delta_cells} |")

            disp = METHOD_DISPLAY_NAMES.get(k, k)
            cells = []
            for m in TABLE2_METRIC_COLUMNS:
                val = results[k][m]
                formatted = f"{val:.3f}"
                if abs(val - col_max[m]) < 1e-6 and len(present_keys) > 1:
                    formatted = f"**{formatted}**"
                cells.append(formatted)
            lines.append(f"| {disp} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def format_latex_table(results: Dict[str, Dict[str, float]]) -> str:
        """Render Paper Table 2 as publication-ready LaTeX `booktabs` source."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Quantitative comparison on the Dense Layout test set.}",
            r"\begin{tabular}{lcccc ccccc}",
            r"\toprule",
            r"\textbf{Method} & \multicolumn{4}{c}{\textbf{Region-wise Quality}} & \multicolumn{5}{c}{\textbf{Global-wise Quality}} \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-10}",
            r" & Spatial$\uparrow$ & Color$\uparrow$ & Texture$\uparrow$ & Shape$\uparrow$ & PICK$\uparrow$ & DINO$\uparrow$ & $\text{DINO}_{\text{refs}}\uparrow$ & CLIP-I$\uparrow$ & CLIP-T$\uparrow$ \\",
            r"\midrule",
        ]
        for k, row in results.items():
            disp = METHOD_DISPLAY_NAMES.get(k, k)
            vals = " & ".join(f"{row[m]:.3f}" for m in TABLE2_METRIC_COLUMNS)
            lines.append(f"{disp} & {vals} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    @staticmethod
    def export_all_reports(
        results: Dict[str, Dict[str, float]],
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "table2_metrics.json"
        csv_path = out / "table2_metrics.csv"
        md_path = out / "table2_metrics.md"
        tex_path = out / "table2_metrics.tex"

        payload: Dict[str, Any] = {"methods": results}
        deltas = DenseLayoutBenchmarkEvaluator.compute_improvement_row(results)
        if deltas is not None:
            payload["vs_best_reference_baseline"] = deltas

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["method"] + TABLE2_METRIC_COLUMNS)
            for k, row in results.items():
                writer.writerow([k] + [f"{row[m]:.4f}" for m in TABLE2_METRIC_COLUMNS])

        md_path.write_text(DenseLayoutBenchmarkEvaluator.format_markdown_table(results) + "\n", encoding="utf-8")
        tex_path.write_text(DenseLayoutBenchmarkEvaluator.format_latex_table(results) + "\n", encoding="utf-8")

        return {
            "json": json_path,
            "csv": csv_path,
            "markdown": md_path,
            "latex": tex_path,
        }
