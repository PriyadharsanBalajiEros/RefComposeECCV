#!/usr/bin/env python3
"""
CLI: Quantitative Evaluation on Dense Layout Benchmark (Paper Section 4.4, Table 2).

Computes all 9 metrics:
  Region-wise (LayoutSAM-Eval): Spatial ↑, Color ↑, Texture ↑, Shape ↑
  Global-wise:                  PICK ↑, DINO ↑, DINO_refs ↑, CLIP-I ↑, CLIP-T ↑

Usage:
  # Full Table 2 evaluation across RefCompose (N=5, N>5) and all baselines:
  python scripts/evaluate.py --config configs/eval_denselayout.yaml

  # Offline verification run:
  python scripts/evaluate.py --config configs/eval_denselayout.yaml --max-samples 4 --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refcompose.benchmarks.baselines import build_benchmark_runner
from refcompose.config import load_config
from refcompose.data.denselayout_loader import DenseLayoutBenchmarkLoader
from refcompose.evaluation.evaluator import DenseLayoutBenchmarkEvaluator
from refcompose.models.flux2_wrapper import Flux2BackboneWrapper
from refcompose.models.vlm_annotator import Qwen3VLAnnotator
from refcompose.utils.logging_utils import set_global_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RefCompose DenseLayout Quantitative Evaluation (Table 2)")
    parser.add_argument("--config", type=str, default="configs/eval_denselayout.yaml", help="Eval YAML config")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap evaluation sample count")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output report directory")
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default=None,
        help="Optional root containing `<method>/<sample_id>.png` pre-generated images",
    )
    parser.add_argument("--mock", action="store_true", help="Run with mock models & synthetic DenseLayout split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.mock:
        overrides["resolution"] = {"width": 320, "height": 192, "vae_scale_factor": 8, "patch_size": 2}
    if args.output_dir:
        overrides.setdefault("evaluation", {})["output_dir"] = args.output_dir

    cfg = load_config(args.config, overrides=overrides)
    set_global_seed(cfg.sampling.seed)
    logger = setup_logger("refcompose.evaluate")

    canonicalizer = Flux2BackboneWrapper(
        cfg=cfg,
        use_mock_backbone=args.mock,
        inject_lora=False,
    )
    vlm = Qwen3VLAnnotator(
        repo_id=cfg.model.qwen3_vl_repo_id,
        dtype=cfg.model.torch_dtype,
        use_mock_vlm=args.mock,
    )
    loader = DenseLayoutBenchmarkLoader(
        cfg=cfg,
        canonicalizer=canonicalizer,
        vlm=vlm,
        use_mock_data=args.mock,
    )
    evaluator = DenseLayoutBenchmarkEvaluator(
        cfg=cfg,
        loader=loader,
        use_mock_metrics=args.mock,
    )

    methods = args.methods or cfg.evaluation.methods
    out_dir = Path(cfg.evaluation.output_dir)
    results = {}

    for m_key in methods:
        pre_dir = None
        if args.precomputed_dir is not None:
            candidate = Path(args.precomputed_dir) / m_key
            if candidate.exists():
                pre_dir = candidate

        runner = build_benchmark_runner(
            method_key=m_key,
            cfg=cfg,
            use_mock=args.mock,
            precomputed_dir=pre_dir,
        )
        scores = evaluator.evaluate_method(
            method_key=m_key,
            runner=runner,
            max_samples=args.max_samples,
            save_images_dir=out_dir / "generated" / m_key,
        )
        results[m_key] = scores
        logger.info("Finished %s: %s", m_key, {k: round(v, 3) for k, v in scores.items()})

    exported = evaluator.export_all_reports(results, output_dir=out_dir)
    print("\n=== Table 2: Quantitative Comparison on Dense Layout ===\n")
    print(evaluator.format_markdown_table(results))
    logger.info("Saved JSON/CSV/Markdown/LaTeX evaluation tables to %s", exported["markdown"].parent)


if __name__ == "__main__":
    main()

