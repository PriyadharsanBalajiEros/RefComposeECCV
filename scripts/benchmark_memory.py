#!/usr/bin/env python3
"""
CLI: Peak GPU Memory (GB) & Latency Scaling Benchmark (Paper Section 4.4, Table 1).

Usage:
  # Full H100 VRAM scaling benchmark at 1280x720 across N=2, 4, 6, 8, 12, 15:
  python scripts/benchmark_memory.py --config configs/benchmark_memory.yaml

  # Offline verification run:
  python scripts/benchmark_memory.py --config configs/benchmark_memory.yaml --ref-counts 2 4 6 --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refcompose.benchmarks.memory_profiler import GPUMemoryBenchmarkSuite
from refcompose.config import load_config
from refcompose.utils.logging_utils import set_global_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RefCompose Peak GPU Memory Scaling Benchmark (Table 1)")
    parser.add_argument("--config", type=str, default="configs/benchmark_memory.yaml", help="Benchmark YAML")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to profile")
    parser.add_argument("--ref-counts", nargs="+", type=int, default=None, help="Reference counts N to test")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output folder")
    parser.add_argument("--mock", action="store_true", help="Run with mock models for offline verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.mock:
        overrides["resolution"] = {"width": 320, "height": 192, "vae_scale_factor": 8, "patch_size": 2}
        overrides.setdefault("memory_benchmark", {})["width"] = 320
        overrides.setdefault("memory_benchmark", {})["height"] = 192
        overrides.setdefault("memory_benchmark", {})["timed_runs"] = 1
        overrides.setdefault("memory_benchmark", {})["warmup_runs"] = 0
    if args.output_dir:
        overrides.setdefault("memory_benchmark", {})["output_dir"] = args.output_dir

    cfg = load_config(args.config, overrides=overrides)
    set_global_seed(cfg.sampling.seed)
    logger = setup_logger("refcompose.benchmark_memory")

    suite = GPUMemoryBenchmarkSuite(cfg=cfg, use_mock=args.mock)
    results = suite.run_suite(methods=args.methods, reference_counts=args.ref_counts)

    cols = args.ref_counts or cfg.memory_benchmark.table1_reference_counts
    exported = suite.export_reports(
        results=results,
        output_dir=cfg.memory_benchmark.output_dir,
        table1_columns=cols,
    )

    print("\n=== Table 1: Peak GPU Memory (GB) at Inference ===\n")
    print(suite.format_table1_markdown(results, ref_columns=cols))
    logger.info("Saved Table 1 reports to %s", exported["markdown"].parent)


if __name__ == "__main__":
    main()

