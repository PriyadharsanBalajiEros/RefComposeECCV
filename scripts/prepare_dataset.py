#!/usr/bin/env python3
"""
CLI: 5-Stage RefCompose Dataset Construction & Synthetic Corpus Generation (Paper Section 4.1).

Usage:
  # Full production 50%/50% cinematic + synthetic dataset curation:
  python scripts/prepare_dataset.py --config configs/data_prep.yaml --generate-synthetic

  # Quick offline dry-run verification:
  python scripts/prepare_dataset.py --config configs/data_prep.yaml --generate-synthetic --num-synthetic 4 --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refcompose.config import load_config
from refcompose.data.curation_pipeline import DatasetCurationPipeline
from refcompose.data.synthetic_generator import SyntheticSceneGenerator
from refcompose.utils.logging_utils import set_global_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RefCompose 5-Stage Dataset Curation Pipeline")
    parser.add_argument("--config", type=str, default="configs/data_prep.yaml", help="Path to YAML config")
    parser.add_argument("--raw-cinematic-dir", type=str, default=None, help="Directory of raw movie frames")
    parser.add_argument("--processed-dir", type=str, default=None, help="Output directory for manifests")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic FLUX.2 scenes")
    parser.add_argument("--num-synthetic", type=int, default=None, help="Override synthetic sample count")
    parser.add_argument("--max-cinematic", type=int, default=None, help="Override cinematic sample count")
    parser.add_argument("--mock", action="store_true", help="Use lightweight mock models for offline verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.sampling.seed)
    logger = setup_logger("refcompose.prepare_dataset")

    proc_dir = Path(args.processed_dir or cfg.curation.processed_output_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    curation = DatasetCurationPipeline(cfg=cfg, use_mock_models=args.mock)

    # 1. Optional Synthetic FLUX.2 Scene Generation + 5-Stage Curation (50% of corpus)
    if args.generate_synthetic:
        num_synth = args.num_synthetic or cfg.curation.target_synthetic_samples
        synth_raw_dir = Path(cfg.curation.synthetic_output_dir)
        logger.info("Generating %d synthetic FLUX.2 multi-object scenes into %s...", num_synth, synth_raw_dir)
        synth_gen = SyntheticSceneGenerator(cfg=cfg, use_mock_backbone=args.mock)
        synth_gen.generate_corpus(output_dir=synth_raw_dir, num_samples=num_synth, seed=cfg.sampling.seed)

        synth_manifest = proc_dir / "synthetic_manifest.jsonl"
        kept_synth = curation.curate_directory(
            input_dir=synth_raw_dir,
            manifest_path=synth_manifest,
            source_domain="synthetic",
            max_samples=num_synth,
        )
        logger.info("Retained %d curated synthetic samples -> %s", kept_synth, synth_manifest)

    # 2. Private Cinematic Movie Frames 5-Stage Curation (50% of corpus)
    cin_raw_dir = Path(args.raw_cinematic_dir or cfg.curation.raw_cinematic_dir)
    if cin_raw_dir.exists():
        cin_manifest = proc_dir / "cinematic_manifest.jsonl"
        kept_cin = curation.curate_directory(
            input_dir=cin_raw_dir,
            manifest_path=cin_manifest,
            source_domain="cinematic",
            max_samples=args.max_cinematic or cfg.curation.target_cinematic_samples,
        )
        logger.info("Retained %d curated cinematic samples -> %s", kept_cin, cin_manifest)
    else:
        logger.info("Raw cinematic directory %s not found; skipped cinematic pass.", cin_raw_dir)


if __name__ == "__main__":
    main()

