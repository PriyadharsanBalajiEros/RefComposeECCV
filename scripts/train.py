#!/usr/bin/env python3
"""
CLI: 3-Stage RefCompose DSA LoRA Training (Paper Section 3.4 & Section 4.3).

Usage:
  # Single or multi-GPU H100 production run (50,000 iterations at 1280x720, r=32):
  accelerate launch scripts/train.py --config configs/train_dsa_lora.yaml

  # Quick standalone smoke-test across all 3 curriculum stages:
  python scripts/train.py --config configs/train_dsa_lora.yaml --synthetic-in-memory --mock --max-steps 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refcompose.config import load_config
from refcompose.data.dataset import RefComposeTrainingDataset, SyntheticInMemoryDataset
from refcompose.training.trainer import RefComposeTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RefCompose DSA LoRA (3-Stage Curriculum)")
    parser.add_argument("--config", type=str, default="configs/train_dsa_lora.yaml", help="Training YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total training iterations")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint folder to resume from")
    parser.add_argument(
        "--synthetic-in-memory",
        action="store_true",
        help="Use procedural in-memory multi-object dataset (useful for immediate testing)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use lightweight StandaloneMMDiT backbone instead of 32B FLUX.2-dev weights",
    )
    parser.add_argument("--no-accelerate", action="store_true", help="Disable HuggingFace Accelerate wrapper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.output_dir:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.resume:
        overrides.setdefault("training", {})["resume_from_checkpoint"] = args.resume
    if args.mock:
        # Use compact 256x144 (16:9) resolution in mock CPU mode for rapid execution
        overrides["resolution"] = {"width": 256, "height": 144, "vae_scale_factor": 8, "patch_size": 2}

    cfg = load_config(args.config, overrides=overrides)

    if args.max_steps is not None and args.max_steps < cfg.training.curriculum.stage1_end_step:
        # Proportionally scale stage boundaries if user requested a short smoke-test run
        s1 = max(1, args.max_steps // 5)
        s2 = max(s1 + 1, (args.max_steps * 3) // 5)
        cfg.training.curriculum.stage1_end_step = s1
        cfg.training.curriculum.stage2_end_step = s2
        cfg.training.curriculum.stage3_end_step = args.max_steps

    if args.synthetic_in_memory:
        dataset = SyntheticInMemoryDataset(cfg=cfg, length=32, initial_stage=1)
    else:
        dataset = RefComposeTrainingDataset(cfg=cfg, initial_stage=1)

    trainer = RefComposeTrainer(
        cfg=cfg,
        dataset=dataset,
        use_mock_backbone=args.mock,
        use_accelerate=not args.no_accelerate,
    )
    trainer.train(max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()

