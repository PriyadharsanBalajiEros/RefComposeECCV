#!/usr/bin/env python3
"""
CLI: Qualitative & Quantitative Ablation Studies (Paper Sections 4.5 & 4.6, Figures 4, 5, 6, 7).

Usage:
  # Run all figures (Fig 4 grid, Fig 5 multi-ref swap, Fig 6 depth control, Fig 7 ablation):
  python scripts/run_ablations.py --config configs/eval_denselayout.yaml --lora-dir ./outputs/checkpoints/refcompose_dsa_lora/final

  # Offline verification run:
  python scripts/run_ablations.py --config configs/eval_denselayout.yaml --num-samples 2 --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refcompose.benchmarks.ablations import AblationAndQualitativeSuite
from refcompose.config import load_config
from refcompose.data.denselayout_loader import DenseLayoutBenchmarkLoader
from refcompose.prompts.template import PromptTemplateBuilder
from refcompose.utils.logging_utils import set_global_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RefCompose Qualitative & Ablation Suite (Figs 4-7)")
    parser.add_argument("--config", type=str, default="configs/eval_denselayout.yaml", help="Config YAML")
    parser.add_argument("--lora-dir", type=str, default=None, help="Trained DSA LoRA directory")
    parser.add_argument("--output-dir", type=str, default="./outputs/ablations", help="Output folder")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples for quantitative ablation")
    parser.add_argument("--mock", action="store_true", help="Run with mock models for offline verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.mock:
        overrides["resolution"] = {"width": 320, "height": 192, "vae_scale_factor": 8, "patch_size": 2}
    cfg = load_config(args.config, overrides=overrides)
    set_global_seed(cfg.sampling.seed)
    logger = setup_logger("refcompose.ablations")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DenseLayoutBenchmarkLoader(cfg=cfg, use_mock_data=args.mock)
    suite = AblationAndQualitativeSuite(
        cfg=cfg,
        lora_checkpoint_dir=args.lora_dir,
        use_mock=args.mock,
    )

    # 1. Figure 7 & Section 4.6 Quantitative + Visual Ablation
    logger.info("Running Figure 7 / Section 4.6 ablation study...")
    ablation_metrics = suite.run_figure7_ablation_study(
        loader=loader,
        output_dir=out_dir / "figure7_ablation",
        num_samples=args.num_samples,
    )
    logger.info("Ablation metrics complete: %s", list(ablation_metrics.keys()))

    # 2. Figure 5 Multi-Reference Flexibility (two distinct reference sets on shared layout)
    logger.info("Running Figure 5 multi-reference flexibility experiment...")
    descs = ["traditional stool", "carved storage vessel", "brass lotus bowl", "draped person"]
    set_a = [Image.new("RGB", (256, 256), color=(180, 60 + 30 * i, 50)) for i in range(4)]
    set_b = [Image.new("RGB", (256, 256), color=(50, 90 + 30 * i, 195)) for i in range(4)]
    palace_prompt = PromptTemplateBuilder.from_object_descriptions(
        descs, scene_environment="regal palace courtyard"
    )
    fig5_path = suite.run_figure5_multi_reference_swap(
        ref_set_a=set_a,
        ref_set_b=set_b,
        reference_descriptions=descs,
        prompt=palace_prompt,
        output_dir=out_dir / "figure5_multi_ref",
    )
    logger.info("Saved Figure 5 grid -> %s", fig5_path)

    # 3. Figure 6 Depth-Controlled Orientation (static vs dynamic floating debris depth)
    logger.info("Running Figure 6 depth-controlled orientation experiment...")
    sweet_descs = ["sweet shop counter", "brass cauldron", "wicker basket", "turbaned shopkeeper"]
    sweet_refs = [Image.new("RGB", (256, 256), color=(190, 140, 50 + 35 * i)) for i in range(4)]
    static_p = PromptTemplateBuilder.from_object_descriptions(
        sweet_descs, scene_environment="traditional street sweet shop"
    )
    dynamic_p = PromptTemplateBuilder.from_object_descriptions(
        sweet_descs,
        scene_environment="chaotic dynamic street sweet shop with floating debris and airborne bricks",
        camera_perspective="dramatic low-angle wide camera perspective",
    )
    fig6_path = suite.run_figure6_depth_controlled_orientation(
        references=sweet_refs,
        reference_descriptions=sweet_descs,
        static_prompt=static_p,
        dynamic_prompt=dynamic_p,
        output_dir=out_dir / "figure6_depth_control",
    )
    logger.info("Saved Figure 6 grid -> %s", fig6_path)

    # 4. Figure 4 Qualitative 6-Column Baseline Comparison Grid
    logger.info("Running Figure 4 multi-scene qualitative comparison grid...")
    fig4_path = suite.run_figure4_qualitative_grid(
        loader=loader,
        output_dir=out_dir / "figure4_comparison",
        num_scenes=min(7, args.num_samples),
    )
    logger.info("Saved Figure 4 grid -> %s", fig4_path)


if __name__ == "__main__":
    main()

