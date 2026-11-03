#!/usr/bin/env python3
"""
CLI: RefCompose Multi-Reference Image Generation Inference (Paper Section 3 & Figure 2).

Usage:
  python scripts/infer.py \
      --config configs/default.yaml \
      --lora-dir ./outputs/checkpoints/refcompose_dsa_lora/final \
      --references ref1.png ref2.png ref3.png \
      --descriptions "wooden bed" "vintage table lamp" "woven rug" \
      --prompt "A wide-angle interior shot of a cozy modern bedroom..." \
      --output-dir ./outputs/inference_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refcompose.config import load_config
from refcompose.pipeline.refcompose_pipeline import RefComposePipeline
from refcompose.utils.logging_utils import set_global_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RefCompose Multi-Reference Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--lora-dir", type=str, default=None, help="Directory with trained DSA LoRA weights")
    parser.add_argument("--references", nargs="+", default=[], help="Paths to N RGB reference images")
    parser.add_argument("--descriptions", nargs="+", required=True, help="Text description per reference image")
    parser.add_argument("--prompt", type=str, default=None, help="Optional full compositional prompt p")
    parser.add_argument("--steps", type=int, default=None, help="Override T=40 flow-matching steps")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override classifier-free guidance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="full",
        choices=["full", "wo_canvas", "wo_depth", "merged_single_lora"],
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/inference", help="Output folder")
    parser.add_argument("--mock", action="store_true", help="Run with standalone mock models (offline test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.mock:
        overrides["resolution"] = {"width": 320, "height": 192, "vae_scale_factor": 8, "patch_size": 2}
    cfg = load_config(args.config, overrides=overrides)
    set_global_seed(args.seed)
    logger = setup_logger("refcompose.infer")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.references:
        ref_imgs = [Image.open(p).convert("RGB") for p in args.references]
    elif args.mock:
        ref_imgs = [
            Image.new("RGB", (256, 256), color=((70 * (i + 1)) % 255, 110, (150 * (i + 1)) % 255))
            for i in range(len(args.descriptions))
        ]
    else:
        raise ValueError("Please provide `--references ref1.png ref2.png ...` (or `--mock` for dry-run).")

    if len(ref_imgs) != len(args.descriptions):
        raise ValueError(
            f"Got {len(ref_imgs)} reference images but {len(args.descriptions)} descriptions; counts must match."
        )

    pipe = RefComposePipeline(
        cfg=cfg,
        lora_checkpoint_dir=args.lora_dir,
        use_mock_models=args.mock,
    )
    result = pipe(
        references=ref_imgs,
        reference_descriptions=args.descriptions,
        prompt=args.prompt,
        ablation_mode=args.ablation_mode,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    result.generated_image.save(out_dir / "generated_composite.png")
    result.canvas_image.save(out_dir / "reference_canvas.png")
    if result.layout_image is not None:
        result.layout_image.save(out_dir / "stage1_layout.png")
    if result.depth_vis_image is not None:
        result.depth_vis_image.save(out_dir / "stage1_depth_da3.png")
    if result.boxes_overlay_image is not None:
        result.boxes_overlay_image.save(out_dir / "stage1_boxes_overlay.png")
    (out_dir / "prompt.txt").write_text(result.prompt + "\n", encoding="utf-8")

    logger.info("Saved generated composite and Stage 1/2 diagnostics to %s", out_dir.resolve())


if __name__ == "__main__":
    main()

