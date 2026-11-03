"""Unit tests for Reference Canvas construction, depth-sorted overwrite, and Stage-3 augmentations."""

from __future__ import annotations

import random
import unittest
import numpy as np
from PIL import Image

from refcompose.canvas.composer import (
    BoundingBox,
    ReferenceCanvasComposer,
    compute_mean_box_depths,
    sort_far_to_near,
)


class TestCanvasAndDepth(unittest.TestCase):
    def test_constant_canvas_shape_independent_of_reference_count(self) -> None:
        composer = ReferenceCanvasComposer(width=320, height=192)
        depth = np.linspace(1.0, 0.2, 192, dtype=np.float32)[:, None].repeat(320, axis=1)

        for n in (1, 5, 10, 15):
            refs = [Image.new("RGB", (64, 64), color=(i * 15 % 255, 80, 160)) for i in range(n)]
            boxes = [
                BoundingBox(x=float((i * 18) % 240), y=float((i * 12) % 120), w=60.0, h=50.0)
                for i in range(n)
            ]
            out = composer.compose(references=refs, boxes=boxes, depth_map=depth)
            self.assertEqual(out.canvas_tensor.shape, (3, 192, 320))
            assert out.depth_tensor is not None
            self.assertEqual(out.depth_tensor.shape, (3, 192, 320))

    def test_depth_sorted_far_to_near_overwrites_with_nearer_reference(self) -> None:
        """
        Verify Paper Eq. (6): when two boxes overlap, the nearer reference (smaller relative
        depth in DA3 convention `higher_is_farther=True`) overwrites the farther reference.
        """
        composer = ReferenceCanvasComposer(width=200, height=200)
        depth = np.full((200, 200), 0.5, dtype=np.float32)
        # Left half is farther (depth=0.9), right half is nearer (depth=0.1)
        depth[:, :100] = 0.9
        depth[:, 100:] = 0.1

        red_far = Image.new("RGB", (80, 80), color=(255, 0, 0))
        blue_near = Image.new("RGB", (80, 80), color=(0, 0, 255))

        # Overlapping boxes in [60, 140]: box 0 centered mostly in far half, box 1 mostly in near half
        box_far = BoundingBox(x=20, y=60, w=100, h=80, label="far_bed")    # spans x=[20..120], mostly far
        box_near = BoundingBox(x=80, y=60, w=100, h=80, label="near_toy")  # spans x=[80..180], mostly near

        mean_d = compute_mean_box_depths(depth, [box_far, box_near])
        self.assertGreater(mean_d[0], mean_d[1])
        pi = sort_far_to_near(mean_d, higher_is_farther=True)
        self.assertEqual(pi, [0, 1])

        out = composer.compose(
            references=[red_far, blue_near],
            boxes=[box_far, box_near],
            depth_map=depth,
            higher_depth_is_farther=True,
        )
        # Overlap pixel at (x=100, y=100) MUST belong to `blue_near` (0, 0, 255)
        overlap_rgb = out.canvas_pil.getpixel((100, 100))
        self.assertEqual(overlap_rgb, (0, 0, 255))
        # Non-overlapped far pixel at (x=40, y=100) MUST remain `red_far` (255, 0, 0)
        far_rgb = out.canvas_pil.getpixel((40, 100))
        self.assertEqual(far_rgb, (255, 0, 0))

    def test_stage1_black_canvas_and_stage3_perturbations(self) -> None:
        composer = ReferenceCanvasComposer(width=160, height=128)
        ref = Image.new("RGB", (64, 64), color=(200, 150, 100))
        box = BoundingBox(x=30, y=20, w=60, h=60)
        depth = np.ones((128, 160), dtype=np.float32) * 0.5

        black_out = composer.compose([ref], [box], depth_map=depth, black_canvas_override=True)
        self.assertTrue(np.all(np.asarray(black_out.canvas_pil) == 0))

        pert_out = composer.compose(
            [ref],
            [box],
            depth_map=depth,
            apply_stage3_perturbations=True,
            rng=random.Random(7),
        )
        self.assertEqual(pert_out.canvas_pil.size, (160, 128))


if __name__ == "__main__":
    unittest.main()

