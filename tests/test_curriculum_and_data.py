"""Unit tests for the 3-Stage Curriculum Scheduler (10k / 30k / 50k) and Trainer loop."""

from __future__ import annotations

import tempfile
import unittest

from refcompose.config import CurriculumConfig, load_config
from refcompose.data.dataset import SyntheticInMemoryDataset
from refcompose.training.curriculum import ThreeStageCurriculumScheduler
from refcompose.training.trainer import RefComposeTrainer


class TestCurriculumAndTraining(unittest.TestCase):
    def test_exact_paper_stage_boundaries(self) -> None:
        sched = ThreeStageCurriculumScheduler(
            CurriculumConfig(stage1_end_step=10000, stage2_end_step=30000, stage3_end_step=50000)
        )
        s0 = sched.get_state(0)
        self.assertEqual(s0.stage, 1)
        self.assertTrue(s0.use_black_canvas)
        self.assertTrue(s0.train_structure)
        self.assertFalse(s0.train_appearance)

        s9999 = sched.get_state(9999)
        self.assertEqual(s9999.stage, 1)

        s10000 = sched.get_state(10000)
        self.assertEqual(s10000.stage, 2)
        self.assertFalse(s10000.use_black_canvas)
        self.assertFalse(s10000.apply_perturbations)
        self.assertTrue(s10000.train_appearance)

        s30000 = sched.get_state(30000)
        self.assertEqual(s30000.stage, 3)
        self.assertTrue(s30000.apply_perturbations)

    def test_end_to_end_three_stage_trainer_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config(
                overrides={
                    "resolution": {"width": 128, "height": 64, "vae_scale_factor": 8, "patch_size": 2},
                    "lora": {"rank": 4, "alpha": 4.0},
                    "training": {
                        "output_dir": f"{tmp_dir}/ckpts",
                        "logging_dir": f"{tmp_dir}/logs",
                        "max_train_steps": 6,
                        "batch_size": 2,
                        "gradient_accumulation_steps": 1,
                        "save_every_steps": 3,
                        "log_every_steps": 2,
                        "curriculum": {
                            "stage1_end_step": 2,
                            "stage2_end_step": 4,
                            "stage3_end_step": 6,
                        },
                    },
                }
            )
            ds = SyntheticInMemoryDataset(cfg=cfg, length=8, initial_stage=1)
            trainer = RefComposeTrainer(
                cfg=cfg,
                dataset=ds,
                use_mock_backbone=True,
                use_accelerate=False,
            )
            summary = trainer.train()
            self.assertEqual(summary["final_step"], 6)
            self.assertEqual(summary["stage_history"], [1, 1, 2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()

