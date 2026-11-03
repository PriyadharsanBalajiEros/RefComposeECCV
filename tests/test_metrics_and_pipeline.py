"""Unit tests for Prompt Template, End-to-End Pipeline, DenseLayout Metrics, and Table Formatters."""

from __future__ import annotations

import tempfile
import unittest
from PIL import Image

from refcompose.benchmarks.memory_profiler import GPUMemoryBenchmarkSuite
from refcompose.config import load_config
from refcompose.data.denselayout_loader import DenseLayoutBenchmarkLoader
from refcompose.evaluation.evaluator import DenseLayoutBenchmarkEvaluator, TABLE2_METRIC_COLUMNS
from refcompose.pipeline.refcompose_pipeline import RefComposePipeline
from refcompose.prompts.template import PromptTemplateBuilder, ScenePromptSlots


class TestMetricsAndPipeline(unittest.TestCase):
    def test_structured_prompt_template_and_scale_steering(self) -> None:
        slots = ScenePromptSlots(
            shot_type="wide-angle interior shot",
            scene_environment="cozy modern bedroom",
            primary_objects="a wooden bed, a study desk",
            central_subject="a woven rug in the center",
            secondary_objects="potted plants and posters",
            priority_subjects=["wooden bed"],
        )
        rendered = PromptTemplateBuilder.render(slots, num_references=8)
        self.assertIn("Captured from a", rendered)
        self.assertIn("prominent, large-scale foreground wooden bed", rendered)

    def test_pipeline_inference_and_ablation_modes(self) -> None:
        cfg = load_config(
            overrides={
                "resolution": {"width": 160, "height": 96, "vae_scale_factor": 8, "patch_size": 2},
                "lora": {"rank": 4, "alpha": 4.0},
                "sampling": {"num_inference_steps": 4, "layout_inference_steps": 3},
            }
        )
        pipe = RefComposePipeline(cfg=cfg, use_mock_models=True)
        refs = [Image.new("RGB", (64, 64), color=(200, 40 * i, 90)) for i in range(6)]
        descs = [f"reference prop {i + 1}" for i in range(6)]

        for mode in ("full", "wo_canvas", "wo_depth", "merged_single_lora"):
            out = pipe(references=refs, reference_descriptions=descs, ablation_mode=mode, seed=11)
            self.assertEqual(out.generated_image.size, (160, 96))

    def test_denselayout_evaluator_and_memory_table_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = load_config(
                overrides={
                    "resolution": {"width": 160, "height": 96, "vae_scale_factor": 8, "patch_size": 2},
                    "lora": {"rank": 4, "alpha": 4.0},
                    "sampling": {"num_inference_steps": 3, "layout_inference_steps": 2},
                    "evaluation": {
                        "canonical_cache_dir": f"{tmp_dir}/cache",
                        "output_dir": f"{tmp_dir}/eval_out",
                        "num_eval_samples": 2,
                    },
                    "memory_benchmark": {
                        "output_dir": f"{tmp_dir}/mem_out",
                        "width": 160,
                        "height": 96,
                        "warmup_runs": 0,
                        "timed_runs": 1,
                    },
                }
            )
            loader = DenseLayoutBenchmarkLoader(cfg=cfg, use_mock_data=True)
            evaluator = DenseLayoutBenchmarkEvaluator(cfg=cfg, loader=loader, use_mock_metrics=True)

            from refcompose.benchmarks.baselines import build_benchmark_runner

            results = {}
            for m in ("uno", "refcompose_n5", "refcompose_n_gt_5"):
                runner = build_benchmark_runner(m, cfg, use_mock=True)
                results[m] = evaluator.evaluate_method(m, runner, max_samples=2)
                for col in TABLE2_METRIC_COLUMNS:
                    self.assertIn(col, results[m])

            exported = evaluator.export_all_reports(results, f"{tmp_dir}/eval_out")
            self.assertTrue(exported["markdown"].exists())
            self.assertTrue(exported["latex"].exists())

            mem_suite = GPUMemoryBenchmarkSuite(cfg=cfg, use_mock=True)
            mem_res = mem_suite.run_suite(methods=["uno", "refcompose"], reference_counts=[2, 6, 12])
            # Verify RefCompose token count is strictly constant across N=2, 6, 12!
            rc_t2 = mem_res["refcompose"][2].token_sequence_length
            rc_t6 = mem_res["refcompose"][6].token_sequence_length
            rc_t12 = mem_res["refcompose"][12].token_sequence_length
            self.assertEqual(rc_t2, rc_t6)
            self.assertEqual(rc_t6, rc_t12)
            # Verify UNO token count scales linearly with N!
            self.assertLess(
                mem_res["uno"][2].token_sequence_length,  # type: ignore[arg-type]
                mem_res["uno"][6].token_sequence_length,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()

