"""
Peak GPU Memory (GB) & Latency Scaling Benchmark (Paper Section 4.4, Table 1):
    Measures peak VRAM at inference across varying reference count N (N=2, 4, 6, ..., 15)
    at batch size 1 and 1280 x 720 resolution, catching CUDA Out-Of-Memory (`OOM`) cleanly
    and exporting Table 1 in JSON, CSV, Markdown, and LaTeX formats.
"""

from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image
import torch

from refcompose.benchmarks.baselines import BaseBenchmarkMethod, build_benchmark_runner
from refcompose.canvas.composer import BoundingBox
from refcompose.config import RefComposeConfig
from refcompose.data.denselayout_loader import BenchmarkInstanceSample
from refcompose.evaluation.evaluator import METHOD_DISPLAY_NAMES
from refcompose.prompts.template import PromptTemplateBuilder


@dataclass
class MemoryProfileMeasurement:
    method: str
    num_references: int
    status: str                  # "OK" or "OOM"
    peak_gpu_memory_gb: Optional[float]
    mean_latency_sec: Optional[float]
    token_sequence_length: Optional[int]


class GPUMemoryBenchmarkSuite:
    """Profiles peak GPU VRAM and inference latency across reference cardinalities N."""

    def __init__(
        self,
        cfg: RefComposeConfig,
        use_mock: bool = False,
    ) -> None:
        self.cfg = cfg
        self.use_mock = use_mock

    def _make_synthetic_sample_for_n(self, num_refs: int) -> BenchmarkInstanceSample:
        w = self.cfg.memory_benchmark.width
        h = self.cfg.memory_benchmark.height
        gt_img = Image.new("RGB", (w, h), color=(32, 32, 32))

        boxes: List[BoundingBox] = []
        crops: List[Image.Image] = []
        descs: List[str] = []

        step_x = max(16, w // max(1, num_refs))
        for k in range(num_refs):
            bx = float(min(w - 64, k * step_x))
            by = float(40 + (k % 3) * 120)
            boxes.append(BoundingBox(x=bx, y=by, w=120.0, h=140.0, label=f"reference_subject_{k + 1}", ref_index=k))
            crops.append(Image.new("RGB", (512, 512), color=((40 * k) % 255, 120, 180)))
            descs.append(f"reference subject {k + 1}")

        prompt = PromptTemplateBuilder.from_object_descriptions(descs)
        return BenchmarkInstanceSample(
            sample_id=f"mem_bench_n{num_refs}",
            ground_truth_image=gt_img,
            prompt=prompt,
            gt_boxes=boxes,
            reference_crops=crops,
            reference_descriptions=descs,
            total_annotated_objects=num_refs,
        )

    def _compute_theoretical_conditioning_tokens(self, method_key: str, num_refs: int) -> int:
        """
        Compute conditioning image token count at 1280x720 (packed 45x80 = 3,600 tokens/canvas):
          - RefCompose: 1 canvas + 1 depth (cached at step 0) = constant 7,200 condition tokens regardless of N!
          - UNO / XVerse: N * 3,600 tokens (scales linearly with N!).
          - CreatiLayout / InstanceAssemble: box grounding tokens only (~N).
        """
        base_tokens = self.cfg.resolution.num_image_tokens  # 3,600 at 1280x720
        key = method_key.lower()
        if "refcompose" in key:
            return 2 * base_tokens  # Strictly constant for all N!
        if key in ("uno", "xverse"):
            return num_refs * base_tokens
        return num_refs

    @staticmethod
    def _reset_cuda_stats() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def profile_single_point(
        self,
        runner: BaseBenchmarkMethod,
        method_key: str,
        num_refs: int,
    ) -> MemoryProfileMeasurement:
        sample = self._make_synthetic_sample_for_n(num_refs)
        cond_tokens = self._compute_theoretical_conditioning_tokens(method_key, num_refs)

        try:
            # Warmup run(s)
            for _ in range(self.cfg.memory_benchmark.warmup_runs):
                _ = runner.generate(sample)

            self._reset_cuda_stats()
            latencies: List[float] = []
            for _ in range(max(1, self.cfg.memory_benchmark.timed_runs)):
                t0 = time.perf_counter()
                _ = runner.generate(sample)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_gb = round(peak_bytes / (1024.0 ** 3), 2)
            else:
                # CPU mode: report parameter + activation tensor footprint estimate in GB
                peak_gb = 0.0
                if hasattr(runner, "pipeline"):
                    param_bytes = sum(
                        p.numel() * p.element_size()
                        for p in runner.pipeline.backbone.parameters()
                    )
                    peak_gb = round(param_bytes / (1024.0 ** 3), 4)

            return MemoryProfileMeasurement(
                method=method_key,
                num_references=num_refs,
                status="OK",
                peak_gpu_memory_gb=peak_gb,
                mean_latency_sec=round(sum(latencies) / len(latencies), 4),
                token_sequence_length=cond_tokens,
            )

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._reset_cuda_stats()
                return MemoryProfileMeasurement(
                    method=method_key,
                    num_references=num_refs,
                    status="OOM",
                    peak_gpu_memory_gb=None,
                    mean_latency_sec=None,
                    token_sequence_length=cond_tokens,
                )
            raise

    def run_suite(
        self,
        methods: Optional[Sequence[str]] = None,
        reference_counts: Optional[Sequence[int]] = None,
    ) -> Dict[str, Dict[int, MemoryProfileMeasurement]]:
        method_list = list(methods or self.cfg.memory_benchmark.methods)
        ref_counts = list(reference_counts or self.cfg.memory_benchmark.reference_counts)

        all_results: Dict[str, Dict[int, MemoryProfileMeasurement]] = {}
        for m_key in method_list:
            all_results[m_key] = {}
            runner = build_benchmark_runner(m_key, self.cfg, use_mock=self.use_mock)
            for n_val in ref_counts:
                measurement = self.profile_single_point(runner, m_key, n_val)
                all_results[m_key][n_val] = measurement

            del runner
            self._reset_cuda_stats()

        return all_results

    @staticmethod
    def format_table1_markdown(
        results: Dict[str, Dict[int, MemoryProfileMeasurement]],
        ref_columns: Sequence[int] = (2, 4, 6),
    ) -> str:
        """Format Paper Table 1 (Peak GPU Memory in GB across N=2, 4, 6) as Markdown."""
        cols_hdr = " | ".join(f"N={n}" for n in ref_columns)
        sep_hdr = " | ".join("---:" for _ in ref_columns)
        lines = [f"| Method | {cols_hdr} |", f"| :--- | {sep_hdr} |"]

        for m_key, row in results.items():
            disp = METHOD_DISPLAY_NAMES.get(m_key, "RefCompose" if m_key == "refcompose" else m_key)
            if m_key == "refcompose":
                disp = "**RefCompose**"
            cells = []
            for n in ref_columns:
                meas = row.get(n)
                if meas is None:
                    cells.append("-")
                elif meas.status == "OOM" or meas.peak_gpu_memory_gb is None:
                    cells.append("OOM")
                else:
                    val_str = f"{meas.peak_gpu_memory_gb:.1f}"
                    cells.append(f"**{val_str}**" if m_key == "refcompose" else val_str)
            lines.append(f"| {disp} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def format_table1_latex(
        results: Dict[str, Dict[int, MemoryProfileMeasurement]],
        ref_columns: Sequence[int] = (2, 4, 6),
    ) -> str:
        """Format Paper Table 1 as publication-ready LaTeX `booktabs` source."""
        col_spec = "l" + "c" * len(ref_columns)
        hdr_cols = " & ".join(f"$N={n}$" for n in ref_columns)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Peak GPU memory (GB) at inference for varying reference count $N$ (batch size 1; $1280 \times 720$).}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            rf"\textbf{{Method}} & {hdr_cols} \\",
            r"\midrule",
        ]
        for m_key, row in results.items():
            disp = METHOD_DISPLAY_NAMES.get(m_key, "RefCompose" if m_key == "refcompose" else m_key)
            cells = []
            for n in ref_columns:
                meas = row.get(n)
                if meas is None:
                    cells.append("-")
                elif meas.status == "OOM" or meas.peak_gpu_memory_gb is None:
                    cells.append("OOM")
                else:
                    cells.append(f"{meas.peak_gpu_memory_gb:.1f}")
            vals = " & ".join(cells)
            if m_key == "refcompose":
                lines.append(r"\midrule")
                lines.append(rf"\textbf{{{disp}}} & " + " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\")
            else:
                lines.append(f"{disp} & {vals} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    @staticmethod
    def export_reports(
        results: Dict[str, Dict[int, MemoryProfileMeasurement]],
        output_dir: str | Path,
        table1_columns: Sequence[int] = (2, 4, 6),
    ) -> Dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "table1_memory.json"
        csv_path = out / "table1_memory.csv"
        md_path = out / "table1_memory.md"
        tex_path = out / "table1_memory.tex"

        serializable = {
            m: {str(n): asdict(meas) for n, meas in row.items()}
            for m, row in results.items()
        }
        json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

        all_ns = sorted({n for row in results.values() for n in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["method", "N", "status", "peak_gpu_memory_gb", "mean_latency_sec", "cond_tokens"])
            for m, row in results.items():
                for n in all_ns:
                    if n in row:
                        meas = row[n]
                        writer.writerow(
                            [
                                m,
                                n,
                                meas.status,
                                meas.peak_gpu_memory_gb,
                                meas.mean_latency_sec,
                                meas.token_sequence_length,
                            ]
                        )

        md_path.write_text(
            GPUMemoryBenchmarkSuite.format_table1_markdown(results, table1_columns) + "\n",
            encoding="utf-8",
        )
        tex_path.write_text(
            GPUMemoryBenchmarkSuite.format_table1_latex(results, table1_columns) + "\n",
            encoding="utf-8",
        )
        return {"json": json_path, "csv": csv_path, "markdown": md_path, "latex": tex_path}

