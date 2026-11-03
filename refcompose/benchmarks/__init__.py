"""Benchmark runners, GPU memory scaling profiler (Table 1), and ablation studies (Figures 4-7)."""

from refcompose.benchmarks.ablations import (
    ABLATION_DISPLAY_LABELS,
    ABLATION_MODES,
    AblationAndQualitativeSuite,
)
from refcompose.benchmarks.baselines import (
    BaseBenchmarkMethod,
    CreatiLayoutRunner,
    InstanceAssembleRunner,
    PrecomputedDirectoryRunner,
    RefComposeBenchmarkRunner,
    UNORunner,
    XVerseRunner,
    build_benchmark_runner,
)
from refcompose.benchmarks.memory_profiler import (
    GPUMemoryBenchmarkSuite,
    MemoryProfileMeasurement,
)

__all__ = [
    "ABLATION_DISPLAY_LABELS",
    "ABLATION_MODES",
    "AblationAndQualitativeSuite",
    "BaseBenchmarkMethod",
    "CreatiLayoutRunner",
    "InstanceAssembleRunner",
    "PrecomputedDirectoryRunner",
    "RefComposeBenchmarkRunner",
    "UNORunner",
    "XVerseRunner",
    "build_benchmark_runner",
    "GPUMemoryBenchmarkSuite",
    "MemoryProfileMeasurement",
]

