"""Quantitative evaluation metrics and DenseLayout benchmark reporting (Paper Section 4.4)."""

from refcompose.evaluation.evaluator import (
    METHOD_DISPLAY_NAMES,
    TABLE2_METRIC_COLUMNS,
    DenseLayoutBenchmarkEvaluator,
)
from refcompose.evaluation.metrics_global import GlobalMetricsSuite
from refcompose.evaluation.metrics_region import RegionMetricsSuite

__all__ = [
    "METHOD_DISPLAY_NAMES",
    "TABLE2_METRIC_COLUMNS",
    "DenseLayoutBenchmarkEvaluator",
    "GlobalMetricsSuite",
    "RegionMetricsSuite",
]

