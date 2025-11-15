"""Benchmarking and performance evaluation tools."""

from swarm_intelligence.benchmarks.runner import BenchmarkRunner, BenchmarkConfig, RunResult
from swarm_intelligence.benchmarks.metrics import (
    calculate_metrics,
    compare_algorithms,
    generate_performance_report,
    calculate_convergence_speed,
    calculate_area_under_curve
)
from swarm_intelligence.benchmarks.statistical_tests import (
    wilcoxon_test,
    friedman_test,
    nemenyi_test,
    mann_whitney_u_test,
    generate_statistical_report
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "RunResult",
    "calculate_metrics",
    "compare_algorithms",
    "generate_performance_report",
    "calculate_convergence_speed",
    "calculate_area_under_curve",
    "wilcoxon_test",
    "friedman_test",
    "nemenyi_test",
    "mann_whitney_u_test",
    "generate_statistical_report",
]

