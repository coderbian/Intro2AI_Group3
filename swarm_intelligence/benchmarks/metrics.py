"""
Performance Metrics

Calculate various performance metrics for comparing optimization algorithms.

Metrics include:
- Statistical measures (mean, median, std, quartiles)
- Success rate
- Convergence speed
- Robustness measures
- Efficiency metrics

Author: Group 3
Date: 2024
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    success_rate: float
    mean_time: float


def calculate_metrics(
    fitness_values: List[float],
    times: Optional[List[float]] = None,
    optimal_value: Optional[float] = None,
    tolerance: float = 1e-8
) -> PerformanceMetrics:
    """
    Calculate performance metrics from multiple runs.
    
    Args:
        fitness_values: List of best fitness values from multiple runs
        times: List of execution times (optional)
        optimal_value: Known optimal value (optional)
        tolerance: Tolerance for success determination
        
    Returns:
        PerformanceMetrics object
    """
    arr = np.array(fitness_values)
    
    # Basic statistics
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_val = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Quartiles
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    # Success rate (if optimal value known)
    if optimal_value is not None:
        success_rate = np.sum(np.abs(arr - optimal_value) <= tolerance) / len(arr)
    else:
        success_rate = 1.0
    
    # Mean time
    mean_time = np.mean(times) if times else 0.0
    
    return PerformanceMetrics(
        mean=float(mean_val),
        median=float(median_val),
        std=float(std_val),
        min=float(min_val),
        max=float(max_val),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        success_rate=float(success_rate),
        mean_time=float(mean_time)
    )


def calculate_convergence_speed(
    history: List[float],
    target_fitness: Optional[float] = None,
    threshold: float = 0.01
) -> int:
    """
    Calculate convergence speed (number of iterations to reach target).
    
    Args:
        history: Fitness history over iterations
        target_fitness: Target fitness value (uses best if None)
        threshold: Relative threshold for convergence
        
    Returns:
        Number of iterations to convergence (-1 if not converged)
    """
    if target_fitness is None:
        target_fitness = min(history)
    
    target = target_fitness * (1 + threshold)
    
    for idx, fitness in enumerate(history):
        if fitness <= target:
            return idx + 1
    
    return -1  # Not converged


def calculate_area_under_curve(history: List[float]) -> float:
    """
    Calculate area under convergence curve (AUC).
    
    Lower AUC indicates faster convergence.
    
    Args:
        history: Fitness history over iterations
        
    Returns:
        AUC value
    """
    return float(np.trapz(history))


def calculate_reliability(
    fitness_values: List[float],
    threshold: float = 0.1
) -> float:
    """
    Calculate reliability (percentage of runs within threshold of median).
    
    Args:
        fitness_values: List of best fitness values
        threshold: Relative threshold from median
        
    Returns:
        Reliability score (0-1)
    """
    arr = np.array(fitness_values)
    median = np.median(arr)
    
    within_threshold = np.abs(arr - median) <= (threshold * np.abs(median))
    return float(np.sum(within_threshold) / len(arr))


def calculate_efficiency(
    fitness_value: float,
    time_taken: float,
    optimal_value: Optional[float] = None
) -> float:
    """
    Calculate efficiency metric (quality per unit time).
    
    Args:
        fitness_value: Achieved fitness
        time_taken: Execution time
        optimal_value: Known optimal value (optional)
        
    Returns:
        Efficiency score
    """
    if time_taken <= 0:
        return 0.0
    
    if optimal_value is not None:
        # Normalized quality
        quality = 1.0 / (1.0 + abs(fitness_value - optimal_value))
    else:
        # Inverse of fitness (assuming minimization)
        quality = 1.0 / (1.0 + abs(fitness_value))
    
    return quality / time_taken


def compare_algorithms(
    results: Dict[str, List[float]],
    optimal_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compare multiple algorithms on same problem.
    
    Args:
        results: Dictionary mapping algorithm names to fitness lists
        optimal_value: Known optimal value (optional)
        
    Returns:
        Comparison statistics dictionary
    """
    comparison = {}
    
    for alg_name, fitness_list in results.items():
        metrics = calculate_metrics(fitness_list, optimal_value=optimal_value)
        
        comparison[alg_name] = {
            'mean': metrics.mean,
            'median': metrics.median,
            'std': metrics.std,
            'best': metrics.min,
            'worst': metrics.max,
            'success_rate': metrics.success_rate
        }
    
    # Rank algorithms
    rankings = {}
    for metric in ['mean', 'median', 'best']:
        sorted_algs = sorted(
            results.keys(),
            key=lambda x: comparison[x][metric]
        )
        rankings[metric] = {alg: rank + 1 for rank, alg in enumerate(sorted_algs)}
    
    comparison['rankings'] = rankings
    
    return comparison


def calculate_statistical_significance(
    group1: List[float],
    group2: List[float],
    test: str = 'wilcoxon'
) -> Tuple[float, float]:
    """
    Calculate statistical significance between two groups.
    
    Args:
        group1: First group of fitness values
        group2: Second group of fitness values
        test: Statistical test ('wilcoxon' or 't-test')
        
    Returns:
        Tuple of (statistic, p-value)
    """
    try:
        from scipy import stats
        
        if test == 'wilcoxon':
            # Wilcoxon signed-rank test (paired)
            if len(group1) != len(group2):
                raise ValueError("Groups must have same length for paired test")
            stat, pval = stats.wilcoxon(group1, group2)
        else:
            # Independent t-test
            stat, pval = stats.ttest_ind(group1, group2)
        
        return float(stat), float(pval)
        
    except ImportError:
        raise ImportError("scipy required for statistical tests")


def generate_performance_report(
    results: Dict[str, List[float]],
    problem_name: str,
    optimal_value: Optional[float] = None
) -> str:
    """
    Generate a formatted performance report.
    
    Args:
        results: Dictionary mapping algorithm names to fitness lists
        problem_name: Name of the problem
        optimal_value: Known optimal value (optional)
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append(f"Performance Report: {problem_name}")
    report.append("=" * 70)
    
    if optimal_value is not None:
        report.append(f"Optimal Value: {optimal_value:.6f}\n")
    
    # Calculate metrics for each algorithm
    all_metrics = {}
    for alg_name, fitness_list in results.items():
        all_metrics[alg_name] = calculate_metrics(fitness_list, optimal_value=optimal_value)
    
    # Table header
    report.append(f"{'Algorithm':<15} {'Mean±Std':<20} {'Median':<12} {'Best':<12} {'Success%':<10}")
    report.append("-" * 70)
    
    # Sort by mean
    sorted_algs = sorted(all_metrics.keys(), key=lambda x: all_metrics[x].mean)
    
    for alg_name in sorted_algs:
        metrics = all_metrics[alg_name]
        mean_std = f"{metrics.mean:.6f}±{metrics.std:.6f}"
        success_pct = f"{metrics.success_rate * 100:.1f}%"
        
        report.append(
            f"{alg_name:<15} {mean_std:<20} {metrics.median:<12.6f} "
            f"{metrics.min:<12.6f} {success_pct:<10}"
        )
    
    report.append("=" * 70)
    
    return "\n".join(report)
