"""
Statistical Tests for Algorithm Comparison

Implements non-parametric statistical tests for comparing optimization algorithms.

Tests include:
- Wilcoxon signed-rank test (pairwise comparison)
- Friedman test (multiple algorithms comparison)
- Nemenyi post-hoc test (pairwise comparisons after Friedman)

These tests are recommended for comparing optimization algorithms as they don't
assume normal distribution of results.

Author: Group 3
Date: 2024
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from itertools import combinations


def wilcoxon_test(
    group1: List[float],
    group2: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float, str]:
    """
    Wilcoxon signed-rank test for paired samples.
    
    Tests whether two paired samples have different medians.
    Appropriate when comparing two algorithms on same problems.
    
    Args:
        group1: First group of values (e.g., algorithm A results)
        group2: Second group of values (e.g., algorithm B results)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Tuple of (statistic, p-value, interpretation)
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired test")
    
    if len(group1) < 2:
        raise ValueError("Need at least 2 samples")
    
    # Perform Wilcoxon test
    try:
        statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
    except ValueError as e:
        # Handle case where all differences are zero
        return 0.0, 1.0, "No significant difference (all values equal)"
    
    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        if alternative == 'two-sided':
            interpretation = "Significant difference (p < 0.05)"
        elif alternative == 'less':
            interpretation = "Group 1 significantly less than Group 2 (p < 0.05)"
        else:
            interpretation = "Group 1 significantly greater than Group 2 (p < 0.05)"
    else:
        interpretation = "No significant difference (p >= 0.05)"
    
    return float(statistic), float(p_value), interpretation


def friedman_test(
    results: Dict[str, List[float]]
) -> Tuple[float, float, str]:
    """
    Friedman test for comparing multiple algorithms.
    
    Non-parametric test for comparing more than two algorithms across multiple
    problems. Tests whether algorithm ranks differ significantly.
    
    Args:
        results: Dictionary mapping algorithm names to result lists
                Each list should contain results on same set of problems
        
    Returns:
        Tuple of (chi-square statistic, p-value, interpretation)
    """
    if len(results) < 3:
        raise ValueError("Friedman test requires at least 3 algorithms")
    
    # Convert to matrix (algorithms × problems)
    algorithm_names = list(results.keys())
    data_matrix = np.array([results[alg] for alg in algorithm_names])
    
    # Check all algorithms tested on same number of problems
    n_problems = data_matrix.shape[1]
    if n_problems < 2:
        raise ValueError("Need at least 2 problems")
    
    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*data_matrix)
    
    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        interpretation = "Significant difference among algorithms (p < 0.05)"
    else:
        interpretation = "No significant difference among algorithms (p >= 0.05)"
    
    return float(statistic), float(p_value), interpretation


def nemenyi_test(
    results: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Nemenyi post-hoc test after Friedman test.
    
    Performs pairwise comparisons between all algorithms to identify
    which specific pairs differ significantly.
    
    Args:
        results: Dictionary mapping algorithm names to result lists
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary mapping algorithm pairs to test results
    """
    algorithm_names = list(results.keys())
    n_algorithms = len(algorithm_names)
    
    if n_algorithms < 2:
        raise ValueError("Need at least 2 algorithms for comparison")
    
    # Convert to matrix and compute ranks
    data_matrix = np.array([results[alg] for alg in algorithm_names])
    n_problems = data_matrix.shape[1]
    
    # Rank each problem (1 = best)
    ranks = np.zeros_like(data_matrix, dtype=float)
    for j in range(n_problems):
        ranks[:, j] = stats.rankdata(data_matrix[:, j])
    
    # Average rank for each algorithm
    avg_ranks = np.mean(ranks, axis=1)
    
    # Critical difference for Nemenyi test
    # CD = q_alpha * sqrt(k(k+1) / (6n))
    # where q_alpha is studentized range statistic
    
    # Approximate q_alpha for common cases
    q_alpha_values = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850},
        0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589}
    }
    
    q_alpha = q_alpha_values.get(alpha, {}).get(n_algorithms, 2.728)  # Default
    cd = q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_problems))
    
    # Pairwise comparisons
    comparisons = {}
    
    for i, j in combinations(range(n_algorithms), 2):
        alg1 = algorithm_names[i]
        alg2 = algorithm_names[j]
        
        rank_diff = abs(avg_ranks[i] - avg_ranks[j])
        is_significant = rank_diff > cd
        
        comparisons[(alg1, alg2)] = {
            'rank_difference': float(rank_diff),
            'critical_difference': float(cd),
            'significant': is_significant,
            'avg_rank_1': float(avg_ranks[i]),
            'avg_rank_2': float(avg_ranks[j]),
            'better_algorithm': alg1 if avg_ranks[i] < avg_ranks[j] else alg2
        }
    
    return comparisons


def mann_whitney_u_test(
    group1: List[float],
    group2: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float, str]:
    """
    Mann-Whitney U test for independent samples.
    
    Non-parametric test for comparing two independent groups.
    Use when algorithms tested on different problem sets.
    
    Args:
        group1: First group of values
        group2: Second group of values
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Tuple of (U statistic, p-value, interpretation)
    """
    statistic, p_value = stats.mannwhitneyu(
        group1, group2, alternative=alternative
    )
    
    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        interpretation = "Significant difference (p < 0.05)"
    else:
        interpretation = "No significant difference (p >= 0.05)"
    
    return float(statistic), float(p_value), interpretation


def calculate_effect_size(
    group1: List[float],
    group2: List[float]
) -> float:
    """
    Calculate Vargha-Delaney A effect size.
    
    A measure of effect size for comparing two groups.
    Values: 0.5 (no effect), >0.5 (group1 better), <0.5 (group2 better)
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Effect size (0-1)
    """
    m = len(group1)
    n = len(group2)
    
    # Count how many times group1[i] < group2[j]
    r = 0
    for x in group1:
        for y in group2:
            if x < y:
                r += 1
            elif x == y:
                r += 0.5
    
    return r / (m * n)


def generate_statistical_report(
    results: Dict[str, List[float]],
    test_type: str = 'friedman'
) -> str:
    """
    Generate comprehensive statistical analysis report.
    
    Args:
        results: Dictionary mapping algorithm names to result lists
        test_type: 'friedman' or 'wilcoxon'
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Test Type: {test_type.upper()}")
    report.append(f"Algorithms: {', '.join(results.keys())}")
    report.append(f"Number of runs: {len(next(iter(results.values())))}")
    report.append("")
    
    if test_type == 'friedman' and len(results) >= 3:
        # Friedman test
        chi2, pval, interp = friedman_test(results)
        report.append(f"Friedman Test:")
        report.append(f"  Chi-square statistic: {chi2:.4f}")
        report.append(f"  P-value: {pval:.6f}")
        report.append(f"  {interp}")
        report.append("")
        
        # If significant, perform Nemenyi post-hoc
        if pval < 0.05:
            report.append("Nemenyi Post-hoc Test (pairwise comparisons):")
            comparisons = nemenyi_test(results)
            
            for (alg1, alg2), result in comparisons.items():
                sig_marker = "***" if result['significant'] else ""
                report.append(
                    f"  {alg1} vs {alg2}: "
                    f"ΔRank={result['rank_difference']:.3f} "
                    f"(CD={result['critical_difference']:.3f}) "
                    f"{sig_marker}"
                )
                if result['significant']:
                    report.append(f"    → {result['better_algorithm']} is significantly better")
    
    elif len(results) == 2:
        # Pairwise Wilcoxon test
        algs = list(results.keys())
        stat, pval, interp = wilcoxon_test(results[algs[0]], results[algs[1]])
        
        report.append(f"Wilcoxon Signed-Rank Test:")
        report.append(f"  Comparing: {algs[0]} vs {algs[1]}")
        report.append(f"  Statistic: {stat:.4f}")
        report.append(f"  P-value: {pval:.6f}")
        report.append(f"  {interp}")
        report.append("")
        
        # Effect size
        effect = calculate_effect_size(results[algs[0]], results[algs[1]])
        report.append(f"Effect Size (Vargha-Delaney A):")
        report.append(f"  A = {effect:.4f}")
        if effect > 0.56:
            report.append(f"  Small effect ({algs[0]} better)")
        elif effect < 0.44:
            report.append(f"  Small effect ({algs[1]} better)")
        else:
            report.append(f"  Negligible effect")
    
    report.append("=" * 70)
    
    return "\n".join(report)
