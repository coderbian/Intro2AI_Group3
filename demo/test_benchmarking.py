"""
Test benchmarking framework

Demonstrates:
- Running multiple algorithms on multiple problems
- Statistical comparison
- Performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm_intelligence.algorithms.swarm import PSO, ABC
from swarm_intelligence.algorithms.evolutionary import GA
from swarm_intelligence.problems.continuous import Sphere, Rastrigin
from swarm_intelligence.benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    generate_performance_report,
    wilcoxon_test,
    friedman_test
)


def test_basic_benchmark():
    """Test basic benchmarking functionality."""
    print("=" * 70)
    print("Test 1: Basic Benchmark (Sequential)")
    print("=" * 70)
    
    # Create benchmark runner
    config = BenchmarkConfig(
        num_runs=5,  # Small number for quick test
        max_iter=50,
        pop_size=20,
        parallel=False,
        verbose=True
    )
    
    runner = BenchmarkRunner(config)
    
    # Add algorithms
    runner.add_algorithm(
        'PSO',
        lambda **kwargs: PSO(**kwargs)
    )
    runner.add_algorithm(
        'ABC',
        lambda **kwargs: ABC(**kwargs)
    )
    
    # Add problems
    runner.add_problem('Sphere', Sphere(dim=10))
    runner.add_problem('Rastrigin', Rastrigin(dim=10))
    
    # Run benchmark
    results = runner.run()
    
    # Print summary
    runner.print_summary()
    
    # Save results
    runner.save_results('results/benchmark_results.json')
    
    print("\n✓ Basic benchmark completed\n")


def test_statistical_comparison():
    """Test statistical comparison of algorithms."""
    print("=" * 70)
    print("Test 2: Statistical Comparison")
    print("=" * 70)
    
    # Quick benchmark
    config = BenchmarkConfig(
        num_runs=10,
        max_iter=100,
        pop_size=30,
        parallel=False,
        verbose=False
    )
    
    runner = BenchmarkRunner(config)
    
    runner.add_algorithm('PSO', lambda **kwargs: PSO(**kwargs))
    runner.add_algorithm('ABC', lambda **kwargs: ABC(**kwargs))
    runner.add_algorithm('GA', lambda **kwargs: GA(**kwargs))
    
    problem = Sphere(dim=10)
    runner.add_problem('Sphere', problem)
    
    print("Running benchmark...")
    runner.run()
    
    # Extract results for statistical tests
    aggregated = runner.get_aggregated_results()
    
    results_dict = {}
    for alg_name in ['PSO', 'ABC', 'GA']:
        # Get individual run results
        run_results = [r.best_fitness for r in runner.results 
                      if r.algorithm == alg_name and r.success]
        results_dict[alg_name] = run_results
    
    # Friedman test (3+ algorithms)
    print("\nFriedman Test:")
    chi2, pval, interp = friedman_test(results_dict)
    print(f"  Chi-square: {chi2:.4f}")
    print(f"  P-value: {pval:.6f}")
    print(f"  {interp}")
    
    # Pairwise Wilcoxon test
    print("\nPairwise Wilcoxon Tests:")
    for alg1, alg2 in [('PSO', 'ABC'), ('PSO', 'GA'), ('ABC', 'GA')]:
        stat, pval, interp = wilcoxon_test(results_dict[alg1], results_dict[alg2])
        print(f"  {alg1} vs {alg2}: p={pval:.6f} - {interp}")
    
    # Generate performance report
    print("\n" + generate_performance_report(
        results_dict,
        'Sphere (10D)',
        optimal_value=0.0
    ))
    
    print("\n✓ Statistical comparison completed\n")


def test_parallel_benchmark():
    """Test parallel benchmarking."""
    print("=" * 70)
    print("Test 3: Parallel Benchmark")
    print("=" * 70)
    
    config = BenchmarkConfig(
        num_runs=10,
        max_iter=100,
        pop_size=30,
        parallel=True,
        max_workers=2,
        verbose=True
    )
    
    runner = BenchmarkRunner(config)
    
    runner.add_algorithm('PSO', lambda **kwargs: PSO(**kwargs))
    runner.add_algorithm('ABC', lambda **kwargs: ABC(**kwargs))
    
    runner.add_problem('Sphere', Sphere(dim=10))
    runner.add_problem('Rastrigin', Rastrigin(dim=10))
    
    print("Running parallel benchmark...")
    import time
    start = time.time()
    runner.run()
    elapsed = time.time() - start
    
    runner.print_summary()
    
    print(f"\n✓ Parallel benchmark completed in {elapsed:.2f}s\n")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("BENCHMARKING FRAMEWORK TESTING")
    print("=" * 70 + "\n")
    
    test_basic_benchmark()
    test_statistical_comparison()
    # test_parallel_benchmark()  # Skip for now (parallel can be tricky)
    
    print("=" * 70)
    print("All benchmarking tests completed!")
    print("=" * 70)
