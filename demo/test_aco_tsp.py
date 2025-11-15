"""
Test ACO algorithm on Traveling Salesman Problem (TSP)

This script demonstrates:
- ACO implementation
- TSP problem with random cities
- Solution quality assessment
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm_intelligence.algorithms.swarm import ACO
from swarm_intelligence.problems.discrete import TSP
import time


def test_aco_on_tsp():
    """Test ACO algorithm on TSP with different sizes."""
    
    print("=" * 70)
    print("Testing ACO on Traveling Salesman Problem (TSP)")
    print("=" * 70)
    
    test_cases = [
        {'num_cities': 10, 'max_iter': 100},
        {'num_cities': 20, 'max_iter': 200},
        {'num_cities': 30, 'max_iter': 300},
    ]
    
    for idx, config in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {idx}: {config['num_cities']} cities, {config['max_iter']} iterations")
        print(f"{'='*70}")
        
        # Create TSP problem
        problem = TSP(num_cities=config['num_cities'], seed=42)
        
        print(f"\nProblem Info:")
        info = problem.get_problem_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create ACO optimizer
        optimizer = ACO(
            objective_func=problem.evaluate,
            dim=problem.dim,
            bounds=problem.get_bounds(),
            pop_size=30,
            max_iter=config['max_iter'],
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            Q=1.0,
            heuristic_func=problem.get_heuristic_matrix,
            seed=42
        )
        
        # Run optimization
        print(f"\nRunning ACO...")
        start_time = time.time()
        result = optimizer.optimize(verbose=False)
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"\nResults:")
        print(f"  Best tour length: {result['best_fitness']:.4f}")
        print(f"  Best tour: {result['best_solution'][:10]}...")  # Show first 10 cities
        print(f"  Time: {elapsed_time:.3f}s")
        print(f"  Iterations: {config['max_iter']}")
        
        # Calculate improvement
        # Compare with random tour
        import numpy as np
        random_tour = np.random.RandomState(42).permutation(config['num_cities'])
        random_fitness = problem.evaluate(random_tour)
        improvement = ((random_fitness - result['best_fitness']) / random_fitness) * 100
        
        print(f"\nComparison with random tour:")
        print(f"  Random tour length: {random_fitness:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        
        # Quality assessment
        if improvement > 30:
            quality = "EXCELLENT"
        elif improvement > 20:
            quality = "GOOD"
        elif improvement > 10:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS TUNING"
        
        print(f"  Quality: {quality}")


def test_aco_on_clustered_tsp():
    """Test ACO on TSP with clustered cities."""
    
    print("\n" + "=" * 70)
    print("Testing ACO on Clustered TSP")
    print("=" * 70)
    
    # Create clustered TSP
    problem = TSP.generate_clustered_instance(num_cities=25, num_clusters=5, seed=42)
    
    print(f"\nProblem: 25 cities in 5 clusters")
    
    # Create ACO optimizer
    optimizer = ACO(
        objective_func=problem.evaluate,
        dim=problem.dim,
        bounds=problem.get_bounds(),
        pop_size=40,
        max_iter=200,
        alpha=1.0,
        beta=3.0,  # Higher beta for exploiting heuristic
        rho=0.15,
        Q=1.0,
        heuristic_func=problem.get_heuristic_matrix,
        seed=42
    )
    
    # Run optimization
    print(f"\nRunning ACO with β=3.0 (strong heuristic bias)...")
    start_time = time.time()
    result = optimizer.optimize(verbose=False)
    elapsed_time = time.time() - start_time
    
    # Display results
    print(f"\nResults:")
    print(f"  Best tour length: {result['best_fitness']:.4f}")
    print(f"  Time: {elapsed_time:.3f}s")


if __name__ == "__main__":
    test_aco_on_tsp()
    test_aco_on_clustered_tsp()
    
    print("\n" + "=" * 70)
    print("ACO Testing Complete!")
    print("=" * 70)
