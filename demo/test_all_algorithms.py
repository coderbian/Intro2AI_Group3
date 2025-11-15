"""
Comprehensive test of all implemented algorithms.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_intelligence.algorithms.swarm import PSO, ABC, FA, CS
from swarm_intelligence.algorithms.evolutionary import GA
from swarm_intelligence.algorithms.local_search import HillClimbing, SimulatedAnnealing
from swarm_intelligence.problems.continuous import Sphere, Rastrigin, Rosenbrock, Ackley

def test_algorithm(algorithm_class, algorithm_params, problem, max_iter=200):
    """Test a single algorithm on a problem."""
    print(f"\n{'='*70}")
    print(f"Testing {algorithm_class.__name__} on {problem.name}")
    print(f"{'='*70}")
    
    try:
        # Create optimizer
        optimizer = algorithm_class(
            objective_func=problem.evaluate,
            dim=problem.dim,
            bounds=problem.get_bounds(),
            max_iter=max_iter,
            seed=42,
            **algorithm_params
        )
        
        # Run optimization
        start = time.time()
        result = optimizer.optimize(verbose=False)
        elapsed = time.time() - start
        
        # Print results
        optimal = problem.get_optimal_value()
        error = abs(result['best_fitness'] - optimal)
        
        print(f"Best fitness: {result['best_fitness']:.6f}")
        print(f"Optimal value: {optimal:.6f}")
        print(f"Error: {error:.6f}")
        print(f"Time: {elapsed:.3f}s")
        
        # Assess quality
        if error < 0.01:
            quality = "✓ EXCELLENT"
        elif error < 0.1:
            quality = "✓ GOOD"
        elif error < 1.0:
            quality = "~ ACCEPTABLE"
        else:
            quality = "⚠ NEEDS TUNING"
        
        print(f"Quality: {quality}")
        
        return {
            'algorithm': algorithm_class.__name__,
            'problem': problem.name,
            'best_fitness': result['best_fitness'],
            'error': error,
            'time': elapsed,
            'quality': quality
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run comprehensive tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE ALGORITHM TEST SUITE")
    print("="*70)
    
    # Test problems (smaller dimensions for speed)
    problems = [
        Sphere(dim=10),
        Rastrigin(dim=10),
    ]
    
    # Algorithms to test with their parameters
    algorithms = [
        (PSO, {'pop_size': 30, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}),
        (ABC, {'pop_size': 30, 'limit': 50}),
        (FA, {'pop_size': 25, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0}),
        (CS, {'pop_size': 20, 'pa': 0.25, 'beta': 1.5}),
        (GA, {'pop_size': 40, 'crossover_rate': 0.8, 'mutation_rate': 0.1}),
        (HillClimbing, {'step_size': 0.1, 'n_neighbors': 8}),
        (SimulatedAnnealing, {'initial_temp': 100.0, 'cooling_rate': 0.95, 'min_temp': 0.01}),
    ]
    
    # Run tests
    results = []
    
    for problem in problems:
        print(f"\n{'#'*70}")
        print(f"# Problem: {problem.name} (dim={problem.dim})")
        print(f"{'#'*70}")
        
        for alg_class, params in algorithms:
            result = test_algorithm(alg_class, params, problem)
            if result:
                results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Algorithm':<20} {'Problem':<15} {'Error':<12} {'Time':<10} {'Quality'}")
    print("-"*70)
    
    for r in results:
        print(f"{r['algorithm']:<20} {r['problem']:<15} {r['error']:<12.6f} "
              f"{r['time']:<10.3f} {r['quality']}")
    
    print("\n" + "="*70)
    print(f"Total tests completed: {len(results)}")
    print("="*70)

if __name__ == "__main__":
    main()
