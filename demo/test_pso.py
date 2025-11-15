"""
Quick test script to verify PSO with Sphere function.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_intelligence.algorithms.swarm import PSO
from swarm_intelligence.problems.continuous import Sphere

def test_pso_sphere():
    """Test PSO on Sphere function."""
    print("=" * 60)
    print("Testing PSO on Sphere Function")
    print("=" * 60)
    
    # Create problem
    problem = Sphere(dim=10)
    print(f"\nProblem: {problem}")
    print(f"Optimal value: {problem.get_optimal_value()}")
    print(f"Domain: {problem.domain}")
    
    # Create optimizer
    optimizer = PSO(
        objective_func=problem.evaluate,
        dim=problem.dim,
        bounds=problem.get_bounds(),
        pop_size=30,
        max_iter=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42
    )
    
    # Run optimization
    result = optimizer.optimize(verbose=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Best fitness: {result['best_fitness']:.10f}")
    print(f"Best solution: {result['best_solution'][:5]}... (showing first 5)")
    print(f"Time: {result['time']:.2f} seconds")
    print(f"Final iteration best: {result['fitness_history'][-1]:.10f}")
    
    # Check if close to optimal
    error = abs(result['best_fitness'] - problem.get_optimal_value())
    print(f"\nError from optimal: {error:.10f}")
    
    if error < 1e-5:
        print("✓ SUCCESS: Solution is very close to optimal!")
    elif error < 0.01:
        print("✓ GOOD: Solution is reasonably close to optimal")
    else:
        print("⚠ WARNING: Solution is far from optimal")

if __name__ == "__main__":
    test_pso_sphere()
