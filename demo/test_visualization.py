"""
Test visualization modules

Demonstrates:
- Convergence plotting
- Multiple algorithm comparison
- Real-time plotting capabilities
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swarm_intelligence.algorithms.swarm import PSO, ABC
from swarm_intelligence.algorithms.evolutionary import GA
from swarm_intelligence.problems.continuous import Sphere, Rastrigin
from swarm_intelligence.visualization import plot_convergence, plot_multiple_convergence
import numpy as np


def test_single_convergence_plot():
    """Test single algorithm convergence plot."""
    print("=" * 70)
    print("Test 1: Single Algorithm Convergence Plot")
    print("=" * 70)
    
    # Run PSO
    problem = Sphere(dim=10)
    optimizer = PSO(
        objective_func=problem.evaluate,
        dim=problem.dim,
        bounds=problem.get_bounds(),
        pop_size=30,
        max_iter=100,
        seed=42
    )
    
    result = optimizer.optimize(verbose=False)
    
    # Plot convergence
    plot_convergence(
        history=result['fitness_history'],
        title=f"PSO Convergence on {problem.name}",
        save_path="results/figures/pso_convergence.png"
    )
    print(f"✓ Best fitness: {result['best_fitness']:.6f}")
    print(f"✓ Plot saved to: results/figures/pso_convergence.png\n")


def test_multiple_algorithm_comparison():
    """Test multiple algorithm comparison plot."""
    print("=" * 70)
    print("Test 2: Multiple Algorithm Comparison")
    print("=" * 70)
    
    problem = Rastrigin(dim=10)
    algorithms = {
        'PSO': PSO,
        'ABC': ABC,
        'GA': GA
    }
    
    histories = {}
    
    for name, AlgClass in algorithms.items():
        print(f"Running {name}...")
        
        optimizer = AlgClass(
            objective_func=problem.evaluate,
            dim=problem.dim,
            bounds=problem.get_bounds(),
            pop_size=30,
            max_iter=200,
            seed=42
        )
        
        result = optimizer.optimize(verbose=False)
        histories[name] = result['fitness_history']
        print(f"  Best fitness: {result['best_fitness']:.6f}")
    
    # Plot comparison
    plot_multiple_convergence(
        histories=histories,
        title=f"Algorithm Comparison on {problem.name}",
        log_scale=False,
        save_path="results/figures/algorithm_comparison.png"
    )
    print(f"\n✓ Comparison plot saved to: results/figures/algorithm_comparison.png\n")


def test_multiple_problems_comparison():
    """Test algorithms on multiple problems."""
    print("=" * 70)
    print("Test 3: Multiple Problems Comparison")
    print("=" * 70)
    
    problems = {
        'Sphere': Sphere(dim=10),
        'Rastrigin': Rastrigin(dim=10)
    }
    
    algorithms = ['PSO', 'ABC']
    
    for prob_name, problem in problems.items():
        print(f"\n{prob_name}:")
        histories = {}
        
        for alg_name in algorithms:
            if alg_name == 'PSO':
                optimizer = PSO(
                    objective_func=problem.evaluate,
                    dim=problem.dim,
                    bounds=problem.get_bounds(),
                    pop_size=30,
                    max_iter=150,
                    seed=42
                )
            else:  # ABC
                optimizer = ABC(
                    objective_func=problem.evaluate,
                    dim=problem.dim,
                    bounds=problem.get_bounds(),
                    pop_size=30,
                    max_iter=150,
                    seed=42
                )
            
            result = optimizer.optimize(verbose=False)
            histories[alg_name] = result['fitness_history']
            print(f"  {alg_name}: {result['best_fitness']:.6f}")
        
        # Save individual plots
        safe_name = prob_name.lower().replace(' ', '_')
        plot_multiple_convergence(
            histories=histories,
            title=f"Convergence on {prob_name}",
            save_path=f"results/figures/{safe_name}_comparison.png"
        )
    
    print(f"\n✓ All comparison plots saved to results/figures/\n")


def test_log_scale_plot():
    """Test convergence plot with log scale."""
    print("=" * 70)
    print("Test 4: Log Scale Convergence Plot")
    print("=" * 70)
    
    problem = Sphere(dim=30)  # Higher dimension for wider fitness range
    
    optimizer = ABC(
        objective_func=problem.evaluate,
        dim=problem.dim,
        bounds=problem.get_bounds(),
        pop_size=50,
        max_iter=200,
        seed=42
    )
    
    result = optimizer.optimize(verbose=False)
    
    # Plot with log scale
    plot_convergence(
        history=result['fitness_history'],
        title=f"ABC on {problem.name} (30D) - Log Scale",
        log_scale=True,
        save_path="results/figures/abc_logscale.png"
    )
    
    print(f"✓ Best fitness: {result['best_fitness']:.6f}")
    print(f"✓ Log scale plot saved to: results/figures/abc_logscale.png\n")


if __name__ == "__main__":
    # Create results directory
    os.makedirs("results/figures", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION MODULE TESTING")
    print("=" * 70 + "\n")
    
    test_single_convergence_plot()
    test_multiple_algorithm_comparison()
    test_multiple_problems_comparison()
    test_log_scale_plot()
    
    print("=" * 70)
    print("All visualization tests completed!")
    print("Check results/figures/ for generated plots")
    print("=" * 70)
