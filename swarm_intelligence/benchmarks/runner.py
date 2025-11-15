"""
Benchmark Runner

Executes optimization algorithms on multiple problems with multiple runs for statistical
analysis. Supports parallel execution and comprehensive result tracking.

Features:
- Multiple algorithm-problem combinations
- Multiple independent runs per combination
- Parallel execution support
- Progress tracking
- Result aggregation and export

Author: Group 3
Date: 2024
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings

from swarm_intelligence.core.base_algorithm import BaseOptimizer
from swarm_intelligence.core.base_problem import OptimizationProblem


@dataclass
class RunResult:
    """Single run result."""
    algorithm: str
    problem: str
    run_id: int
    best_fitness: float
    convergence_history: List[float]
    time: float
    success: bool
    error_message: Optional[str] = None
    

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    num_runs: int = 30
    max_iter: int = 1000
    pop_size: int = 50
    parallel: bool = True
    max_workers: Optional[int] = None
    save_convergence: bool = True
    verbose: bool = True


class BenchmarkRunner:
    """
    Run benchmarking experiments for optimization algorithms.
    
    Attributes:
        config: Benchmark configuration
        results: List of all run results
        algorithms: Dictionary of algorithm factories
        problems: Dictionary of problem instances
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration (default: BenchmarkConfig())
        """
        self.config = config if config is not None else BenchmarkConfig()
        self.results: List[RunResult] = []
        self.algorithms: Dict[str, Callable] = {}
        self.problems: Dict[str, OptimizationProblem] = {}
    
    def add_algorithm(
        self,
        name: str,
        algorithm_factory: Callable[[Callable, int, np.ndarray, int, int, int], BaseOptimizer]
    ):
        """
        Add an algorithm to benchmark.
        
        Args:
            name: Algorithm name
            algorithm_factory: Factory function that creates algorithm instance
                              Signature: (objective_func, dim, bounds, pop_size, max_iter, seed) -> BaseOptimizer
        """
        self.algorithms[name] = algorithm_factory
    
    def add_problem(self, name: str, problem: OptimizationProblem):
        """
        Add a problem to benchmark.
        
        Args:
            name: Problem name
            problem: Problem instance
        """
        self.problems[name] = problem
    
    def _run_single_experiment(
        self,
        algorithm_name: str,
        problem_name: str,
        run_id: int,
        seed: int
    ) -> RunResult:
        """
        Run a single optimization experiment.
        
        Args:
            algorithm_name: Name of algorithm
            problem_name: Name of problem
            run_id: Run identifier
            seed: Random seed
            
        Returns:
            RunResult object
        """
        try:
            problem = self.problems[problem_name]
            
            # Create algorithm instance
            algorithm = self.algorithms[algorithm_name](
                objective_func=problem.evaluate,
                dim=problem.dim,
                bounds=problem.get_bounds(),
                pop_size=self.config.pop_size,
                max_iter=self.config.max_iter,
                seed=seed
            )
            
            # Run optimization
            start_time = time.time()
            result = algorithm.optimize(verbose=False)
            elapsed_time = time.time() - start_time
            
            # Create run result
            return RunResult(
                algorithm=algorithm_name,
                problem=problem_name,
                run_id=run_id,
                best_fitness=result['best_fitness'],
                convergence_history=result['fitness_history'] if self.config.save_convergence else [],
                time=elapsed_time,
                success=True
            )
            
        except Exception as e:
            warnings.warn(f"Run failed: {algorithm_name} on {problem_name}, run {run_id}: {str(e)}")
            return RunResult(
                algorithm=algorithm_name,
                problem=problem_name,
                run_id=run_id,
                best_fitness=float('inf'),
                convergence_history=[],
                time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run(self) -> List[RunResult]:
        """
        Execute all benchmark experiments.
        
        Returns:
            List of all run results
        """
        if not self.algorithms or not self.problems:
            raise ValueError("Must add at least one algorithm and one problem")
        
        total_runs = len(self.algorithms) * len(self.problems) * self.config.num_runs
        
        if self.config.verbose:
            print("=" * 70)
            print("BENCHMARK EXECUTION")
            print("=" * 70)
            print(f"Algorithms: {len(self.algorithms)}")
            print(f"Problems: {len(self.problems)}")
            print(f"Runs per combination: {self.config.num_runs}")
            print(f"Total runs: {total_runs}")
            print(f"Parallel: {self.config.parallel}")
            print("=" * 70)
        
        # Prepare all experiments
        experiments = []
        for alg_name in self.algorithms:
            for prob_name in self.problems:
                for run_id in range(self.config.num_runs):
                    seed = run_id  # Use run_id as seed for reproducibility
                    experiments.append((alg_name, prob_name, run_id, seed))
        
        start_time = time.time()
        results = []
        
        if self.config.parallel:
            # Parallel execution
            max_workers = self.config.max_workers or min(4, len(experiments))
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_single_experiment,
                        alg_name, prob_name, run_id, seed
                    ): (alg_name, prob_name, run_id)
                    for alg_name, prob_name, run_id, seed in experiments
                }
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if self.config.verbose and completed % max(1, total_runs // 20) == 0:
                        progress = (completed / total_runs) * 100
                        print(f"Progress: {completed}/{total_runs} ({progress:.1f}%)")
        else:
            # Sequential execution
            for idx, (alg_name, prob_name, run_id, seed) in enumerate(experiments):
                result = self._run_single_experiment(alg_name, prob_name, run_id, seed)
                results.append(result)
                
                if self.config.verbose and (idx + 1) % max(1, total_runs // 20) == 0:
                    progress = ((idx + 1) / total_runs) * 100
                    print(f"Progress: {idx + 1}/{total_runs} ({progress:.1f}%)")
        
        elapsed_time = time.time() - start_time
        
        if self.config.verbose:
            print("=" * 70)
            print(f"Benchmark completed in {elapsed_time:.2f} seconds")
            print(f"Successful runs: {sum(1 for r in results if r.success)}/{total_runs}")
            print("=" * 70)
        
        self.results = results
        return results
    
    def get_aggregated_results(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Aggregate results by algorithm and problem.
        
        Returns:
            Nested dictionary: {algorithm: {problem: {metric: value}}}
        """
        aggregated = {}
        
        for alg_name in self.algorithms:
            aggregated[alg_name] = {}
            
            for prob_name in self.problems:
                # Get all runs for this combination
                runs = [r for r in self.results 
                       if r.algorithm == alg_name and r.problem == prob_name and r.success]
                
                if not runs:
                    continue
                
                fitness_values = [r.best_fitness for r in runs]
                times = [r.time for r in runs]
                
                aggregated[alg_name][prob_name] = {
                    'mean_fitness': np.mean(fitness_values),
                    'std_fitness': np.std(fitness_values),
                    'median_fitness': np.median(fitness_values),
                    'best_fitness': np.min(fitness_values),
                    'worst_fitness': np.max(fitness_values),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'success_rate': len(runs) / self.config.num_runs,
                    'num_runs': len(runs)
                }
        
        return aggregated
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to JSON file.
        
        Args:
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        data = {
            'config': asdict(self.config),
            'algorithms': list(self.algorithms.keys()),
            'problems': list(self.problems.keys()),
            'results': [asdict(r) for r in self.results],
            'aggregated': self.get_aggregated_results()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        aggregated = self.get_aggregated_results()
        
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        for prob_name in self.problems:
            print(f"\n{prob_name}:")
            print(f"{'Algorithm':<15} {'Mean±Std':<20} {'Best':<12} {'Time(s)':<10}")
            print("-" * 70)
            
            for alg_name in self.algorithms:
                if prob_name in aggregated.get(alg_name, {}):
                    stats = aggregated[alg_name][prob_name]
                    mean_std = f"{stats['mean_fitness']:.6f}±{stats['std_fitness']:.6f}"
                    print(f"{alg_name:<15} {mean_std:<20} {stats['best_fitness']:<12.6f} {stats['mean_time']:<10.3f}")
        
        print("=" * 70)
