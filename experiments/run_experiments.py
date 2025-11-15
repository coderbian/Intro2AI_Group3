"""
Run experiments from configuration file.

Command-line interface for running optimization experiments with YAML config.

Usage:
    python experiments/run_experiments.py --config experiments/default_config.yaml
    python experiments/run_experiments.py --config myconfig.yaml --parallel --workers 4

Author: Group 3
Date: 2024
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
import time

from swarm_intelligence.algorithms.swarm import PSO, ABC, FA, CS, ACO
from swarm_intelligence.algorithms.evolutionary import GA
from swarm_intelligence.algorithms.local_search import HillClimbing, SimulatedAnnealing
from swarm_intelligence.problems.continuous import Sphere, Rastrigin, Rosenbrock, Ackley
from swarm_intelligence.problems.discrete import TSP, Knapsack
from swarm_intelligence.benchmarks import BenchmarkRunner, BenchmarkConfig
from swarm_intelligence.utils import load_config, setup_logger, ResultsExporter
from swarm_intelligence.visualization import plot_multiple_convergence


# Algorithm registry
ALGORITHM_REGISTRY = {
    'PSO': PSO,
    'ABC': ABC,
    'FA': FA,
    'CS': CS,
    'ACO': ACO,
    'GA': GA,
    'HillClimbing': HillClimbing,
    'SimulatedAnnealing': SimulatedAnnealing,
}

# Problem registry
PROBLEM_REGISTRY = {
    'Sphere': Sphere,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    'Ackley': Ackley,
    'TSP': TSP,
    'Knapsack': Knapsack,
}


def create_algorithm_factory(alg_config):
    """Create algorithm factory from config."""
    AlgClass = ALGORITHM_REGISTRY.get(alg_config.name)
    if AlgClass is None:
        raise ValueError(f"Unknown algorithm: {alg_config.name}")
    
    def factory(**kwargs):
        # Merge config params with runtime params
        params = {**alg_config.params, **kwargs}
        # Override with algorithm-specific parameters
        if 'pop_size' in kwargs:
            params['pop_size'] = kwargs['pop_size']
        if 'max_iter' in kwargs:
            params['max_iter'] = kwargs['max_iter']
        
        return AlgClass(**params)
    
    return factory


def create_problem_instance(prob_config):
    """Create problem instance from config."""
    ProbClass = PROBLEM_REGISTRY.get(prob_config.name)
    if ProbClass is None:
        raise ValueError(f"Unknown problem: {prob_config.name}")
    
    params = {'dim': prob_config.dim, **prob_config.params}
    return ProbClass(**params)


def run_experiment(config_path: str, parallel: bool = False, max_workers: int = None):
    """
    Run experiment from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        parallel: Enable parallel execution
        max_workers: Number of parallel workers
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    exp_config = config.experiment
    
    # Setup logger
    logger = setup_logger(
        name=exp_config.name,
        log_file=f"logs/{exp_config.name}.log"
    )
    
    logger.info(f"Starting experiment: {exp_config.name}")
    logger.info(f"Configuration file: {config_path}")
    
    # Create benchmark runner
    bench_config = BenchmarkConfig(
        num_runs=exp_config.num_runs,
        max_iter=config.algorithms[0].max_iter,  # Use first algorithm's max_iter
        pop_size=config.algorithms[0].pop_size,
        parallel=parallel,
        max_workers=max_workers,
        save_convergence=True,
        verbose=True
    )
    
    runner = BenchmarkRunner(bench_config)
    
    # Add algorithms
    logger.info("Configuring algorithms...")
    for alg_config in config.algorithms:
        logger.info(f"  - {alg_config.name}")
        factory = create_algorithm_factory(alg_config)
        runner.add_algorithm(alg_config.name, factory)
    
    # Add problems
    logger.info("Configuring problems...")
    for prob_config in config.problems:
        logger.info(f"  - {prob_config.name}")
        problem = create_problem_instance(prob_config)
        runner.add_problem(prob_config.name, problem)
    
    # Run benchmark
    logger.info("Running optimization experiments...")
    start_time = time.time()
    
    results = runner.run()
    
    elapsed = time.time() - start_time
    logger.info(f"Experiments completed in {elapsed:.2f}s")
    
    # Get aggregated results
    aggregated = runner.get_aggregated_results()
    
    # Export results
    if exp_config.save_results:
        logger.info("Exporting results...")
        
        exporter = ResultsExporter(output_dir=f"{exp_config.output_dir}/exports")
        
        # Export benchmark results
        exporter.export_benchmark_results(aggregated, exp_config.name)
        
        # Export convergence data (first run only for each algorithm-problem)
        for alg_name in runner.algorithms:
            for prob_name in runner.problems:
                histories = {}
                # Get first run's history
                for result in runner.results:
                    if (result.algorithm == alg_name and 
                        result.problem == prob_name and 
                        result.run_id == 0 and
                        result.success):
                        histories[alg_name] = result.convergence_history
                        break
                
                if histories:
                    # Plot convergence
                    plot_path = f"{exp_config.output_dir}/figures/{alg_name}_{prob_name}_convergence.png"
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                    
                    from swarm_intelligence.visualization import plot_convergence
                    plot_convergence(
                        history=histories[alg_name],
                        title=f"{alg_name} on {prob_name}",
                        save_path=plot_path
                    )
        
        # Create summary report
        exporter.create_summary_report(aggregated, exp_config.name)
        
        # Save runner results
        runner.save_results(f"{exp_config.output_dir}/{exp_config.name}_full_results.json")
    
    # Print summary
    runner.print_summary()
    
    logger.info("Experiment completed successfully!")
    
    return runner, aggregated


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run swarm intelligence optimization experiments"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='experiments/default_config.yaml',
        help='Path to configuration file (default: experiments/default_config.yaml)'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Enable parallel execution'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto)'
    )
    
    parser.add_argument(
        '--create-config',
        type=str,
        help='Create a default configuration file at specified path'
    )
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        from swarm_intelligence.utils import create_default_config, save_config
        
        config = create_default_config()
        save_config(config, args.create_config)
        print(f"Default configuration created at: {args.create_config}")
        return
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        print("Use --create-config to generate a default configuration.")
        sys.exit(1)
    
    # Run experiment
    print("=" * 70)
    print("SWARM INTELLIGENCE OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    print()
    
    try:
        run_experiment(
            config_path=args.config,
            parallel=args.parallel,
            max_workers=args.workers
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
