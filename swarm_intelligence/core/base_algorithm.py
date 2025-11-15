"""
Base Algorithm Module.

This module provides the abstract base class for all optimization algorithms
using the Template Method design pattern.
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, List, Any
import numpy as np
from numpy.typing import NDArray


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms.
    
    This class implements the Template Method pattern, defining the skeleton
    of the optimization process while allowing subclasses to implement
    specific algorithm behaviors.
    
    Attributes:
        objective_func (Callable): Function to optimize (minimize by default).
        dim (int): Dimensionality of the search space.
        bounds (NDArray): Array of shape (2, dim) with [lower, upper] bounds.
        pop_size (int): Population size.
        max_iter (int): Maximum number of iterations.
        seed (Optional[int]): Random seed for reproducibility.
        visualizer (Optional[Any]): Visualization object for real-time plotting.
        minimize (bool): If True, minimize; if False, maximize.
        best_solution (NDArray): Current best solution found.
        best_fitness (float): Fitness value of the best solution.
        fitness_history (List[float]): Best fitness at each iteration.
        population (NDArray): Current population of solutions.
        iteration (int): Current iteration counter.
        rng (np.random.Generator): Random number generator for reproducibility.
        
    Example:
        >>> class MyOptimizer(BaseOptimizer):
        ...     def initialize_population(self):
        ...         self.population = np.random.uniform(
        ...             self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        ...     def update_population(self):
        ...         # Algorithm-specific update logic
        ...         pass
        ...     def get_algorithm_name(self):
        ...         return "MyOptimizer"
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 50,
        max_iter: int = 1000,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None,
        minimize: bool = True
    ):
        """Initialize the base optimizer.
        
        Args:
            objective_func: Function to optimize, takes array and returns float.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Number of solutions in the population.
            max_iter: Maximum number of iterations to run.
            seed: Random seed for reproducibility. If None, results are non-deterministic.
            visualizer: Optional visualizer object for real-time plotting.
            minimize: If True, minimize the objective; if False, maximize.
            
        Raises:
            ValueError: If bounds shape is incorrect or invalid.
        """
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        self.visualizer = visualizer
        self.minimize = minimize
        
        # Validate bounds
        self._validate_bounds()
        
        # Initialize tracking variables
        self.best_solution: Optional[NDArray[np.float64]] = None
        self.best_fitness: float = float('inf') if minimize else float('-inf')
        self.fitness_history: List[float] = []
        self.population: Optional[NDArray[np.float64]] = None
        self.iteration: int = 0
        
        # Initialize random number generator
        self.rng = np.random.default_rng(seed=seed)
        # Also set global numpy seed for compatibility
        if seed is not None:
            np.random.seed(seed)
    
    def _validate_bounds(self) -> None:
        """Validate bounds array.
        
        Raises:
            ValueError: If bounds shape is incorrect or lower >= upper.
        """
        if self.bounds.shape != (2, self.dim):
            raise ValueError(
                f"Bounds must have shape (2, {self.dim}), got {self.bounds.shape}"
            )
        if np.any(self.bounds[0] >= self.bounds[1]):
            raise ValueError("Lower bounds must be less than upper bounds")
    
    def evaluate(self, solution: NDArray[np.float64]) -> float:
        """Evaluate the objective function for a solution.
        
        Args:
            solution: Array of decision variables.
            
        Returns:
            float: Fitness value (lower is better for minimization).
        """
        fitness = self.objective_func(solution)
        # For maximization, negate the fitness
        return fitness if self.minimize else -fitness
    
    def clip_bounds(self, solution: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip solution to stay within bounds.
        
        Args:
            solution: Solution array to clip.
            
        Returns:
            NDArray: Clipped solution within bounds.
        """
        return np.clip(solution, self.bounds[0], self.bounds[1])
    
    def is_better(self, fitness1: float, fitness2: float) -> bool:
        """Compare two fitness values.
        
        Args:
            fitness1: First fitness value.
            fitness2: Second fitness value.
            
        Returns:
            bool: True if fitness1 is better than fitness2.
        """
        return fitness1 < fitness2 if self.minimize else fitness1 > fitness2
    
    @abstractmethod
    def initialize_population(self) -> None:
        """Initialize the algorithm population.
        
        This method must be implemented by subclasses to create the initial
        population according to the algorithm's requirements.
        """
        pass
    
    @abstractmethod
    def update_population(self) -> None:
        """Perform one iteration of the algorithm.
        
        This method must be implemented by subclasses to define how the
        population is updated in each iteration.
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the algorithm.
        
        Returns:
            str: Algorithm name (e.g., "PSO", "ABC", "GA").
        """
        pass
    
    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the optimization process (Template Method).
        
        This is the main optimization loop that orchestrates the algorithm
        execution. It initializes the population, runs iterations, tracks
        the best solution, and optionally visualizes progress.
        
        Args:
            verbose: If True, print progress information.
            
        Returns:
            dict: Results dictionary containing:
                - best_solution: The best solution found
                - best_fitness: The fitness of the best solution
                - fitness_history: List of best fitness at each iteration
                - iterations: Total number of iterations performed
                - time: Total execution time in seconds
                - algorithm: Name of the algorithm
                
        Example:
            >>> optimizer = MyOptimizer(func, dim=10, bounds=bounds)
            >>> result = optimizer.optimize(verbose=True)
            >>> print(f"Best fitness: {result['best_fitness']:.6f}")
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting {self.get_algorithm_name()} optimization...")
            print(f"Problem dimension: {self.dim}")
            print(f"Population size: {self.pop_size}")
            print(f"Max iterations: {self.max_iter}")
            print("-" * 60)
        
        # Initialize population
        self.initialize_population()
        
        # Setup visualizer if provided
        if self.visualizer is not None:
            self.visualizer.setup_figure()
        
        # Main optimization loop
        for self.iteration in range(self.max_iter):
            # Update population using algorithm-specific logic
            self.update_population()
            
            # Update visualizer
            if self.visualizer is not None and self.iteration % self.visualizer.update_interval == 0:
                self.visualizer.update(self.iteration, self.population, self.best_fitness)
            
            # Print progress
            if verbose and (self.iteration % max(1, self.max_iter // 10) == 0 or self.iteration == self.max_iter - 1):
                print(f"Iteration {self.iteration + 1}/{self.max_iter}: "
                      f"Best fitness = {self.best_fitness:.6f}")
        
        # Finalize visualizer
        if self.visualizer is not None:
            result_dict = {
                'best_solution': self.best_solution,
                'best_fitness': self.best_fitness,
                'fitness_history': self.fitness_history
            }
            self.visualizer.finalize(result_dict)
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print("-" * 60)
            print(f"Optimization completed in {elapsed_time:.2f} seconds")
            print(f"Best fitness: {self.best_fitness:.6f}")
        
        # Return results
        return {
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None,
            'best_fitness': self.best_fitness if self.minimize else -self.best_fitness,
            'fitness_history': self.fitness_history.copy(),
            'iterations': self.max_iter,
            'time': elapsed_time,
            'algorithm': self.get_algorithm_name()
        }
    
    def save_results(self, filepath: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Save optimization results to a JSON file.
        
        Args:
            filepath: Path to save the JSON file.
            result: Results dictionary. If None, runs optimize() first.
        """
        if result is None:
            result = self.optimize(verbose=False)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {
            'algorithm': result['algorithm'],
            'best_fitness': float(result['best_fitness']),
            'best_solution': result['best_solution'].tolist() if result['best_solution'] is not None else None,
            'fitness_history': [float(f) for f in result['fitness_history']],
            'iterations': int(result['iterations']),
            'time': float(result['time']),
            'parameters': {
                'dim': self.dim,
                'pop_size': self.pop_size,
                'max_iter': self.max_iter,
                'seed': self.seed
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
