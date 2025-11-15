"""
Hill Climbing (HC) Algorithm.

Hill Climbing is a local search algorithm that iteratively moves to
neighboring solutions with better fitness.

Variants:
    - Steepest Ascent: Always move to best neighbor
    - First Improvement: Move to first better neighbor found
    - Random Restart: Restart from random position if stuck

References:
    Russell, S. J., & Norvig, P. (2010). Artificial intelligence:
    a modern approach. Pearson.
"""

import numpy as np
from typing import Optional, Callable, Any
from numpy.typing import NDArray

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class HillClimbing(BaseOptimizer):
    """Hill Climbing optimization algorithm.
    
    HC is a simple local search that iteratively moves to better
    neighboring solutions.
    
    Attributes:
        step_size (float): Size of neighborhood search.
        n_neighbors (int): Number of neighbors to generate per iteration.
        current_solution (NDArray): Current solution.
        current_fitness (float): Fitness of current solution.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Sphere
        >>> problem = Sphere(dim=10)
        >>> optimizer = HillClimbing(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     max_iter=1000,
        ...     step_size=0.1,
        ...     n_neighbors=8
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 1,  # HC uses single solution
        max_iter: int = 1000,
        step_size: float = 0.1,
        n_neighbors: int = 8,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize Hill Climbing optimizer.
        
        Args:
            objective_func: Function to minimize.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Not used, kept for interface compatibility.
            max_iter: Maximum number of iterations.
            step_size: Size of neighborhood search (fraction of range).
            n_neighbors: Number of neighbors to generate each iteration.
            seed: Random seed for reproducibility.
            visualizer: Optional visualizer for real-time plotting.
        """
        super().__init__(
            objective_func=objective_func,
            dim=dim,
            bounds=bounds,
            pop_size=1,
            max_iter=max_iter,
            seed=seed,
            visualizer=visualizer
        )
        
        self.step_size = step_size
        self.n_neighbors = n_neighbors
        
        # Calculate step sizes for each dimension
        self.search_range = (bounds[1] - bounds[0]) * step_size
        
        # Initialize HC-specific attributes
        self.current_solution: Optional[NDArray[np.float64]] = None
        self.current_fitness: float = float('inf')
        
    def initialize_population(self) -> None:
        """Initialize with random solution."""
        # Initialize current solution
        self.current_solution = self.rng.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=self.dim
        )
        
        # Evaluate
        self.current_fitness = self.evaluate(self.current_solution)
        
        # Set as population (for visualization)
        self.population = self.current_solution.reshape(1, -1)
        
        # Initialize best
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
        self.fitness_history.append(self.best_fitness)
        
    def _generate_neighbor(self) -> NDArray[np.float64]:
        """Generate a neighbor solution.
        
        Returns:
            NDArray: Neighbor solution.
        """
        # Generate random perturbation
        perturbation = self.rng.uniform(-1, 1, self.dim) * self.search_range
        
        # Create neighbor
        neighbor = self.current_solution + perturbation
        
        # Clip to bounds
        neighbor = self.clip_bounds(neighbor)
        
        return neighbor
    
    def update_population(self) -> None:
        """Perform one iteration of hill climbing (steepest ascent)."""
        # Generate neighbors
        best_neighbor = None
        best_neighbor_fitness = self.current_fitness
        
        for _ in range(self.n_neighbors):
            neighbor = self._generate_neighbor()
            neighbor_fitness = self.evaluate(neighbor)
            
            # Keep best neighbor
            if neighbor_fitness < best_neighbor_fitness:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness
        
        # Move to best neighbor if better
        if best_neighbor is not None and best_neighbor_fitness < self.current_fitness:
            self.current_solution = best_neighbor
            self.current_fitness = best_neighbor_fitness
            
            # Update best solution
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.current_solution.copy()
                self.best_fitness = self.current_fitness
        
        # Update population (for visualization)
        self.population = self.current_solution.reshape(1, -1)
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "HillClimbing"
        """
        return "HillClimbing"
