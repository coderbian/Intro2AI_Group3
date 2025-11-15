"""
Simulated Annealing (SA) Algorithm.

Simulated Annealing is a probabilistic technique for approximating the
global optimum of a function. It uses a temperature parameter that decreases
over time to control the acceptance of worse solutions.

Mathematical Formulation:
    Acceptance Probability: P(accept) = exp(-ΔE / T)
    where:
        ΔE = new_fitness - current_fitness (change in energy)
        T = current temperature

References:
    Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
    Optimization by simulated annealing. science, 220(4598), 671-680.
"""

import numpy as np
from typing import Optional, Callable, Any
from numpy.typing import NDArray

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class SimulatedAnnealing(BaseOptimizer):
    """Simulated Annealing optimization algorithm.
    
    SA uses a temperature schedule to control acceptance of worse
    solutions, allowing escape from local optima.
    
    Attributes:
        initial_temp (float): Starting temperature.
        cooling_rate (float): Temperature reduction rate.
        min_temp (float): Minimum temperature (stopping criterion).
        current_solution (NDArray): Current solution.
        current_fitness (float): Fitness of current solution.
        temperature (float): Current temperature.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Rastrigin
        >>> problem = Rastrigin(dim=10)
        >>> optimizer = SimulatedAnnealing(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     max_iter=1000,
        ...     initial_temp=100.0,
        ...     cooling_rate=0.95,
        ...     min_temp=0.01
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 1,  # SA uses single solution
        max_iter: int = 1000,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.01,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize Simulated Annealing optimizer.
        
        Args:
            objective_func: Function to minimize.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Not used, kept for interface compatibility.
            max_iter: Maximum number of iterations.
            initial_temp: Starting temperature (higher = more exploration).
            cooling_rate: Temperature reduction factor (0 to 1).
            min_temp: Minimum temperature (termination condition).
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
        
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
        # Initialize SA-specific attributes
        self.current_solution: Optional[NDArray[np.float64]] = None
        self.current_fitness: float = float('inf')
        self.temperature: float = initial_temp
        
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
        
        # Reset temperature
        self.temperature = self.initial_temp
        
    def _generate_neighbor(self) -> NDArray[np.float64]:
        """Generate a neighbor solution.
        
        Returns:
            NDArray: Neighbor solution.
        """
        # Generate neighbor with adaptive step size based on temperature
        search_range = (self.bounds[1] - self.bounds[0]) * 0.1
        perturbation = self.rng.normal(0, 1, self.dim) * search_range * (self.temperature / self.initial_temp)
        
        # Create neighbor
        neighbor = self.current_solution + perturbation
        
        # Clip to bounds
        neighbor = self.clip_bounds(neighbor)
        
        return neighbor
    
    def _acceptance_probability(self, delta_fitness: float) -> float:
        """Calculate acceptance probability for worse solution.
        
        Args:
            delta_fitness: Change in fitness (new - current).
            
        Returns:
            float: Acceptance probability.
        """
        if delta_fitness < 0:
            # Better solution, always accept
            return 1.0
        else:
            # Worse solution, accept with probability
            return np.exp(-delta_fitness / self.temperature)
    
    def _cool_down(self) -> None:
        """Reduce temperature according to cooling schedule."""
        self.temperature *= self.cooling_rate
        
        # Ensure temperature doesn't go below minimum
        self.temperature = max(self.temperature, self.min_temp)
    
    def update_population(self) -> None:
        """Perform one iteration of simulated annealing."""
        # Generate neighbor
        neighbor = self._generate_neighbor()
        neighbor_fitness = self.evaluate(neighbor)
        
        # Calculate fitness change
        delta_fitness = neighbor_fitness - self.current_fitness
        
        # Decide whether to accept neighbor
        accept_prob = self._acceptance_probability(delta_fitness)
        
        if self.rng.random() < accept_prob:
            # Accept neighbor
            self.current_solution = neighbor
            self.current_fitness = neighbor_fitness
            
            # Update best solution if improved
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.current_solution.copy()
                self.best_fitness = self.current_fitness
        
        # Cool down
        self._cool_down()
        
        # Update population (for visualization)
        self.population = self.current_solution.reshape(1, -1)
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "SimulatedAnnealing"
        """
        return "SimulatedAnnealing"
