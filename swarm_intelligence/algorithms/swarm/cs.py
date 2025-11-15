"""
Cuckoo Search (CS) Algorithm.

The Cuckoo Search algorithm is inspired by the brood parasitism behavior
of cuckoo birds. It uses Lévy flights for efficient exploration.

Algorithm Steps:
    1. Generate new solution using Lévy flight
    2. Evaluate and compare with random nest
    3. Abandon worst nests with probability pa
    4. Keep best solutions

References:
    Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.
    World congress on nature & biologically inspired computing (pp. 210-214).
"""

import numpy as np
from typing import Optional, Callable, Any
from numpy.typing import NDArray
from scipy import special

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class CS(BaseOptimizer):
    """Cuckoo Search optimization algorithm.
    
    CS uses Lévy flights for exploration and a discovery rate
    for nest abandonment, balancing exploration and exploitation.
    
    Attributes:
        pa (float): Probability of abandoning worst nests.
        beta (float): Lévy flight parameter (typically 1.5).
        nests (NDArray): Positions of nests (solutions).
        fitness_values (NDArray): Fitness of nests.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Rosenbrock
        >>> problem = Rosenbrock(dim=10)
        >>> optimizer = CS(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     pop_size=25,
        ...     max_iter=1000,
        ...     pa=0.25,
        ...     beta=1.5
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 25,
        max_iter: int = 1000,
        pa: float = 0.25,
        beta: float = 1.5,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize CS optimizer.
        
        Args:
            objective_func: Function to minimize.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Number of nests (solutions).
            max_iter: Maximum number of iterations.
            pa: Probability of abandoning worst nests (0 to 1).
            beta: Lévy flight parameter (typically 1.5).
            seed: Random seed for reproducibility.
            visualizer: Optional visualizer for real-time plotting.
        """
        super().__init__(
            objective_func=objective_func,
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter,
            seed=seed,
            visualizer=visualizer
        )
        
        self.pa = pa
        self.beta = beta
        
        # Calculate Lévy flight parameters
        self.sigma = self._calculate_sigma()
        
        # Initialize CS-specific attributes
        self.nests: Optional[NDArray[np.float64]] = None
        self.fitness_values: Optional[NDArray[np.float64]] = None
        
    def _calculate_sigma(self) -> float:
        """Calculate sigma parameter for Lévy flight.
        
        Returns:
            float: Sigma value.
        """
        numerator = special.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        denominator = special.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        return (numerator / denominator) ** (1 / self.beta)
    
    def _levy_flight(self, size: int) -> NDArray[np.float64]:
        """Generate Lévy flight step.
        
        Args:
            size: Dimension of the step.
            
        Returns:
            NDArray: Lévy flight step.
        """
        u = self.rng.normal(0, self.sigma, size)
        v = self.rng.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / self.beta))
        return step
    
    def initialize_population(self) -> None:
        """Initialize nest positions randomly within bounds."""
        # Initialize nests
        self.nests = self.rng.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.dim)
        )
        self.population = self.nests.copy()
        
        # Evaluate initial nests
        self.fitness_values = np.array([self.evaluate(nest) for nest in self.nests])
        
        # Find initial best
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.nests[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
        
    def _get_cuckoo(self, index: int) -> NDArray[np.float64]:
        """Generate new solution via Lévy flight.
        
        Args:
            index: Index of the current nest.
            
        Returns:
            NDArray: New solution.
        """
        # Lévy flight
        step_size = 0.01
        step = self._levy_flight(self.dim)
        
        # Generate new solution
        new_nest = self.nests[index] + step_size * step * (self.nests[index] - self.best_solution)
        
        # Clip to bounds
        new_nest = self.clip_bounds(new_nest)
        
        return new_nest
    
    def _abandon_worst_nests(self) -> None:
        """Abandon worst nests with probability pa."""
        # Number of nests to abandon
        n_abandon = int(self.pa * self.pop_size)
        
        # Get indices of worst nests
        worst_indices = np.argsort(self.fitness_values)[-n_abandon:]
        
        # Generate new nests for abandoned ones
        for i in worst_indices:
            # Generate random walk
            step_size = self.rng.random(self.dim)
            
            # Select two random nests
            idx1, idx2 = self.rng.choice(self.pop_size, size=2, replace=False)
            
            # Generate new nest
            self.nests[i] = self.nests[i] + step_size * (self.nests[idx1] - self.nests[idx2])
            
            # Clip to bounds
            self.nests[i] = self.clip_bounds(self.nests[i])
            
            # Evaluate new nest
            self.fitness_values[i] = self.evaluate(self.nests[i])
    
    def update_population(self) -> None:
        """Perform one iteration of CS algorithm."""
        # Generate new solutions via Lévy flights
        for i in range(self.pop_size):
            # Get a cuckoo randomly by Lévy flights
            new_nest = self._get_cuckoo(i)
            
            # Evaluate new solution
            new_fitness = self.evaluate(new_nest)
            
            # Choose a random nest
            j = self.rng.integers(0, self.pop_size)
            
            # Replace if better
            if new_fitness < self.fitness_values[j]:
                self.nests[j] = new_nest
                self.fitness_values[j] = new_fitness
        
        # Abandon worst nests
        self._abandon_worst_nests()
        
        # Update population
        self.population = self.nests.copy()
        
        # Update best solution
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.nests[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "CS"
        """
        return "CS"
