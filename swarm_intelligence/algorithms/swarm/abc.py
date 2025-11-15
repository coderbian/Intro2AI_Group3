"""
Artificial Bee Colony (ABC) Algorithm.

The ABC algorithm mimics the foraging behavior of honey bees. It uses three types
of bees: employed bees, onlooker bees, and scout bees.

Algorithm Phases:
    1. Employed Bee Phase: Each employed bee searches around its food source
    2. Onlooker Bee Phase: Onlookers select food sources probabilistically
    3. Scout Bee Phase: Exhausted food sources are abandoned and replaced

References:
    Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for
    numerical function optimization: artificial bee colony (ABC) algorithm.
    Journal of global optimization, 39(3), 459-471.
"""

import numpy as np
from typing import Optional, Callable, Any
from numpy.typing import NDArray

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class ABC(BaseOptimizer):
    """Artificial Bee Colony optimization algorithm.
    
    ABC simulates the foraging behavior of honey bee swarms. The algorithm
    uses three types of bees with different roles to explore and exploit
    the search space.
    
    Attributes:
        limit (int): Abandonment limit for exhausted food sources.
        n_employed (int): Number of employed bees.
        n_onlooker (int): Number of onlooker bees.
        food_sources (NDArray): Positions of food sources.
        fitness_values (NDArray): Fitness of food sources.
        trial_counters (NDArray): Trial counters for each food source.
        probabilities (NDArray): Selection probabilities for onlookers.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Rastrigin
        >>> problem = Rastrigin(dim=10)
        >>> optimizer = ABC(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     pop_size=50,
        ...     max_iter=1000,
        ...     limit=100
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 50,
        max_iter: int = 1000,
        limit: int = 100,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize ABC optimizer.
        
        Args:
            objective_func: Function to minimize.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Number of food sources (employed bees).
            max_iter: Maximum number of iterations.
            limit: Abandonment limit for exhausted sources.
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
        
        self.limit = limit
        self.n_employed = pop_size
        self.n_onlooker = pop_size
        
        # Initialize ABC-specific attributes
        self.food_sources: Optional[NDArray[np.float64]] = None
        self.fitness_values: Optional[NDArray[np.float64]] = None
        self.trial_counters: Optional[NDArray[np.int32]] = None
        self.probabilities: Optional[NDArray[np.float64]] = None
        
    def initialize_population(self) -> None:
        """Initialize food sources randomly within bounds."""
        # Initialize food sources
        self.food_sources = self.rng.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.n_employed, self.dim)
        )
        self.population = self.food_sources.copy()
        
        # Evaluate initial food sources
        self.fitness_values = np.array([self.evaluate(food) for food in self.food_sources])
        
        # Initialize trial counters
        self.trial_counters = np.zeros(self.n_employed, dtype=np.int32)
        
        # Find initial best
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.food_sources[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
        
    def _employed_bee_phase(self) -> None:
        """Employed bees search around their food sources."""
        for i in range(self.n_employed):
            # Select a random dimension
            j = self.rng.integers(0, self.dim)
            
            # Select a random neighbor (different from i)
            k = self.rng.integers(0, self.n_employed)
            while k == i:
                k = self.rng.integers(0, self.n_employed)
            
            # Generate new solution
            new_solution = self.food_sources[i].copy()
            phi = self.rng.uniform(-1, 1)
            new_solution[j] = self.food_sources[i][j] + phi * (
                self.food_sources[i][j] - self.food_sources[k][j]
            )
            
            # Clip to bounds
            new_solution = self.clip_bounds(new_solution)
            
            # Evaluate new solution
            new_fitness = self.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
                
    def _calculate_probabilities(self) -> None:
        """Calculate selection probabilities for onlooker bees."""
        # Use fitness proportionate selection
        # For minimization, better fitness means higher probability
        max_fitness = np.max(self.fitness_values)
        if max_fitness > 0:
            adjusted_fitness = max_fitness - self.fitness_values + 1e-10
        else:
            adjusted_fitness = 1.0 / (self.fitness_values + 1e-10)
        
        self.probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
    def _onlooker_bee_phase(self) -> None:
        """Onlooker bees select food sources probabilistically."""
        self._calculate_probabilities()
        
        for _ in range(self.n_onlooker):
            # Select a food source based on probability
            i = self.rng.choice(self.n_employed, p=self.probabilities)
            
            # Select a random dimension
            j = self.rng.integers(0, self.dim)
            
            # Select a random neighbor
            k = self.rng.integers(0, self.n_employed)
            while k == i:
                k = self.rng.integers(0, self.n_employed)
            
            # Generate new solution
            new_solution = self.food_sources[i].copy()
            phi = self.rng.uniform(-1, 1)
            new_solution[j] = self.food_sources[i][j] + phi * (
                self.food_sources[i][j] - self.food_sources[k][j]
            )
            
            # Clip to bounds
            new_solution = self.clip_bounds(new_solution)
            
            # Evaluate new solution
            new_fitness = self.evaluate(new_solution)
            
            # Greedy selection
            if new_fitness < self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
                
    def _scout_bee_phase(self) -> None:
        """Scout bees replace exhausted food sources."""
        for i in range(self.n_employed):
            if self.trial_counters[i] >= self.limit:
                # Generate new random food source
                self.food_sources[i] = self.rng.uniform(
                    low=self.bounds[0],
                    high=self.bounds[1],
                    size=self.dim
                )
                self.fitness_values[i] = self.evaluate(self.food_sources[i])
                self.trial_counters[i] = 0
                
    def update_population(self) -> None:
        """Perform one iteration of ABC algorithm."""
        # Employed bee phase
        self._employed_bee_phase()
        
        # Onlooker bee phase
        self._onlooker_bee_phase()
        
        # Scout bee phase
        self._scout_bee_phase()
        
        # Update population and best solution
        self.population = self.food_sources.copy()
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.food_sources[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "ABC"
        """
        return "ABC"
