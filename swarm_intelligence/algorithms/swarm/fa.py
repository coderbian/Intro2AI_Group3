"""
Firefly Algorithm (FA).

The Firefly Algorithm is inspired by the flashing behavior of fireflies.
Fireflies are attracted to brighter fireflies, and the attractiveness
decreases with distance.

Mathematical Formulation:
    Attractiveness: β(r) = β₀ * exp(-γ * r²)
    Movement: x_i = x_i + β(r_ij) * (x_j - x_i) + α * (rand - 0.5)
    
    where:
        β₀: attractiveness at r=0
        γ: light absorption coefficient
        α: randomization parameter
        r_ij: distance between firefly i and j

References:
    Yang, X. S. (2009). Firefly algorithms for multimodal optimization.
    International symposium on stochastic algorithms (pp. 169-178).
"""

import numpy as np
from typing import Optional, Callable, Any
from numpy.typing import NDArray

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class FA(BaseOptimizer):
    """Firefly Algorithm for optimization.
    
    FA simulates the flashing behavior and movement of fireflies.
    Fireflies move towards brighter (better fitness) fireflies.
    
    Attributes:
        alpha (float): Randomization parameter.
        beta0 (float): Attractiveness at distance r=0.
        gamma (float): Light absorption coefficient.
        fireflies (NDArray): Positions of fireflies.
        intensities (NDArray): Light intensities (fitness values).
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Ackley
        >>> problem = Ackley(dim=10)
        >>> optimizer = FA(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     pop_size=30,
        ...     max_iter=1000,
        ...     alpha=0.5,
        ...     beta0=1.0,
        ...     gamma=1.0
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 30,
        max_iter: int = 1000,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 1.0,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize FA optimizer.
        
        Args:
            objective_func: Function to minimize.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Number of fireflies.
            max_iter: Maximum number of iterations.
            alpha: Randomization parameter (0 to 1).
            beta0: Attractiveness at r=0 (typically 1.0).
            gamma: Light absorption coefficient (typically 0.01 to 10).
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
        
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        # Initialize FA-specific attributes
        self.fireflies: Optional[NDArray[np.float64]] = None
        self.intensities: Optional[NDArray[np.float64]] = None
        
    def initialize_population(self) -> None:
        """Initialize firefly positions randomly within bounds."""
        # Initialize fireflies
        self.fireflies = self.rng.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.dim)
        )
        self.population = self.fireflies.copy()
        
        # Evaluate initial intensities (fitness)
        self.intensities = np.array([self.evaluate(firefly) for firefly in self.fireflies])
        
        # Find initial best
        best_idx = np.argmin(self.intensities)
        self.best_solution = self.fireflies[best_idx].copy()
        self.best_fitness = self.intensities[best_idx]
        self.fitness_history.append(self.best_fitness)
        
    def _calculate_distance(self, firefly_i: NDArray[np.float64], 
                          firefly_j: NDArray[np.float64]) -> float:
        """Calculate Euclidean distance between two fireflies.
        
        Args:
            firefly_i: Position of firefly i.
            firefly_j: Position of firefly j.
            
        Returns:
            float: Euclidean distance.
        """
        return float(np.linalg.norm(firefly_i - firefly_j))
    
    def _calculate_attractiveness(self, distance: float) -> float:
        """Calculate attractiveness based on distance.
        
        Args:
            distance: Distance between fireflies.
            
        Returns:
            float: Attractiveness value.
        """
        return self.beta0 * np.exp(-self.gamma * distance**2)
    
    def update_population(self) -> None:
        """Perform one iteration of FA algorithm."""
        # Move each firefly towards brighter ones
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                # If firefly j is brighter (better fitness) than firefly i
                if self.intensities[j] < self.intensities[i]:
                    # Calculate distance
                    r_ij = self._calculate_distance(self.fireflies[i], self.fireflies[j])
                    
                    # Calculate attractiveness
                    beta = self._calculate_attractiveness(r_ij)
                    
                    # Move firefly i towards firefly j
                    random_term = self.alpha * (self.rng.random(self.dim) - 0.5)
                    self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + random_term
                    
                    # Clip to bounds
                    self.fireflies[i] = self.clip_bounds(self.fireflies[i])
                    
                    # Evaluate new position
                    self.intensities[i] = self.evaluate(self.fireflies[i])
        
        # Update population
        self.population = self.fireflies.copy()
        
        # Update best solution
        best_idx = np.argmin(self.intensities)
        if self.intensities[best_idx] < self.best_fitness:
            self.best_solution = self.fireflies[best_idx].copy()
            self.best_fitness = self.intensities[best_idx]
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "FA"
        """
        return "FA"
