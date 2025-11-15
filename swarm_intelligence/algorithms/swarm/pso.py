"""
Particle Swarm Optimization (PSO) Algorithm.

This module implements the PSO algorithm for continuous optimization problems.

Mathematical Formulation:
    Velocity update:
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
    
    Position update:
        x_i(t+1) = x_i(t) + v_i(t+1)
    
    where:
        w: inertia weight
        c1, c2: cognitive and social coefficients
        r1, r2: random numbers in [0,1]
        pbest_i: personal best position of particle i
        gbest: global best position

References:
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95 - International Conference on Neural Networks,
    4, 1942-1948.
"""

import numpy as np
from typing import Optional, Callable, Any
from numpy.typing import NDArray

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class PSO(BaseOptimizer):
    """Particle Swarm Optimization algorithm.
    
    PSO is a population-based stochastic optimization technique inspired by
    the social behavior of bird flocking or fish schooling.
    
    Attributes:
        w (float): Inertia weight controlling previous velocity influence.
        c1 (float): Cognitive coefficient (personal best attraction).
        c2 (float): Social coefficient (global best attraction).
        v_max (NDArray): Maximum velocity magnitude per dimension.
        velocities (NDArray): Current velocities of all particles.
        personal_best_positions (NDArray): Best positions found by each particle.
        personal_best_fitness (NDArray): Fitness values of personal bests.
        global_best_position (NDArray): Best position found by entire swarm.
        global_best_fitness (float): Fitness of global best position.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Sphere
        >>> problem = Sphere(dim=10)
        >>> optimizer = PSO(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     pop_size=50,
        ...     max_iter=1000,
        ...     w=0.7,
        ...     c1=1.5,
        ...     c2=1.5
        ... )
        >>> result = optimizer.optimize()
        >>> print(f"Best fitness: {result['best_fitness']:.6f}")
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 50,
        max_iter: int = 1000,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: Optional[NDArray[np.float64]] = None,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize PSO optimizer.
        
        Args:
            objective_func: Function to minimize, takes array and returns float.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Number of particles in the swarm.
            max_iter: Maximum number of iterations.
            w: Inertia weight (typical range: 0.4-0.9).
            c1: Cognitive coefficient (typical range: 1.5-2.0).
            c2: Social coefficient (typical range: 1.5-2.0).
            v_max: Maximum velocity per dimension (if None, set to 10% of search range).
            seed: Random seed for reproducibility.
            visualizer: Optional visualizer for real-time plotting.
        
        Raises:
            ValueError: If bounds shape is incorrect or parameters are invalid.
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
        
        # PSO-specific parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Set maximum velocity
        if v_max is None:
            search_range = bounds[1] - bounds[0]
            self.v_max = 0.1 * search_range
        else:
            self.v_max = np.array(v_max, dtype=np.float64)
        
        # Initialize PSO-specific attributes
        self.velocities: Optional[NDArray[np.float64]] = None
        self.personal_best_positions: Optional[NDArray[np.float64]] = None
        self.personal_best_fitness: Optional[NDArray[np.float64]] = None
        self.global_best_position: Optional[NDArray[np.float64]] = None
        self.global_best_fitness: float = float('inf')
        
    def initialize_population(self) -> None:
        """Initialize particle positions and velocities.
        
        Particles are initialized uniformly within bounds.
        Velocities are initialized randomly in range [-v_max, v_max].
        """
        # Initialize positions uniformly within bounds
        self.population = self.rng.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.dim)
        )
        
        # Initialize velocities
        self.velocities = self.rng.uniform(
            low=-self.v_max,
            high=self.v_max,
            size=(self.pop_size, self.dim)
        )
        
        # Evaluate initial population
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        
        # Initialize personal bests
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = fitness.copy()
        
        # Initialize global best
        best_idx = np.argmin(fitness)
        self.global_best_position = self.population[best_idx].copy()
        self.global_best_fitness = fitness[best_idx]
        
        # Update tracking
        self.best_solution = self.global_best_position.copy()
        self.best_fitness = self.global_best_fitness
        self.fitness_history.append(self.best_fitness)
        
    def update_population(self) -> None:
        """Perform one iteration of PSO updates.
        
        Updates velocities and positions according to PSO equations,
        then updates personal and global bests.
        """
        # Generate random coefficients
        r1 = self.rng.random((self.pop_size, self.dim))
        r2 = self.rng.random((self.pop_size, self.dim))
        
        # Update velocities
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.population)
        social = self.c2 * r2 * (self.global_best_position - self.population)
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Clip velocities to v_max
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        
        # Update positions
        self.population += self.velocities
        
        # Enforce bounds
        self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
        
        # Evaluate new positions
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        
        # Update personal bests
        improved = fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.population[improved]
        self.personal_best_fitness[improved] = fitness[improved]
        
        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.global_best_fitness:
            self.global_best_position = self.population[best_idx].copy()
            self.global_best_fitness = fitness[best_idx]
            self.best_solution = self.global_best_position.copy()
            self.best_fitness = self.global_best_fitness
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "PSO"
        """
        return "PSO"
