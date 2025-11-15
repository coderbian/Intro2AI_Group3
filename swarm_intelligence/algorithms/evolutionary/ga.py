"""
Genetic Algorithm (GA).

The Genetic Algorithm is inspired by the process of natural selection.
It uses selection, crossover, and mutation operators to evolve solutions.

Algorithm Steps:
    1. Selection: Select parents based on fitness
    2. Crossover: Combine parents to create offspring
    3. Mutation: Apply random changes to offspring
    4. Replacement: Form new population

References:
    Holland, J. H. (1992). Adaptation in natural and artificial systems.
    MIT press.
"""

import numpy as np
from typing import Optional, Callable, Any, Literal
from numpy.typing import NDArray

from swarm_intelligence.core.base_algorithm import BaseOptimizer


class GA(BaseOptimizer):
    """Genetic Algorithm for optimization.
    
    GA evolves a population of solutions using selection, crossover,
    and mutation operators inspired by natural evolution.
    
    Attributes:
        crossover_rate (float): Probability of crossover.
        mutation_rate (float): Probability of mutation.
        selection_method (str): Selection method ('tournament', 'roulette', 'rank').
        tournament_size (int): Size for tournament selection.
        fitness_values (NDArray): Current population fitness.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Rastrigin
        >>> problem = Rastrigin(dim=10)
        >>> optimizer = GA(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     pop_size=50,
        ...     max_iter=1000,
        ...     crossover_rate=0.8,
        ...     mutation_rate=0.1
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
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        selection_method: Literal['tournament', 'roulette', 'rank'] = 'tournament',
        tournament_size: int = 3,
        seed: Optional[int] = None,
        visualizer: Optional[Any] = None
    ):
        """Initialize GA optimizer.
        
        Args:
            objective_func: Function to minimize.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Population size.
            max_iter: Maximum number of iterations.
            crossover_rate: Probability of performing crossover (0 to 1).
            mutation_rate: Probability of mutation per gene (0 to 1).
            selection_method: Method for parent selection.
            tournament_size: Number of individuals in tournament selection.
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
        
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        
        # Initialize GA-specific attributes
        self.fitness_values: Optional[NDArray[np.float64]] = None
        
    def initialize_population(self) -> None:
        """Initialize population randomly within bounds."""
        self.population = self.rng.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.dim)
        )
        
        # Evaluate initial population
        self.fitness_values = np.array([self.evaluate(ind) for ind in self.population])
        
        # Find initial best
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
        
    def _tournament_selection(self) -> NDArray[np.float64]:
        """Select a parent using tournament selection.
        
        Returns:
            NDArray: Selected parent.
        """
        # Select random individuals for tournament
        indices = self.rng.choice(self.pop_size, size=self.tournament_size, replace=False)
        tournament_fitness = self.fitness_values[indices]
        
        # Winner is the one with best fitness
        winner_idx = indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _roulette_selection(self) -> NDArray[np.float64]:
        """Select a parent using roulette wheel selection.
        
        Returns:
            NDArray: Selected parent.
        """
        # Convert fitness to selection probabilities (for minimization)
        max_fitness = np.max(self.fitness_values)
        adjusted_fitness = max_fitness - self.fitness_values + 1e-10
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
        # Select based on probabilities
        idx = self.rng.choice(self.pop_size, p=probabilities)
        return self.population[idx].copy()
    
    def _rank_selection(self) -> NDArray[np.float64]:
        """Select a parent using rank selection.
        
        Returns:
            NDArray: Selected parent.
        """
        # Rank individuals by fitness
        ranks = np.argsort(np.argsort(self.fitness_values))
        
        # Create probabilities based on ranks (lower rank = better fitness)
        probabilities = (self.pop_size - ranks) / np.sum(np.arange(1, self.pop_size + 1))
        
        # Select based on rank probabilities
        idx = self.rng.choice(self.pop_size, p=probabilities)
        return self.population[idx].copy()
    
    def _selection(self) -> NDArray[np.float64]:
        """Select a parent based on the configured selection method.
        
        Returns:
            NDArray: Selected parent.
        """
        if self.selection_method == 'tournament':
            return self._tournament_selection()
        elif self.selection_method == 'roulette':
            return self._roulette_selection()
        elif self.selection_method == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _crossover(self, parent1: NDArray[np.float64], 
                   parent2: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent.
            parent2: Second parent.
            
        Returns:
            tuple: Two offspring.
        """
        if self.rng.random() < self.crossover_rate:
            # Arithmetic crossover
            alpha = self.rng.random()
            offspring1 = alpha * parent1 + (1 - alpha) * parent2
            offspring2 = (1 - alpha) * parent1 + alpha * parent2
            return offspring1, offspring2
        else:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
    
    def _mutation(self, individual: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate.
            
        Returns:
            NDArray: Mutated individual.
        """
        mutant = individual.copy()
        
        for i in range(self.dim):
            if self.rng.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_range = self.bounds[1][i] - self.bounds[0][i]
                mutant[i] += self.rng.normal(0, 0.1 * mutation_range)
        
        # Clip to bounds
        mutant = self.clip_bounds(mutant)
        return mutant
    
    def update_population(self) -> None:
        """Perform one generation of the genetic algorithm."""
        # Create offspring
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Select parents
            parent1 = self._selection()
            parent2 = self._selection()
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutation(child1)
            child2 = self._mutation(child2)
            
            offspring.extend([child1, child2])
        
        # Trim to population size
        offspring = offspring[:self.pop_size]
        
        # Evaluate offspring
        offspring = np.array(offspring)
        offspring_fitness = np.array([self.evaluate(ind) for ind in offspring])
        
        # Elitism: keep best individual from previous generation
        best_idx = np.argmin(self.fitness_values)
        worst_offspring_idx = np.argmax(offspring_fitness)
        
        if self.fitness_values[best_idx] < offspring_fitness[worst_offspring_idx]:
            offspring[worst_offspring_idx] = self.population[best_idx].copy()
            offspring_fitness[worst_offspring_idx] = self.fitness_values[best_idx]
        
        # Replace population
        self.population = offspring
        self.fitness_values = offspring_fitness
        
        # Update best solution
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "GA"
        """
        return "GA"
