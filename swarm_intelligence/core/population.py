"""
Population Module.

This module provides a class to manage a collection of solutions.
"""

from typing import List, Iterator, Optional
import numpy as np
from numpy.typing import NDArray

from swarm_intelligence.core.solution import Solution


class Population:
    """Manages a collection of solutions.
    
    This class provides convenient methods for managing and analyzing
    a population of candidate solutions in optimization algorithms.
    
    Attributes:
        solutions (List[Solution]): List of Solution objects.
        size (int): Number of solutions in the population.
        
    Example:
        >>> pop = Population()
        >>> pop.add(Solution(np.array([1.0, 2.0]), fitness=5.0))
        >>> pop.add(Solution(np.array([3.0, 4.0]), fitness=25.0))
        >>> best = pop.get_best()
        >>> best.fitness
        5.0
    """
    
    def __init__(self, solutions: Optional[List[Solution]] = None):
        """Initialize population.
        
        Args:
            solutions: Optional initial list of solutions.
        """
        self.solutions = solutions if solutions is not None else []
    
    @property
    def size(self) -> int:
        """Get population size.
        
        Returns:
            int: Number of solutions in population.
        """
        return len(self.solutions)
    
    def add(self, solution: Solution) -> None:
        """Add a solution to the population.
        
        Args:
            solution: Solution to add.
        """
        self.solutions.append(solution)
    
    def get_best(self) -> Optional[Solution]:
        """Get the solution with the best (lowest) fitness.
        
        Returns:
            Solution or None: Best solution, or None if population is empty.
        """
        if not self.solutions:
            return None
        valid_solutions = [s for s in self.solutions if s.fitness is not None]
        if not valid_solutions:
            return None
        return min(valid_solutions, key=lambda s: s.fitness)
    
    def get_worst(self) -> Optional[Solution]:
        """Get the solution with the worst (highest) fitness.
        
        Returns:
            Solution or None: Worst solution, or None if population is empty.
        """
        if not self.solutions:
            return None
        valid_solutions = [s for s in self.solutions if s.fitness is not None]
        if not valid_solutions:
            return None
        return max(valid_solutions, key=lambda s: s.fitness)
    
    def get_diversity(self) -> float:
        """Calculate population diversity (average pairwise distance).
        
        Returns:
            float: Average Euclidean distance between all solution pairs.
        """
        if self.size < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(self.size):
            for j in range(i + 1, self.size):
                total_distance += self.solutions[i].distance_to(self.solutions[j])
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def sort(self, reverse: bool = False) -> None:
        """Sort solutions by fitness.
        
        Args:
            reverse: If True, sort in descending order (worst first).
        """
        self.solutions.sort(key=lambda s: s.fitness if s.fitness is not None else float('inf'),
                           reverse=reverse)
    
    def replace(self, index: int, solution: Solution) -> None:
        """Replace solution at given index.
        
        Args:
            index: Index of solution to replace.
            solution: New solution.
            
        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for population of size {self.size}")
        self.solutions[index] = solution
    
    def get_positions(self) -> NDArray[np.float64]:
        """Get positions of all solutions as a 2D array.
        
        Returns:
            NDArray: Array of shape (size, dim) with all positions.
        """
        if not self.solutions:
            return np.array([])
        return np.array([sol.position for sol in self.solutions])
    
    def get_fitness_values(self) -> NDArray[np.float64]:
        """Get fitness values of all solutions.
        
        Returns:
            NDArray: Array of fitness values.
        """
        return np.array([sol.fitness if sol.fitness is not None else float('inf') 
                        for sol in self.solutions])
    
    def get_mean_fitness(self) -> float:
        """Calculate mean fitness of population.
        
        Returns:
            float: Mean fitness value.
        """
        fitness_values = [s.fitness for s in self.solutions if s.fitness is not None]
        return np.mean(fitness_values) if fitness_values else float('inf')
    
    def get_std_fitness(self) -> float:
        """Calculate standard deviation of fitness.
        
        Returns:
            float: Standard deviation of fitness values.
        """
        fitness_values = [s.fitness for s in self.solutions if s.fitness is not None]
        return np.std(fitness_values) if fitness_values else 0.0
    
    def clear(self) -> None:
        """Remove all solutions from population."""
        self.solutions.clear()
    
    def __len__(self) -> int:
        """Get population size.
        
        Returns:
            int: Number of solutions.
        """
        return self.size
    
    def __getitem__(self, index: int) -> Solution:
        """Get solution at index.
        
        Args:
            index: Solution index.
            
        Returns:
            Solution: Solution at the given index.
        """
        return self.solutions[index]
    
    def __iter__(self) -> Iterator[Solution]:
        """Iterate over solutions.
        
        Returns:
            Iterator: Iterator over solutions.
        """
        return iter(self.solutions)
    
    def __str__(self) -> str:
        """String representation.
        
        Returns:
            str: Human-readable description.
        """
        best = self.get_best()
        best_fitness = best.fitness if best and best.fitness is not None else "N/A"
        return f"Population(size={self.size}, best_fitness={best_fitness})"
    
    def __repr__(self) -> str:
        """Detailed string representation.
        
        Returns:
            str: Detailed description for debugging.
        """
        return f"Population(size={self.size}, solutions={self.solutions})"
