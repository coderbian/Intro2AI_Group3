"""
Base Problem Module.

This module provides the abstract base class for optimization problems.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
from numpy.typing import NDArray


class OptimizationProblem(ABC):
    """Abstract base class for optimization problems.
    
    This class defines the interface that all optimization problems must implement.
    It supports both continuous and discrete optimization problems.
    
    Attributes:
        dim (int): Problem dimensionality (number of decision variables).
        problem_type (str): Type of problem - 'continuous' or 'discrete'.
        name (str): Name of the problem.
        
    Example:
        >>> class MyProblem(OptimizationProblem):
        ...     def __init__(self, dim):
        ...         super().__init__(dim, 'continuous', 'MyProblem')
        ...     def evaluate(self, x):
        ...         return np.sum(x**2)
        ...     def get_bounds(self):
        ...         return np.array([[-10]*self.dim, [10]*self.dim])
        ...     def get_optimal_value(self):
        ...         return 0.0
        ...     def get_optimal_solution(self):
        ...         return np.zeros(self.dim)
    """
    
    def __init__(self, dim: int, problem_type: str, name: str):
        """Initialize the optimization problem.
        
        Args:
            dim: Dimensionality of the problem.
            problem_type: Type of problem - 'continuous' or 'discrete'.
            name: Name identifier for the problem.
            
        Raises:
            ValueError: If dim < 1 or problem_type is invalid.
        """
        if dim < 1:
            raise ValueError(f"Dimension must be >= 1, got {dim}")
        if problem_type not in ['continuous', 'discrete']:
            raise ValueError(f"Problem type must be 'continuous' or 'discrete', got {problem_type}")
        
        self.dim = dim
        self.problem_type = problem_type
        self.name = name
    
    @abstractmethod
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate the objective function at point x.
        
        Args:
            x: Input array of decision variables, shape (dim,).
            
        Returns:
            float: Objective function value at x.
            
        Raises:
            ValueError: If x has incorrect shape.
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> NDArray[np.float64]:
        """Get the search space bounds.
        
        Returns:
            NDArray: Array of shape (2, dim) with [lower_bounds, upper_bounds].
        """
        pass
    
    @abstractmethod
    def get_optimal_value(self) -> Optional[float]:
        """Get the known optimal value if available.
        
        Returns:
            float or None: Optimal objective value, or None if unknown.
        """
        pass
    
    @abstractmethod
    def get_optimal_solution(self) -> Optional[NDArray[np.float64]]:
        """Get the known optimal solution if available.
        
        Returns:
            NDArray or None: Optimal solution, or None if unknown.
        """
        pass
    
    def is_feasible(self, x: NDArray[np.float64]) -> bool:
        """Check if a solution is feasible (within bounds).
        
        Args:
            x: Solution to check.
            
        Returns:
            bool: True if solution is within bounds.
        """
        bounds = self.get_bounds()
        return np.all(x >= bounds[0]) and np.all(x <= bounds[1])
    
    def batch_evaluate(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate multiple solutions (vectorized when possible).
        
        Args:
            population: Array of shape (pop_size, dim) with solutions.
            
        Returns:
            NDArray: Array of shape (pop_size,) with fitness values.
        """
        return np.array([self.evaluate(ind) for ind in population])
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get comprehensive problem metadata.
        
        Returns:
            dict: Dictionary containing problem information:
                - name: Problem name
                - type: Problem type
                - dim: Dimensionality
                - bounds: Search space bounds
                - optimal_value: Known optimal value (if available)
                - optimal_solution: Known optimal solution (if available)
        """
        return {
            'name': self.name,
            'type': self.problem_type,
            'dim': self.dim,
            'bounds': self.get_bounds().tolist(),
            'optimal_value': self.get_optimal_value(),
            'optimal_solution': (self.get_optimal_solution().tolist() 
                               if self.get_optimal_solution() is not None else None)
        }
    
    def __str__(self) -> str:
        """String representation of the problem.
        
        Returns:
            str: Problem description.
        """
        return f"{self.name} (dim={self.dim}, type={self.problem_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation.
        
        Returns:
            str: Detailed problem description.
        """
        return (f"{self.__class__.__name__}(dim={self.dim}, "
                f"problem_type='{self.problem_type}', name='{self.name}')")
