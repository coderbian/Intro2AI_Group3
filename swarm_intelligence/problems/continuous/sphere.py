"""
Sphere Function - Simple unimodal benchmark function.

The Sphere function is one of the simplest benchmark functions with a single global minimum.

Mathematical Definition:
    f(x) = Σ x_i²
    
    where n is the dimensionality.

Properties:
    - Domain: x_i ∈ [-100, 100]
    - Global minimum: f(0, ..., 0) = 0
    - Characteristics: Unimodal, convex, separable
    - Difficulty: Easy

References:
    Standard benchmark function in optimization literature.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from swarm_intelligence.core.base_problem import OptimizationProblem


class Sphere(OptimizationProblem):
    """Sphere function for continuous optimization.
    
    A simple unimodal benchmark function with a global minimum at the origin.
    
    Attributes:
        dim (int): Problem dimensionality.
        domain (tuple): Search space domain (-100, 100).
    
    Example:
        >>> problem = Sphere(dim=10)
        >>> x = np.zeros(10)
        >>> fitness = problem.evaluate(x)
        >>> print(f"f(0) = {fitness}")  # Should be 0
        0.0
    """
    
    def __init__(self, dim: int):
        """Initialize Sphere function.
        
        Args:
            dim: Dimensionality of the problem (number of variables).
        
        Raises:
            ValueError: If dim < 1.
        """
        if dim < 1:
            raise ValueError(f"Dimension must be >= 1, got {dim}")
        
        super().__init__(dim=dim, problem_type='continuous', name='Sphere')
        self.domain = (-100.0, 100.0)
        
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate the Sphere function at point x.
        
        Args:
            x: Input array of shape (dim,).
        
        Returns:
            float: Function value at x.
        
        Raises:
            ValueError: If x has wrong shape.
        """
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected array of length {self.dim}, got {x.shape[0]}")
        
        # f(x) = Σ x_i²
        return float(np.sum(x**2))
    
    def get_bounds(self) -> NDArray[np.float64]:
        """Get search space bounds.
        
        Returns:
            NDArray: Array of shape (2, dim) with lower and upper bounds.
        """
        lower = np.full(self.dim, self.domain[0])
        upper = np.full(self.dim, self.domain[1])
        return np.array([lower, upper])
    
    def get_optimal_value(self) -> float:
        """Get known optimal value.
        
        Returns:
            float: Optimal value (0 for Sphere).
        """
        return 0.0
    
    def get_optimal_solution(self) -> NDArray[np.float64]:
        """Get known optimal solution.
        
        Returns:
            NDArray: Optimal solution (origin for Sphere).
        """
        return np.zeros(self.dim)
    
    def get_problem_info(self) -> dict:
        """Get comprehensive problem information.
        
        Returns:
            dict: Problem metadata including name, type, characteristics.
        """
        return {
            'name': self.name,
            'type': self.problem_type,
            'dim': self.dim,
            'domain': self.domain,
            'optimal_value': self.get_optimal_value(),
            'optimal_solution': self.get_optimal_solution().tolist(),
            'characteristics': [
                'Unimodal',
                'Convex',
                'Separable',
                'Global optimum at origin',
                'Continuous and differentiable'
            ],
            'difficulty': 'Easy',
            'formula': 'f(x) = Σ x_i²'
        }
