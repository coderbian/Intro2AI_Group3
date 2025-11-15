"""
Rosenbrock Function - Valley-shaped benchmark function.

The Rosenbrock function is a non-convex function used as a performance test
problem for optimization algorithms. The global minimum is inside a long,
narrow, parabolic shaped flat valley.

Mathematical Definition:
    f(x) = Σ[100*(x_{i+1} - x_i²)² + (1-x_i)²]
    
    where n is the dimensionality.

Properties:
    - Domain: x_i ∈ [-5, 10]
    - Global minimum: f(1, ..., 1) = 0
    - Characteristics: Valley-shaped, difficult for optimization algorithms
    - Difficulty: Medium to Hard

References:
    Rosenbrock, H. H. (1960). An Automatic Method for Finding the Greatest
    or Least Value of a Function. The Computer Journal, 3(3), 175-184.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from swarm_intelligence.core.base_problem import OptimizationProblem


class Rosenbrock(OptimizationProblem):
    """Rosenbrock function for continuous optimization.
    
    A classic benchmark function with a narrow valley leading to the global minimum.
    Finding the valley is easy, but converging to the global minimum is difficult.
    
    Attributes:
        dim (int): Problem dimensionality (must be >= 2).
        domain (tuple): Search space domain (-5, 10).
    
    Example:
        >>> problem = Rosenbrock(dim=10)
        >>> x = np.ones(10)
        >>> fitness = problem.evaluate(x)
        >>> print(f"f(1,...,1) = {fitness}")  # Should be 0
        0.0
    """
    
    def __init__(self, dim: int):
        """Initialize Rosenbrock function.
        
        Args:
            dim: Dimensionality of the problem (number of variables, must be >= 2).
        
        Raises:
            ValueError: If dim < 2.
        """
        if dim < 2:
            raise ValueError(f"Rosenbrock function requires dimension >= 2, got {dim}")
        
        super().__init__(dim=dim, problem_type='continuous', name='Rosenbrock')
        self.domain = (-5.0, 10.0)
        
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate the Rosenbrock function at point x.
        
        Args:
            x: Input array of shape (dim,).
        
        Returns:
            float: Function value at x.
        
        Raises:
            ValueError: If x has wrong shape.
        """
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected array of length {self.dim}, got {x.shape[0]}")
        
        # f(x) = Σ[100*(x_{i+1} - x_i²)² + (1-x_i)²]
        sum_term = 0.0
        for i in range(self.dim - 1):
            sum_term += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        return float(sum_term)
    
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
            float: Optimal value (0 for Rosenbrock).
        """
        return 0.0
    
    def get_optimal_solution(self) -> NDArray[np.float64]:
        """Get known optimal solution.
        
        Returns:
            NDArray: Optimal solution (all ones for Rosenbrock).
        """
        return np.ones(self.dim)
    
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
                'Valley-shaped',
                'Non-convex',
                'Global minimum in narrow parabolic valley',
                'Continuous and differentiable',
                'Easy to find valley, hard to converge'
            ],
            'difficulty': 'Medium to Hard',
            'formula': 'f(x) = Σ[100*(x_{i+1} - x_i²)² + (1-x_i)²]'
        }
