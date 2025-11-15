"""
Rastrigin Function - Multimodal benchmark function.

The Rastrigin function is a highly multimodal function with many local minima,
making it challenging for optimization algorithms.

Mathematical Definition:
    f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]
    
    where n is the dimensionality.

Properties:
    - Domain: x_i ∈ [-5.12, 5.12]
    - Global minimum: f(0, ..., 0) = 0
    - Characteristics: Highly multimodal with many regularly distributed local minima
    - Difficulty: Hard

References:
    Rastrigin, L. A. (1974). Systems of extremal control.
    Mir, Moscow.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from swarm_intelligence.core.base_problem import OptimizationProblem


class Rastrigin(OptimizationProblem):
    """Rastrigin function for continuous optimization.
    
    A highly multimodal benchmark function commonly used to test
    optimization algorithms' ability to escape local optima.
    
    Attributes:
        dim (int): Problem dimensionality.
        A (float): Amplitude parameter (default: 10).
        domain (tuple): Search space domain (-5.12, 5.12).
    
    Example:
        >>> problem = Rastrigin(dim=10)
        >>> x = np.zeros(10)
        >>> fitness = problem.evaluate(x)
        >>> print(f"f(0) = {fitness}")  # Should be 0
        0.0
    """
    
    def __init__(self, dim: int, A: float = 10.0):
        """Initialize Rastrigin function.
        
        Args:
            dim: Dimensionality of the problem (number of variables).
            A: Amplitude parameter (default: 10.0).
        
        Raises:
            ValueError: If dim < 1.
        """
        if dim < 1:
            raise ValueError(f"Dimension must be >= 1, got {dim}")
        
        super().__init__(dim=dim, problem_type='continuous', name='Rastrigin')
        self.A = A
        self.domain = (-5.12, 5.12)
        
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate the Rastrigin function at point x.
        
        Args:
            x: Input array of shape (dim,).
        
        Returns:
            float: Function value at x.
        
        Raises:
            ValueError: If x has wrong shape.
        """
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected array of length {self.dim}, got {x.shape[0]}")
        
        # f(x) = A*n + Σ[x_i² - A*cos(2π*x_i)]
        n = self.dim
        sum_term = np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
        return float(self.A * n + sum_term)
    
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
            float: Optimal value (0 for Rastrigin).
        """
        return 0.0
    
    def get_optimal_solution(self) -> NDArray[np.float64]:
        """Get known optimal solution.
        
        Returns:
            NDArray: Optimal solution (origin for Rastrigin).
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
                'Highly multimodal',
                'Many regularly distributed local minima',
                'Global optimum at origin',
                'Continuous and differentiable'
            ],
            'difficulty': 'Hard',
            'formula': 'f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]'
        }
