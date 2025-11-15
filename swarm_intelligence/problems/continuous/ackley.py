"""
Ackley Function - Multimodal benchmark function.

The Ackley function is a widely used multimodal test function. It has many
local minima with a single global minimum.

Mathematical Definition:
    f(x) = -a*exp(-b*sqrt(1/n * Σ x_i²)) - exp(1/n * Σ cos(c*x_i)) + a + exp(1)
    
    where a=20, b=0.2, c=2π, and n is the dimensionality.

Properties:
    - Domain: x_i ∈ [-32.768, 32.768]
    - Global minimum: f(0, ..., 0) = 0
    - Characteristics: Nearly flat outer region, many local minima
    - Difficulty: Medium

References:
    Ackley, D. H. (1987). A connectionist machine for genetic hillclimbing.
    Kluwer Academic Publishers, Boston MA.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from swarm_intelligence.core.base_problem import OptimizationProblem


class Ackley(OptimizationProblem):
    """Ackley function for continuous optimization.
    
    A multimodal benchmark function with many local minima. The surface
    is characterized by a nearly flat outer region and a large hole at
    the center.
    
    Attributes:
        dim (int): Problem dimensionality.
        a (float): Parameter a (default: 20).
        b (float): Parameter b (default: 0.2).
        c (float): Parameter c (default: 2π).
        domain (tuple): Search space domain (-32.768, 32.768).
    
    Example:
        >>> problem = Ackley(dim=10)
        >>> x = np.zeros(10)
        >>> fitness = problem.evaluate(x)
        >>> print(f"f(0) = {fitness:.10f}")  # Should be ~0
        0.0000000000
    """
    
    def __init__(self, dim: int, a: float = 20.0, b: float = 0.2, c: float = 2*np.pi):
        """Initialize Ackley function.
        
        Args:
            dim: Dimensionality of the problem (number of variables).
            a: Parameter a (default: 20.0).
            b: Parameter b (default: 0.2).
            c: Parameter c (default: 2π).
        
        Raises:
            ValueError: If dim < 1.
        """
        if dim < 1:
            raise ValueError(f"Dimension must be >= 1, got {dim}")
        
        super().__init__(dim=dim, problem_type='continuous', name='Ackley')
        self.a = a
        self.b = b
        self.c = c
        self.domain = (-32.768, 32.768)
        
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate the Ackley function at point x.
        
        Args:
            x: Input array of shape (dim,).
        
        Returns:
            float: Function value at x.
        
        Raises:
            ValueError: If x has wrong shape.
        """
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected array of length {self.dim}, got {x.shape[0]}")
        
        # f(x) = -a*exp(-b*sqrt(1/n * Σ x_i²)) - exp(1/n * Σ cos(c*x_i)) + a + e
        n = self.dim
        
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        
        return float(term1 + term2 + self.a + np.e)
    
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
            float: Optimal value (0 for Ackley).
        """
        return 0.0
    
    def get_optimal_solution(self) -> NDArray[np.float64]:
        """Get known optimal solution.
        
        Returns:
            NDArray: Optimal solution (origin for Ackley).
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
                'Multimodal',
                'Many local minima',
                'Nearly flat outer region',
                'Global optimum at origin',
                'Continuous and differentiable'
            ],
            'difficulty': 'Medium',
            'formula': 'f(x) = -20*exp(-0.2*sqrt(mean(x²))) - exp(mean(cos(2πx))) + 20 + e'
        }
