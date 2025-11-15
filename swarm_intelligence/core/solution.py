"""
Solution Module.

This module provides a class to represent individual candidate solutions
in optimization algorithms.
"""

from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray


class Solution:
    """Represents a candidate solution in optimization.
    
    This class encapsulates a solution with its position, fitness value,
    and optional metadata for algorithm-specific information.
    
    Attributes:
        position (NDArray): Decision variables of the solution.
        fitness (Optional[float]): Fitness value of the solution.
        velocity (Optional[NDArray]): Velocity vector (for PSO-like algorithms).
        metadata (Dict): Additional algorithm-specific data.
        
    Example:
        >>> sol = Solution(position=np.array([1.0, 2.0, 3.0]), fitness=14.0)
        >>> sol.fitness
        14.0
        >>> sol2 = sol.copy()
        >>> sol2.position[0] = 5.0
        >>> sol.position[0]  # Original unchanged
        1.0
    """
    
    def __init__(
        self,
        position: NDArray[np.float64],
        fitness: Optional[float] = None,
        velocity: Optional[NDArray[np.float64]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a solution.
        
        Args:
            position: Array of decision variables.
            fitness: Fitness value (None if not yet evaluated).
            velocity: Velocity vector (optional, for PSO-like algorithms).
            metadata: Dictionary for algorithm-specific data.
        """
        self.position = np.array(position, dtype=np.float64)
        self.fitness = fitness
        self.velocity = np.array(velocity, dtype=np.float64) if velocity is not None else None
        self.metadata = metadata if metadata is not None else {}
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution.
        
        Returns:
            Solution: A new Solution instance with copied data.
        """
        return Solution(
            position=self.position.copy(),
            fitness=self.fitness,
            velocity=self.velocity.copy() if self.velocity is not None else None,
            metadata=self.metadata.copy()
        )
    
    def distance_to(self, other: 'Solution') -> float:
        """Calculate Euclidean distance to another solution.
        
        Args:
            other: Another Solution instance.
            
        Returns:
            float: Euclidean distance between positions.
        """
        return np.linalg.norm(self.position - other.position)
    
    def __lt__(self, other: 'Solution') -> bool:
        """Less than comparison based on fitness (for minimization).
        
        Args:
            other: Another Solution instance.
            
        Returns:
            bool: True if this solution has lower fitness.
        """
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness < other.fitness
    
    def __le__(self, other: 'Solution') -> bool:
        """Less than or equal comparison.
        
        Args:
            other: Another Solution instance.
            
        Returns:
            bool: True if this solution has fitness <= other's fitness.
        """
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness <= other.fitness
    
    def __gt__(self, other: 'Solution') -> bool:
        """Greater than comparison.
        
        Args:
            other: Another Solution instance.
            
        Returns:
            bool: True if this solution has higher fitness.
        """
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness > other.fitness
    
    def __ge__(self, other: 'Solution') -> bool:
        """Greater than or equal comparison.
        
        Args:
            other: Another Solution instance.
            
        Returns:
            bool: True if this solution has fitness >= other's fitness.
        """
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness >= other.fitness
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison.
        
        Args:
            other: Another object.
            
        Returns:
            bool: True if positions and fitness are equal.
        """
        if not isinstance(other, Solution):
            return False
        position_equal = np.allclose(self.position, other.position)
        fitness_equal = (self.fitness == other.fitness if 
                        self.fitness is not None and other.fitness is not None 
                        else self.fitness is other.fitness)
        return position_equal and fitness_equal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the solution.
        """
        return {
            'position': self.position.tolist(),
            'fitness': float(self.fitness) if self.fitness is not None else None,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation.
        
        Returns:
            str: Human-readable description.
        """
        return f"Solution(fitness={self.fitness:.6f if self.fitness is not None else 'None'})"
    
    def __repr__(self) -> str:
        """Detailed string representation.
        
        Returns:
            str: Detailed description for debugging.
        """
        return (f"Solution(position={self.position}, fitness={self.fitness}, "
                f"has_velocity={self.velocity is not None})")
