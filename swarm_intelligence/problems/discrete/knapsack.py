"""
0/1 Knapsack Problem

Select items to maximize total value while staying within weight capacity.

Mathematical Formulation:
------------------------
Maximize: V = Σ v_i · x_i

Subject to:
    Σ w_i · x_i ≤ W
    x_i ∈ {0, 1}

Where:
- v_i: Value of item i
- w_i: Weight of item i
- x_i: Binary variable (1 if item selected, 0 otherwise)
- W: Knapsack capacity

Problem Characteristics:
-----------------------
- NP-hard combinatorial optimization
- Solution space size: 2^n (n items)
- Binary decision variables
- Classic dynamic programming problem

Author: Group 3
Date: 2024
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from swarm_intelligence.core.base_problem import OptimizationProblem


class Knapsack(OptimizationProblem):
    """
    0/1 Knapsack Problem.
    
    Attributes:
        num_items (int): Number of items
        capacity (float): Knapsack weight capacity
        values (np.ndarray): Item values
        weights (np.ndarray): Item weights
        penalty (float): Penalty for constraint violation
    """
    
    def __init__(
        self,
        num_items: int,
        capacity: float,
        values: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        penalty_multiplier: float = 10.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Knapsack instance.
        
        Args:
            num_items: Number of items
            capacity: Knapsack weight capacity
            values: Item values (optional, random if not provided)
            weights: Item weights (optional, random if not provided)
            penalty_multiplier: Multiplier for constraint violation penalty
            seed: Random seed for generating random items
        """
        super().__init__(num_items, 'discrete', 'Knapsack')
        
        self.num_items = num_items
        self.capacity = capacity
        self.penalty_multiplier = penalty_multiplier
        
        rng = np.random.RandomState(seed)
        
        if values is not None and weights is not None:
            if len(values) != num_items or len(weights) != num_items:
                raise ValueError("Values and weights must have length equal to num_items")
            self.values = np.array(values)
            self.weights = np.array(weights)
        else:
            # Generate random items
            self.values = rng.uniform(1, 100, num_items)
            self.weights = rng.uniform(1, capacity / 2, num_items)  # Ensure some items fit
    
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Calculate total value (negative for minimization).
        
        Applies penalty if weight capacity is exceeded.
        
        Args:
            solution: Binary array indicating selected items (0 or 1)
            
        Returns:
            Negative total value (for minimization) with penalty
        """
        # Binarize solution (threshold at 0.5 for continuous optimizers)
        binary_solution = (solution > 0.5).astype(int)
        
        # Calculate total value and weight
        total_value = np.sum(binary_solution * self.values)
        total_weight = np.sum(binary_solution * self.weights)
        
        # Apply penalty for exceeding capacity
        if total_weight > self.capacity:
            penalty = self.penalty_multiplier * (total_weight - self.capacity)
            return -(total_value - penalty)  # Negative for minimization
        
        return -total_value  # Negative for minimization
    
    def is_feasible(self, solution: np.ndarray) -> bool:
        """
        Check if solution satisfies weight capacity constraint.
        
        Args:
            solution: Binary array indicating selected items
            
        Returns:
            True if feasible, False otherwise
        """
        binary_solution = (solution > 0.5).astype(int)
        total_weight = np.sum(binary_solution * self.weights)
        return total_weight <= self.capacity
    
    def get_solution_info(self, solution: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed information about a solution.
        
        Args:
            solution: Binary array indicating selected items
            
        Returns:
            Dictionary with solution details
        """
        binary_solution = (solution > 0.5).astype(int)
        selected_items = np.where(binary_solution == 1)[0]
        
        total_value = np.sum(binary_solution * self.values)
        total_weight = np.sum(binary_solution * self.weights)
        
        return {
            'selected_items': selected_items.tolist(),
            'num_selected': len(selected_items),
            'total_value': total_value,
            'total_weight': total_weight,
            'capacity': self.capacity,
            'utilization': total_weight / self.capacity * 100,
            'feasible': self.is_feasible(solution)
        }
    
    def get_bounds(self) -> np.ndarray:
        """
        Get problem bounds (binary variables: [0, 1]).
        
        Returns:
            Bounds array [0, 1] for each item
        """
        lower = np.zeros(self.dim)
        upper = np.ones(self.dim)
        return np.array([lower, upper])
    
    def get_optimal_value(self) -> Optional[float]:
        """
        Get optimal value (unknown for random instances).
        
        Can be solved exactly using dynamic programming for small instances.
        
        Returns:
            None (optimal value unknown)
        """
        # For small instances, could implement DP solver
        return None
    
    def get_optimal_solution(self) -> Optional[np.ndarray]:
        """
        Get optimal solution (unknown for random instances).
        
        Returns:
            None (optimal solution unknown)
        """
        return None
    
    def get_problem_info(self) -> Dict[str, Any]:
        """
        Get problem metadata.
        
        Returns:
            Dictionary with problem information
        """
        info = {
            'name': '0/1 Knapsack Problem',
            'type': 'discrete',
            'num_items': self.num_items,
            'capacity': self.capacity,
            'solution_space_size': 2**self.num_items,
            'total_value': np.sum(self.values),
            'total_weight': np.sum(self.weights),
            'avg_value': np.mean(self.values),
            'avg_weight': np.mean(self.weights),
            'value_density': np.mean(self.values / self.weights),
            'characteristics': [
                'NP-hard',
                'Binary variables',
                'Constrained',
                'Classic benchmark'
            ]
        }
        return info
    
    @staticmethod
    def generate_random_instance(
        num_items: int,
        capacity_ratio: float = 0.5,
        seed: Optional[int] = None
    ) -> 'Knapsack':
        """
        Generate random Knapsack instance.
        
        Args:
            num_items: Number of items
            capacity_ratio: Capacity as ratio of total weight (default: 0.5)
            seed: Random seed
            
        Returns:
            Knapsack instance
        """
        rng = np.random.RandomState(seed)
        
        # Generate random values and weights
        values = rng.uniform(1, 100, num_items)
        weights = rng.uniform(1, 50, num_items)
        
        # Set capacity based on ratio
        total_weight = np.sum(weights)
        capacity = capacity_ratio * total_weight
        
        return Knapsack(num_items, capacity, values, weights, seed=seed)
    
    @staticmethod
    def generate_correlated_instance(
        num_items: int,
        correlation: float = 0.8,
        capacity_ratio: float = 0.5,
        seed: Optional[int] = None
    ) -> 'Knapsack':
        """
        Generate Knapsack instance with correlated values and weights.
        
        Higher correlation means heavier items tend to be more valuable.
        
        Args:
            num_items: Number of items
            correlation: Correlation between value and weight (0-1)
            capacity_ratio: Capacity as ratio of total weight
            seed: Random seed
            
        Returns:
            Knapsack instance with correlated items
        """
        rng = np.random.RandomState(seed)
        
        # Generate base weights
        weights = rng.uniform(1, 50, num_items)
        
        # Generate values correlated with weights
        noise = rng.normal(0, 1 - correlation, num_items)
        values = correlation * weights + (1 - correlation) * rng.uniform(1, 100, num_items) + noise
        values = np.maximum(values, 1)  # Ensure positive values
        
        # Set capacity
        total_weight = np.sum(weights)
        capacity = capacity_ratio * total_weight
        
        return Knapsack(num_items, capacity, values, weights, seed=seed)
    
    def solve_dp(self) -> Tuple[float, np.ndarray]:
        """
        Solve using dynamic programming (exact solution for small instances).
        
        Time complexity: O(n·W) where W is capacity
        Space complexity: O(n·W)
        
        Returns:
            Tuple of (optimal_value, optimal_solution)
        """
        n = self.num_items
        W = int(self.capacity)
        
        # DP table: dp[i][w] = max value using first i items with capacity w
        dp = np.zeros((n + 1, W + 1))
        
        # Fill DP table
        for i in range(1, n + 1):
            item_idx = i - 1
            value = self.values[item_idx]
            weight = int(self.weights[item_idx])
            
            for w in range(W + 1):
                # Don't take item
                dp[i][w] = dp[i-1][w]
                
                # Take item if it fits
                if weight <= w:
                    dp[i][w] = max(dp[i][w], dp[i-1][w-weight] + value)
        
        # Backtrack to find selected items
        solution = np.zeros(n)
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                solution[i-1] = 1
                w -= int(self.weights[i-1])
        
        optimal_value = dp[n][W]
        return optimal_value, solution
