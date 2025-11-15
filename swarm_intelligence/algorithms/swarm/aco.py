"""
Ant Colony Optimization (ACO) Algorithm

Inspired by the foraging behavior of ants. Ants deposit pheromones on paths they traverse,
and other ants are more likely to follow paths with stronger pheromone trails. This creates
a positive feedback loop that helps the colony find optimal paths.

Mathematical Formulation:
-----------------------
Pheromone Update:
    τ_ij(t+1) = (1-ρ)·τ_ij(t) + Δτ_ij

    where:
    - ρ: Evaporation rate (0 < ρ < 1)
    - Δτ_ij: Sum of pheromone deposited by all ants

Pheromone Deposit:
    Δτ_ij^k = Q/L_k if edge (i,j) used by ant k
            = 0 otherwise

    where:
    - Q: Pheromone deposit constant
    - L_k: Tour length of ant k

Probability of Selecting Edge:
    p_ij^k = (τ_ij^α · η_ij^β) / Σ(τ_il^α · η_il^β)

    where:
    - α: Pheromone importance
    - β: Heuristic importance
    - η_ij: Heuristic information (typically 1/distance)

Parameters:
----------
- num_ants: Number of ants in the colony
- alpha: Pheromone importance factor (typically 1.0)
- beta: Heuristic importance factor (typically 2.0)
- rho: Pheromone evaporation rate (typically 0.1)
- Q: Pheromone deposit constant (typically 1.0)

Author: Group 3
Date: 2024
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
from swarm_intelligence.core.base_algorithm import BaseOptimizer


class ACO(BaseOptimizer):
    """
    Ant Colony Optimization Algorithm.
    
    Designed for discrete optimization problems like TSP, Graph Coloring, and Knapsack.
    Uses pheromone trails and heuristic information to construct solutions.
    
    Attributes:
        num_ants (int): Number of ants in the colony
        alpha (float): Pheromone importance (default: 1.0)
        beta (float): Heuristic importance (default: 2.0)
        rho (float): Evaporation rate (default: 0.1)
        Q (float): Pheromone deposit constant (default: 1.0)
        pheromone (np.ndarray): Pheromone matrix
        heuristic (np.ndarray): Heuristic information matrix
        best_tour (list): Best solution found (for TSP-like problems)
    """
    
    def __init__(
        self,
        objective_func: Callable,
        dim: int,
        bounds: np.ndarray,
        pop_size: int = 30,
        max_iter: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        Q: float = 1.0,
        heuristic_func: Optional[Callable] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize ACO optimizer.
        
        Args:
            objective_func: Objective function to minimize
            dim: Problem dimension (number of cities for TSP)
            bounds: Search space bounds [lower, upper] (not used directly in ACO)
            pop_size: Number of ants (default: 30)
            max_iter: Maximum iterations (default: 100)
            alpha: Pheromone importance (default: 1.0)
            beta: Heuristic importance (default: 2.0)
            rho: Evaporation rate (default: 0.1)
            Q: Pheromone deposit constant (default: 1.0)
            heuristic_func: Function to compute heuristic matrix (optional)
            seed: Random seed for reproducibility
        """
        super().__init__(objective_func, dim, bounds, pop_size, max_iter, seed)
        
        self.num_ants = pop_size
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.heuristic_func = heuristic_func
        
        # Pheromone matrix (dim x dim)
        self.pheromone = None
        
        # Heuristic information matrix (dim x dim)
        self.heuristic = None
        
        # Best tour found
        self.best_tour = None
        
    def initialize_population(self):
        """
        Initialize pheromone and heuristic matrices.
        
        For TSP-like problems:
        - Pheromone matrix: τ_ij initialized to small constant
        - Heuristic matrix: η_ij = 1/distance_ij
        """
        # Initialize pheromone matrix with small constant
        initial_pheromone = 1.0 / self.dim
        self.pheromone = np.ones((self.dim, self.dim)) * initial_pheromone
        np.fill_diagonal(self.pheromone, 0)  # No self-loops
        
        # Initialize heuristic matrix
        if self.heuristic_func is not None:
            self.heuristic = self.heuristic_func()
        else:
            # Default: uniform heuristic (all edges equal)
            self.heuristic = np.ones((self.dim, self.dim))
            np.fill_diagonal(self.heuristic, 0)
        
        # Avoid division by zero
        self.heuristic = np.maximum(self.heuristic, 1e-10)
        
        # Initialize best solution (dummy for now)
        self.best_solution = np.arange(self.dim)
        self.best_fitness = self.evaluate(self.best_solution)
        self.best_tour = list(self.best_solution)
        
    def _construct_solution(self, ant_id: int) -> Tuple[np.ndarray, float]:
        """
        Construct a solution using probabilistic selection based on pheromone and heuristic.
        
        For TSP: Build a tour by selecting cities probabilistically.
        
        Args:
            ant_id: Identifier for the current ant
            
        Returns:
            Tuple of (solution, fitness)
        """
        # Start from random city
        unvisited = list(range(self.dim))
        current = self.rng.choice(unvisited)
        tour = [current]
        unvisited.remove(current)
        
        # Build tour
        while unvisited:
            # Calculate probabilities for unvisited cities
            probabilities = []
            for city in unvisited:
                pheromone_val = self.pheromone[current, city] ** self.alpha
                heuristic_val = self.heuristic[current, city] ** self.beta
                probabilities.append(pheromone_val * heuristic_val)
            
            # Normalize probabilities
            probabilities = np.array(probabilities)
            prob_sum = probabilities.sum()
            
            if prob_sum > 0:
                probabilities /= prob_sum
            else:
                # If all probabilities are zero, use uniform distribution
                probabilities = np.ones(len(unvisited)) / len(unvisited)
            
            # Select next city
            next_city_idx = self.rng.choice(len(unvisited), p=probabilities)
            next_city = unvisited[next_city_idx]
            
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        # Evaluate tour
        solution = np.array(tour)
        fitness = self.evaluate(solution)
        
        return solution, fitness
    
    def _update_pheromone(self, all_tours: list, all_fitness: list):
        """
        Update pheromone matrix based on ant solutions.
        
        Two-step process:
        1. Evaporation: τ_ij ← (1-ρ)·τ_ij
        2. Deposit: τ_ij ← τ_ij + Σ Δτ_ij^k
        
        Args:
            all_tours: List of tours constructed by ants
            all_fitness: List of fitness values for each tour
        """
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Deposit pheromone
        for tour, fitness in zip(all_tours, all_fitness):
            # Better fitness = more pheromone (for minimization)
            # Δτ = Q / fitness (smaller fitness = more pheromone)
            if fitness > 0:
                deposit = self.Q / fitness
            else:
                deposit = self.Q  # Avoid division by zero
            
            # Deposit on edges in tour
            for i in range(len(tour)):
                city_from = tour[i]
                city_to = tour[(i + 1) % len(tour)]  # Wrap around for closed tour
                self.pheromone[city_from, city_to] += deposit
                self.pheromone[city_to, city_from] += deposit  # Symmetric
        
        # Optional: Apply pheromone limits to prevent stagnation
        max_pheromone = 1.0 / self.rho
        min_pheromone = max_pheromone / (2 * self.dim)
        self.pheromone = np.clip(self.pheromone, min_pheromone, max_pheromone)
    
    def update_population(self):
        """
        One iteration of ACO:
        1. Each ant constructs a solution
        2. Update global best
        3. Update pheromone matrix
        """
        all_tours = []
        all_fitness = []
        
        # Each ant constructs a solution
        for ant_id in range(self.num_ants):
            tour, fitness = self._construct_solution(ant_id)
            all_tours.append(tour)
            all_fitness.append(fitness)
            
            # Update global best
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = tour.copy()
                self.best_tour = list(tour)
        
        # Update pheromone trails
        self._update_pheromone(all_tours, all_fitness)
    
    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "ACO"
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get algorithm configuration.
        
        Returns:
            Dictionary with algorithm parameters
        """
        config = super().get_config()
        config.update({
            'num_ants': self.num_ants,
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'Q': self.Q
        })
        return config
