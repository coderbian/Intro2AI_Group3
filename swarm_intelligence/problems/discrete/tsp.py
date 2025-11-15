"""
Traveling Salesman Problem (TSP)

Find the shortest possible route that visits each city exactly once and returns to the origin city.

Mathematical Formulation:
------------------------
Minimize: L = Σ d(city[i], city[i+1]) + d(city[n], city[0])

Where:
- d(i, j): Distance between city i and city j
- n: Number of cities
- city[i]: The i-th city in the tour

Problem Characteristics:
-----------------------
- NP-hard combinatorial optimization
- Solution space size: (n-1)!/2 for symmetric TSP
- Each solution is a permutation of cities
- Commonly used for benchmarking discrete algorithms

Author: Group 3
Date: 2024
"""

import numpy as np
from typing import Optional, Dict, Any
from swarm_intelligence.core.base_problem import OptimizationProblem


class TSP(OptimizationProblem):
    """
    Traveling Salesman Problem.
    
    Attributes:
        num_cities (int): Number of cities
        distance_matrix (np.ndarray): Pairwise distance matrix
        coordinates (np.ndarray): City coordinates (if available)
    """
    
    def __init__(self, num_cities: int, distance_matrix: Optional[np.ndarray] = None, seed: Optional[int] = None):
        """
        Initialize TSP instance.
        
        Args:
            num_cities: Number of cities
            distance_matrix: Pre-defined distance matrix (optional)
            seed: Random seed for generating random cities
        """
        super().__init__(num_cities, 'discrete', 'TSP')
        
        self.num_cities = num_cities
        self.coordinates = None
        
        if distance_matrix is not None:
            # Use provided distance matrix
            if distance_matrix.shape != (num_cities, num_cities):
                raise ValueError(f"Distance matrix shape must be ({num_cities}, {num_cities})")
            self.distance_matrix = distance_matrix
        else:
            # Generate random cities in 2D space
            rng = np.random.RandomState(seed)
            self.coordinates = rng.uniform(0, 100, (num_cities, 2))
            self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute Euclidean distance matrix from coordinates.
        
        Returns:
            Distance matrix (num_cities x num_cities)
        """
        n = self.num_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric
        
        return dist_matrix
    
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Calculate total tour length.
        
        Args:
            solution: Array representing city visit order (permutation)
            
        Returns:
            Total tour length
        """
        tour = solution.astype(int)
        total_distance = 0.0
        
        # Sum distances between consecutive cities
        for i in range(len(tour)):
            city_from = tour[i]
            city_to = tour[(i + 1) % len(tour)]  # Wrap around to start
            total_distance += self.distance_matrix[city_from, city_to]
        
        return total_distance
    
    def get_bounds(self) -> np.ndarray:
        """
        Get problem bounds (not directly applicable for TSP).
        
        Returns:
            Dummy bounds [0, num_cities-1] for each dimension
        """
        lower = np.zeros(self.dim)
        upper = np.ones(self.dim) * (self.num_cities - 1)
        return np.array([lower, upper])
    
    def get_optimal_value(self) -> Optional[float]:
        """
        Get optimal tour length (unknown for random instances).
        
        Returns:
            None (optimal value unknown)
        """
        return None
    
    def get_optimal_solution(self) -> Optional[np.ndarray]:
        """
        Get optimal tour (unknown for random instances).
        
        Returns:
            None (optimal solution unknown)
        """
        return None
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Get distance matrix.
        
        Returns:
            Distance matrix
        """
        return self.distance_matrix
    
    def get_heuristic_matrix(self) -> np.ndarray:
        """
        Get heuristic matrix for ACO (η_ij = 1/d_ij).
        
        Returns:
            Heuristic matrix
        """
        # Avoid division by zero
        heuristic = 1.0 / (self.distance_matrix + 1e-10)
        np.fill_diagonal(heuristic, 0)  # No self-loops
        return heuristic
    
    def get_problem_info(self) -> Dict[str, Any]:
        """
        Get problem metadata.
        
        Returns:
            Dictionary with problem information
        """
        info = {
            'name': 'Traveling Salesman Problem',
            'type': 'discrete',
            'num_cities': self.num_cities,
            'symmetric': True,
            'solution_space_size': f"{self.num_cities-1}!/2",
            'has_coordinates': self.coordinates is not None,
            'characteristics': [
                'NP-hard',
                'Combinatorial',
                'Permutation-based',
                'Classic benchmark'
            ]
        }
        return info
    
    @staticmethod
    def generate_random_instance(num_cities: int, seed: Optional[int] = None) -> 'TSP':
        """
        Generate random TSP instance with cities in 2D plane.
        
        Args:
            num_cities: Number of cities
            seed: Random seed
            
        Returns:
            TSP instance
        """
        return TSP(num_cities, seed=seed)
    
    @staticmethod
    def generate_clustered_instance(num_cities: int, num_clusters: int = 3, seed: Optional[int] = None) -> 'TSP':
        """
        Generate TSP instance with clustered cities.
        
        Args:
            num_cities: Number of cities
            num_clusters: Number of clusters
            seed: Random seed
            
        Returns:
            TSP instance with clustered cities
        """
        rng = np.random.RandomState(seed)
        
        # Generate cluster centers
        cluster_centers = rng.uniform(0, 100, (num_clusters, 2))
        
        # Assign cities to clusters
        coordinates = []
        cities_per_cluster = num_cities // num_clusters
        
        for i in range(num_clusters):
            # Generate cities around cluster center
            cluster_cities = cluster_centers[i] + rng.normal(0, 5, (cities_per_cluster, 2))
            coordinates.extend(cluster_cities)
        
        # Add remaining cities
        remaining = num_cities - len(coordinates)
        if remaining > 0:
            extra_cities = cluster_centers[0] + rng.normal(0, 5, (remaining, 2))
            coordinates.extend(extra_cities)
        
        coordinates = np.array(coordinates)
        
        # Create TSP instance with custom coordinates
        tsp = TSP(num_cities, seed=seed)
        tsp.coordinates = coordinates
        tsp.distance_matrix = tsp._compute_distance_matrix()
        
        return tsp
