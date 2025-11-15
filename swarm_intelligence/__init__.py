"""
Swarm Intelligence Optimization Package.

A comprehensive implementation of swarm intelligence and evolutionary
optimization algorithms with real-time visualization capabilities.

Algorithms Implemented:
    - Particle Swarm Optimization (PSO)
    - Artificial Bee Colony (ABC)
    - Firefly Algorithm (FA)
    - Cuckoo Search (CS)
    - Ant Colony Optimization (ACO)
    - Genetic Algorithm (GA)
    - Hill Climbing (HC)
    - Simulated Annealing (SA)

Problems:
    - Continuous: Sphere, Rastrigin, Rosenbrock, Ackley
    - Discrete: Knapsack, TSP, Graph Coloring

Author: Group 3
Course: Introduction to Artificial Intelligence (CSC14003)
Institution: VNUHCM - University of Science
"""

__version__ = "0.1.0"
__author__ = "Group 3"

# Import core classes for easy access
from swarm_intelligence.core.base_algorithm import BaseOptimizer
from swarm_intelligence.core.base_problem import OptimizationProblem
from swarm_intelligence.core.solution import Solution
from swarm_intelligence.core.population import Population

__all__ = [
    "BaseOptimizer",
    "OptimizationProblem",
    "Solution",
    "Population",
]
