# AI Swarm Intelligence Optimization Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.20+-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of swarm intelligence and evolutionary optimization algorithms with real-time visualization capabilities for the Introduction to AI course (CSC14003) at VNUHCM - University of Science.

## 🎯 Overview

This project implements **7 optimization algorithms** with professional OOP design, comprehensive documentation, and benchmarking capabilities.

### Implemented Algorithms

#### Swarm Intelligence
- **PSO** (Particle Swarm Optimization) - Population-based optimizer inspired by bird flocking
- **ABC** (Artificial Bee Colony) - Mimics foraging behavior of honey bees
- **FA** (Firefly Algorithm) - Based on flashing patterns of fireflies
- **CS** (Cuckoo Search) - Uses Lévy flights for exploration

#### Evolutionary
- **GA** (Genetic Algorithm) - Selection, crossover, and mutation operators

#### Local Search
- **Hill Climbing** - Iterative improvement with neighborhood search
- **Simulated Annealing** - Temperature-based acceptance of worse solutions

### Benchmark Problems

#### Continuous Functions
- **Sphere** - Unimodal, convex (easy baseline)
- **Rastrigin** - Highly multimodal with many local minima (hard)
- **Rosenbrock** - Valley-shaped, difficult convergence (medium-hard)
- **Ackley** - Multimodal with flat outer region (medium)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/coderbian/Intro2AI_Group3.git
cd Intro2AI_Group3

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from swarm_intelligence.algorithms.swarm import PSO
from swarm_intelligence.problems.continuous import Rastrigin

# Create problem
problem = Rastrigin(dim=10)

# Create optimizer
optimizer = PSO(
    objective_func=problem.evaluate,
    dim=problem.dim,
    bounds=problem.get_bounds(),
    pop_size=50,
    max_iter=1000,
    w=0.7,
    c1=1.5,
    c2=1.5,
    seed=42
)

# Run optimization
result = optimizer.optimize(verbose=True)

# Access results
print(f"Best fitness: {result['best_fitness']:.6f}")
print(f"Best solution: {result['best_solution']}")
print(f"Time: {result['time']:.2f}s")
```

### Testing All Algorithms

```bash
# Run comprehensive test suite
python demo/test_all_algorithms.py

# Test single algorithm
python demo/test_pso.py
```

## 📊 Performance Results

Based on our tests (200 iterations, seed=42):

| Algorithm | Sphere (10D) | Rastrigin (10D) | Speed |
|-----------|--------------|-----------------|-------|
| **PSO** | 0.000000 ✓ | 4.975 | Fast |
| **ABC** | 0.000004 ✓ | 0.000208 ✓ | Slow |
| **FA** | 15630.9 | 78.855 | Slow |
| **CS** | 15630.9 | 60.569 | Medium |
| **GA** | 0.074 ✓ | 0.306 ✓ | Medium |
| **HC** | 351.5 | 64.942 | Fast |
| **SA** | 9352.3 | 100.994 | Fast |

✓ = Error < 0.1 (Good performance)

**Key Findings:**
- **PSO and ABC**: Best performers on both problems
- **GA**: Consistent and reliable
- **Local search** (HC, SA): Fast but needs parameter tuning
- **FA and CS**: Need longer iterations or parameter adjustment

## 🏗️ Project Structure

```
Intro2AI_Group3/
├── swarm_intelligence/          # Main package
│   ├── core/                    # Base classes
│   │   ├── base_algorithm.py    # BaseOptimizer (Template Method)
│   │   ├── base_problem.py      # OptimizationProblem
│   │   ├── solution.py          # Solution representation
│   │   └── population.py        # Population management
│   ├── algorithms/              # Algorithm implementations
│   │   ├── swarm/              # PSO, ABC, FA, CS
│   │   ├── evolutionary/       # GA
│   │   └── local_search/       # HC, SA
│   ├── problems/                # Benchmark problems
│   │   ├── continuous/         # Sphere, Rastrigin, Rosenbrock, Ackley
│   │   └── discrete/           # (Future: Knapsack, TSP, Graph Coloring)
│   ├── visualization/           # (Planned)
│   ├── benchmarks/             # (Planned)
│   └── utils/                  # (Planned)
├── notebooks/                   # Jupyter notebooks (Planned)
├── experiments/                 # Experiment configs (Planned)
├── tests/                      # Unit tests (Planned)
├── demo/                       # Demo scripts
│   ├── test_pso.py
│   └── test_all_algorithms.py
├── results/                    # Experiment results
├── requirements.txt
├── setup.py
└── README.md
```

## 🔬 Algorithm Details

### PSO (Particle Swarm Optimization)

**Equations:**
```
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

**Parameters:**
- `w=0.7`: Inertia weight
- `c1=1.5`: Cognitive coefficient
- `c2=1.5`: Social coefficient

### ABC (Artificial Bee Colony)

**Phases:**
1. Employed Bee Phase - Local search around food sources
2. Onlooker Bee Phase - Probabilistic selection
3. Scout Bee Phase - Abandon exhausted sources

**Parameters:**
- `limit=100`: Abandonment limit

### GA (Genetic Algorithm)

**Operators:**
- Tournament selection
- Arithmetic crossover
- Gaussian mutation
- Elitism

**Parameters:**
- `crossover_rate=0.8`
- `mutation_rate=0.1`

## 📖 API Documentation

### Creating a Custom Problem

```python
from swarm_intelligence.core.base_problem import OptimizationProblem
import numpy as np

class MyProblem(OptimizationProblem):
    def __init__(self, dim):
        super().__init__(dim, 'continuous', 'MyProblem')
    
    def evaluate(self, x):
        return np.sum(x**2) + np.prod(np.sin(x))
    
    def get_bounds(self):
        return np.array([[-10]*self.dim, [10]*self.dim])
    
    def get_optimal_value(self):
        return None  # Unknown
    
    def get_optimal_solution(self):
        return None  # Unknown
```

### Creating a Custom Algorithm

```python
from swarm_intelligence.core.base_algorithm import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def initialize_population(self):
        # Initialize your population
        self.population = self.rng.uniform(
            self.bounds[0], self.bounds[1], 
            (self.pop_size, self.dim))
    
    def update_population(self):
        # Your algorithm logic here
        # Update self.best_solution and self.best_fitness
        pass
    
    def get_algorithm_name(self):
        return "MyOptimizer"
```

## 🎓 Course Information

- **Course**: Introduction to Artificial Intelligence (CSC14003)
- **Institution**: VNUHCM - University of Science
- **Project**: Swarm Intelligence Algorithms
- **Group**: Group 3

## 📝 Features

- ✅ Professional OOP design with Template Method pattern
- ✅ Type hints throughout codebase
- ✅ Google-style docstrings
- ✅ 7 optimization algorithms implemented
- ✅ 4 benchmark problems
- ✅ Comprehensive testing suite
- ✅ Reproducible results (seed support)
- ⏳ Real-time visualization (Planned)
- ⏳ Statistical analysis tools (Planned)
- ⏳ Jupyter notebooks (Planned)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Course instructors and TAs
- Original papers by Kennedy & Eberhart (PSO), Karaboga (ABC), Yang (FA, CS)
- Holland (GA), Kirkpatrick (SA)

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project for the Introduction to AI course. Some features are still in development (marked with ⏳).