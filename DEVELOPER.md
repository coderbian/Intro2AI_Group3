# GitHub Copilot Instructions - AI Swarm Intelligence Project

## Project Overview

This is a complete rewrite of a Swarm Intelligence optimization project for an Introduction to AI course. We are building a professional, OOP-based Python package with real-time visualization capabilities.

## Core Requirements

### Technology Stack

- **Python 3.8+**
- **NumPy**: Only allowed library for algorithm implementation
- **Matplotlib**: For visualization
- **Jupyter**: For interactive notebooks
- **Type hints**: Use throughout the codebase
- **Docstrings**: Google style for all classes and methods

### Project Structure

```
AI_Project_SwarmIntelligence/
├── swarm_intelligence/              # Main package
│   ├── __init__.py
│   ├── core/                        # Base classes
│   │   ├── __init__.py
│   │   ├── base_algorithm.py
│   │   ├── base_problem.py
│   │   ├── solution.py
│   │   └── population.py
│   ├── algorithms/                  # Algorithm implementations
│   │   ├── __init__.py
│   │   ├── swarm/
│   │   │   ├── __init__.py
│   │   │   ├── pso.py
│   │   │   ├── abc.py
│   │   │   ├── fa.py
│   │   │   ├── cs.py
│   │   │   └── aco.py
│   │   ├── evolutionary/
│   │   │   ├── __init__.py
│   │   │   └── ga.py
│   │   └── local_search/
│   │       ├── __init__.py
│   │       ├── hc.py
│   │       └── sa.py
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── continuous/
│   │   │   ├── __init__.py
│   │   │   ├── sphere.py
│   │   │   ├── rastrigin.py
│   │   │   ├── rosenbrock.py
│   │   │   └── ackley.py
│   │   └── discrete/
│   │       ├── __init__.py
│   │       ├── knapsack.py
│   │       ├── tsp.py
│   │       └── graph_coloring.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── realtime_plotter.py
│   │   ├── convergence_plot.py
│   │   ├── landscape_plot.py
│   │   ├── swarm_animation.py
│   │   └── comparison_plot.py
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── config.py
│       └── data_export.py
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_algorithms_demo.ipynb
│   ├── 03_continuous_problems.ipynb
│   ├── 04_discrete_problems.ipynb
│   ├── 05_comparison.ipynb
│   ├── 06_visualization.ipynb
│   └── 07_custom_problems.ipynb
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py
│   ├── configs/
│   │   ├── rastrigin_config.yaml
│   │   └── knapsack_config.yaml
│   └── batch_run.py
├── tests/
│   ├── __init__.py
│   ├── test_algorithms/
│   ├── test_problems/
│   └── test_visualization/
├── results/
│   ├── continuous/
│   └── discrete/
├── docs/
├── demo/
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Implementation Guidelines

### 1. Base Classes (`swarm_intelligence/core/`)

#### `base_algorithm.py`

Create an abstract base class `BaseOptimizer` with:

**Attributes:**

- `objective_func`: Callable objective function
- `dim`: Problem dimensionality
- `bounds`: np.ndarray of shape (2, dim) for [lower, upper] bounds
- `pop_size`: Population size
- `max_iter`: Maximum iterations
- `seed`: Random seed for reproducibility
- `visualizer`: Optional RealtimeVisualizer instance
- `best_solution`: Current best solution
- `best_fitness`: Current best fitness value
- `fitness_history`: List of best fitness per iteration
- `population`: Current population (np.ndarray)
- `iteration`: Current iteration counter

**Abstract Methods:**

- `initialize_population()`: Initialize algorithm-specific population
- `update_population()`: Perform one iteration of the algorithm
- `get_algorithm_name()`: Return string name of algorithm

**Concrete Methods:**

- `optimize(verbose=True) -> dict`: Main optimization loop
  - Initialize population
  - Loop through iterations
  - Call update_population()
  - Track best solution
  - Update visualizer if provided
  - Return results dict with keys: best_solution, best_fitness, fitness_history, iterations
- `evaluate(solution)`: Evaluate objective function
- `clip_bounds(solution)`: Clip solution to valid bounds
- `save_results(filepath)`: Save results to JSON

**Design Pattern:**

- Use Template Method pattern
- Support both minimization and maximization (default: minimization)
- Thread-safe if possible

#### `base_problem.py`

Create abstract class `OptimizationProblem` with:

**Attributes:**

- `dim`: Problem dimensionality
- `problem_type`: 'continuous' or 'discrete'
- `name`: Problem name

**Abstract Methods:**

- `evaluate(x: np.ndarray) -> float`: Evaluate solution
- `get_bounds() -> np.ndarray`: Return bounds array shape (2, dim)
- `get_optimal_value() -> float`: Return known optimal (or None)
- `get_optimal_solution() -> np.ndarray`: Return optimal solution (or None)

**Concrete Methods:**

- `is_feasible(x)`: Check solution feasibility
- `batch_evaluate(population)`: Vectorized evaluation
- `get_problem_info() -> dict`: Return problem metadata

#### `solution.py`

Create `Solution` class to represent a candidate solution:

**Attributes:**

- `position`: np.ndarray of decision variables
- `fitness`: float fitness value
- `velocity`: np.ndarray (for PSO-like algorithms, optional)
- `metadata`: dict for algorithm-specific data

**Methods:**

- `copy()`: Deep copy of solution
- `distance_to(other)`: Calculate distance to another solution
- `__lt__`, `__le__`, `__gt__`, `__ge__`: Comparison operators
- `to_dict()`: Serialize to dictionary

#### `population.py`

Create `Population` class to manage a collection of solutions:

**Attributes:**

- `solutions`: List[Solution]
- `size`: int population size

**Methods:**

- `add(solution)`: Add solution to population
- `get_best()`: Return best solution
- `get_worst()`: Return worst solution
- `get_diversity()`: Calculate population diversity
- `sort()`: Sort by fitness
- `replace(index, solution)`: Replace solution at index
- `__iter__`, `__len__`, `__getitem__`: Make iterable

---

### 2. Algorithm Implementations

Each algorithm should:

1. Inherit from `BaseOptimizer`
2. Implement all abstract methods
3. Add algorithm-specific parameters in `__init__`
4. Use type hints for all parameters
5. Include comprehensive docstrings with mathematical formulation
6. Handle both continuous and discrete problems (if applicable)

#### PSO (Particle Swarm Optimization) - `algorithms/swarm/pso.py`

**Additional Parameters:**

- `w`: Inertia weight (default: 0.7)
- `c1`: Cognitive coefficient (default: 1.5)
- `c2`: Social coefficient (default: 1.5)
- `v_max`: Maximum velocity (default: None, auto-compute)

**Key Attributes:**

- `velocities`: np.ndarray of particle velocities
- `personal_best_positions`: np.ndarray
- `personal_best_fitness`: np.ndarray
- `global_best_position`: np.ndarray
- `global_best_fitness`: float

**Algorithm Steps:**

1. Initialize particles and velocities randomly within bounds
2. For each iteration:
   - Update velocities using PSO formula
   - Update positions
   - Evaluate fitness
   - Update personal bests
   - Update global best
   - Clip to bounds

**Docstring should include:**

```
PSO updates velocities and positions using:
v_i(t+1) = w*v_i(t) + c1*r1*(p_i - x_i(t)) + c2*r2*(g - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)

References:
- Kennedy & Eberhart (1995). Particle Swarm Optimization.
```

#### ABC (Artificial Bee Colony) - `algorithms/swarm/abc.py`

**Additional Parameters:**

- `limit`: Abandonment limit (default: 100)
- `n_employed`: Number of employed bees (default: pop_size // 2)
- `n_onlooker`: Number of onlooker bees (default: pop_size // 2)

**Key Attributes:**

- `food_sources`: np.ndarray of food source positions
- `fitness_values`: np.ndarray
- `trial_counters`: np.ndarray tracking unsuccessful trials
- `probabilities`: np.ndarray for onlooker selection

**Algorithm Phases:**

1. Employed bee phase: Local search around food sources
2. Onlooker bee phase: Probabilistic selection and search
3. Scout bee phase: Abandon exhausted sources

#### FA (Firefly Algorithm) - `algorithms/swarm/fa.py`

**Additional Parameters:**

- `alpha`: Randomization parameter (default: 0.5)
- `beta0`: Attractiveness at r=0 (default: 1.0)
- `gamma`: Light absorption coefficient (default: 1.0)

**Key Attributes:**

- `fireflies`: np.ndarray of firefly positions
- `intensities`: np.ndarray of light intensities (fitness)

**Movement Formula:**

```
x_i = x_i + beta0 * exp(-gamma * r_ij^2) * (x_j - x_i) + alpha * (rand - 0.5)
where r_ij is distance between firefly i and j
```

#### CS (Cuckoo Search) - `algorithms/swarm/cs.py`

**Additional Parameters:**

- `pa`: Probability of abandoning worst nests (default: 0.25)
- `beta`: Levy flight parameter (default: 1.5)

**Key Methods:**

- `levy_flight()`: Generate Levy flight step
- `get_cuckoo()`: Generate new solution via Levy flight
- `abandon_worst_nests()`: Replace worst solutions

#### ACO (Ant Colony Optimization) - `algorithms/swarm/aco.py`

**Note:** Specifically for discrete problems (TSP, Knapsack)

**Additional Parameters:**

- `alpha`: Pheromone importance (default: 1.0)
- `beta`: Heuristic importance (default: 2.0)
- `rho`: Evaporation rate (default: 0.5)
- `q0`: Exploitation vs exploration (default: 0.9)

**Key Attributes:**

- `pheromone`: Pheromone matrix
- `heuristic`: Heuristic information matrix

#### GA (Genetic Algorithm) - `algorithms/evolutionary/ga.py`

**Additional Parameters:**

- `crossover_rate`: Probability of crossover (default: 0.8)
- `mutation_rate`: Probability of mutation (default: 0.1)
- `selection_method`: 'tournament', 'roulette', 'rank' (default: 'tournament')
- `tournament_size`: For tournament selection (default: 3)

**Key Methods:**

- `selection()`: Select parents
- `crossover()`: Perform crossover
- `mutation()`: Perform mutation
- `create_offspring()`: Generate new population

#### HC (Hill Climbing) - `algorithms/local_search/hc.py`

**Additional Parameters:**

- `step_size`: Size of neighborhood search (default: 0.1)
- `n_neighbors`: Number of neighbors to generate (default: 8)

**Variants to implement:**

- Steepest ascent
- First improvement
- Random restart

#### SA (Simulated Annealing) - `algorithms/local_search/sa.py`

**Additional Parameters:**

- `initial_temp`: Starting temperature (default: 100.0)
- `cooling_rate`: Temperature reduction rate (default: 0.95)
- `min_temp`: Minimum temperature (default: 0.01)

**Key Methods:**

- `acceptance_probability()`: Calculate acceptance probability
- `cool_down()`: Reduce temperature
- `generate_neighbor()`: Generate neighbor solution

---

### 3. Problem Implementations (`swarm_intelligence/problems/`)

Each problem class should:

1. Inherit from `OptimizationProblem`
2. Implement all abstract methods
3. Provide mathematical formula in docstring
4. Include visualization-friendly metadata

#### Continuous Problems

**Sphere Function** (`continuous/sphere.py`)

```
f(x) = sum(x_i^2)
Domain: [-100, 100]^n
Global minimum: f(0,...,0) = 0
```

**Rastrigin Function** (`continuous/rastrigin.py`)

```
f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
Domain: [-5.12, 5.12]^n
Global minimum: f(0,...,0) = 0
Characteristics: Highly multimodal
```

**Rosenbrock Function** (`continuous/rosenbrock.py`)

```
f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
Domain: [-5, 10]^n
Global minimum: f(1,...,1) = 0
Characteristics: Valley-shaped, difficult to optimize
```

**Ackley Function** (`continuous/ackley.py`)

```
f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e
Domain: [-32.768, 32.768]^n
Global minimum: f(0,...,0) = 0
Characteristics: Many local minima
```

#### Discrete Problems

**Knapsack Problem** (`discrete/knapsack.py`)

**Class Structure:**

```python
class KnapsackProblem(OptimizationProblem):
    def __init__(self, weights, values, capacity):
        # weights: list of item weights
        # values: list of item values
        # capacity: maximum capacity
        
    def evaluate(self, x):
        # x is binary array [0,1,0,1,...]
        # Return negative value (for minimization) if infeasible
        # Return sum of values if feasible
        
    def is_feasible(self, x):
        # Check if total weight <= capacity
        
    @staticmethod
    def generate_random_instance(n_items, seed=None):
        # Generate random weights and values
```

**TSP (Traveling Salesman Problem)** (`discrete/tsp.py`)

**Class Structure:**

```python
class TSPProblem(OptimizationProblem):
    def __init__(self, distance_matrix):
        # distance_matrix: n x n matrix of distances
        
    def evaluate(self, tour):
        # tour: permutation of cities [0,1,2,...,n-1]
        # Return total tour distance
        
    @staticmethod
    def generate_random_instance(n_cities, seed=None):
        # Generate random city coordinates
        # Compute Euclidean distance matrix
```

**Graph Coloring** (`discrete/graph_coloring.py`)

**Class Structure:**

```python
class GraphColoringProblem(OptimizationProblem):
    def __init__(self, adjacency_matrix, n_colors):
        # adjacency_matrix: n x n matrix
        # n_colors: number of available colors
        
    def evaluate(self, coloring):
        # coloring: array of color assignments
        # Return number of conflicts
        
    @staticmethod
    def generate_random_graph(n_nodes, edge_prob, seed=None):
        # Generate random graph
```

---

### 4. Visualization System (`swarm_intelligence/visualization/`)

#### `realtime_plotter.py`

**Class: RealtimeVisualizer**

```python
class RealtimeVisualizer:
    """Real-time visualization during optimization."""
    
    def __init__(self, problem, mode='2d', update_interval=10):
        """
        Args:
            problem: OptimizationProblem instance
            mode: '2d', '3d', or 'simple'
            update_interval: Update plot every N iterations
        """
        
    def setup_figure(self):
        # Create matplotlib figure with subplots
        # For 2D continuous problems:
        #   - Left: Contour plot with particles
        #   - Right: Convergence curve
        # For discrete problems:
        #   - Convergence curve only
        
    def update(self, iteration, population, best_fitness):
        # Update particle positions
        # Update convergence curve
        # Refresh display
        
    def finalize(self, result):
        # Final plot with statistics
        
    def save_animation(self, filepath):
        # Save as GIF or MP4
```

**Design Requirements:**

- Non-blocking updates (use `plt.pause()`)
- Handle matplotlib backend issues
- Efficient redrawing (use `set_data()` instead of replotting)
- Support both Jupyter and standalone scripts

#### `convergence_plot.py`

```python
def plot_convergence_curves(results_dict, title, save_path=None):
    """
    Plot convergence curves for multiple algorithms.
    
    Args:
        results_dict: {algorithm_name: fitness_history}
        title: Plot title
        save_path: Path to save figure
    """
    # Create line plot
    # Different color for each algorithm
    # Log scale option for fitness axis
    # Show mean ± std if multiple runs provided
```

#### `landscape_plot.py`

```python
def plot_3d_surface(problem, ranges=None, resolution=100):
    """Plot 3D surface of objective function."""
    # Create meshgrid
    # Evaluate function
    # Plot surface with matplotlib 3D
    
def plot_contour_with_path(problem, algorithm_history):
    """Plot contour with optimization path."""
    # Contour plot of function
    # Overlay best solution trajectory
    # Mark global optimum if known
```

#### `swarm_animation.py`

```python
class SwarmAnimator:
    """Create animated visualization of swarm behavior."""
    
    def __init__(self, problem, algorithm):
        # Setup animation framework
        
    def animate(self, save_path=None):
        # Use matplotlib.animation.FuncAnimation
        # Show particle movement over iterations
        
    def save_video(self, filepath, fps=30):
        # Save as MP4 using ffmpeg
```

#### `comparison_plot.py`

```python
def plot_algorithm_comparison(results, metrics=['best_fitness', 'time']):
    """
    Create comparison plots for multiple algorithms.
    
    Plots:
    - Box plots for robustness
    - Bar charts for average performance
    - Heatmap for parameter sensitivity
    """
    
def plot_statistical_test_results(test_results):
    """Visualize statistical test results (t-test, ANOVA)."""
```

---

### 5. Benchmarking Framework (`swarm_intelligence/benchmarks/`)

#### `runner.py`

```python
class ExperimentRunner:
    """Run and manage optimization experiments."""
    
    def __init__(self, algorithms, problems, n_runs=20):
        self.algorithms = algorithms
        self.problems = problems
        self.n_runs = n_runs
        self.results = {}
        
    def run_experiment(self, algorithm, problem, seed):
        # Run single experiment
        # Return result dict
        
    def run_all(self, parallel=False, n_jobs=-1):
        # Run all combinations
        # Use multiprocessing if parallel=True
        
    def save_results(self, filepath):
        # Save to CSV/JSON
        
    def load_results(self, filepath):
        # Load previous results
```

#### `metrics.py`

```python
def calculate_convergence_speed(fitness_history, threshold=0.01):
    """Calculate iteration to reach threshold of optimal."""
    
def calculate_success_rate(results, tolerance=1e-6):
    """Calculate percentage of runs reaching optimal."""
    
def calculate_diversity(population):
    """Calculate population diversity metric."""
    
def calculate_time_complexity(algorithm_name, problem_size):
    """Estimate time complexity."""
    
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    @staticmethod
    def convergence_rate(fitness_history):
        # Average fitness improvement per iteration
        
    @staticmethod
    def final_error(best_fitness, optimal_value):
        # Absolute/relative error
        
    @staticmethod
    def stability(results_list):
        # Standard deviation across runs
        
    @staticmethod
    def computational_cost(time_taken, evaluations):
        # Time per evaluation
```

#### `statistical_tests.py`

```python
def wilcoxon_test(results1, results2):
    """Perform Wilcoxon signed-rank test."""
    
def friedman_test(results_dict):
    """Perform Friedman test for multiple algorithms."""
    
def generate_comparison_table(results_dict):
    """Generate LaTeX/Markdown comparison table."""
```

---

### 6. Utilities (`swarm_intelligence/utils/`)

#### `logger.py`

```python
class OptimizationLogger:
    """Custom logger for optimization process."""
    
    def __init__(self, log_file=None, level='INFO'):
        # Setup logging
        
    def log_iteration(self, iteration, best_fitness, population_diversity):
        # Log iteration details
        
    def log_result(self, algorithm_name, problem_name, result):
        # Log final result
```

#### `config.py`

```python
class Config:
    """Configuration management using YAML/dict."""
    
    @staticmethod
    def load_from_yaml(filepath):
        # Load configuration
        
    @staticmethod
    def get_algorithm_config(algorithm_name):
        # Get default parameters for algorithm
        
    @staticmethod
    def get_problem_config(problem_name):
        # Get problem configuration
```

#### `data_export.py`

```python
def export_to_csv(results, filepath):
    """Export results to CSV."""
    
def export_to_latex(results, filepath):
    """Export results as LaTeX table."""
    
def export_to_json(results, filepath):
    """Export results to JSON."""
    
def generate_report(results, output_dir):
    """Generate comprehensive HTML/PDF report."""
```

---

### 7. Jupyter Notebooks (`notebooks/`)

Each notebook should:

1. Have clear markdown explanations
2. Show code execution step-by-step
3. Include visualizations
4. Be executable from top to bottom
5. Have a summary section

#### `01_introduction.ipynb`

**Content:**

- Project overview
- Installation instructions
- Quick start example
- Package structure explanation

#### `02_algorithms_demo.ipynb`

**Content:**

- Demo each of the 8 algorithms
- Show algorithm parameters
- Visualize behavior on simple 2D function
- Compare convergence patterns

#### `03_continuous_problems.ipynb`

**Content:**

- Test algorithms on Sphere, Rastrigin, Rosenbrock, Ackley
- 3D surface visualizations
- Convergence analysis
- Statistical comparison

#### `04_discrete_problems.ipynb`

**Content:**

- Knapsack problem experiments
- TSP experiments
- Graph coloring experiments
- Problem-specific visualizations

#### `05_comparison.ipynb`

**Content:**

- Comprehensive algorithm comparison
- Performance metrics
- Statistical tests
- Recommendation matrix

#### `06_visualization.ipynb`

**Content:**

- Showcase all visualization capabilities
- Real-time animation examples
- Custom plot creation
- Export options

#### `07_custom_problems.ipynb`

**Content:**

- Tutorial: Create custom optimization problem
- Tutorial: Implement custom algorithm variant
- Advanced customization examples

---

### 8. Experiments (`experiments/`)

#### `run_experiments.py`

```python
def main():
    """Main experiment script."""
    # Parse command line arguments
    # Load configuration
    # Initialize algorithms and problems
    # Run experiments
    # Save results
    # Generate plots
    
if __name__ == '__main__':
    main()
```

**Command line interface:**

```bash
python experiments/run_experiments.py \
    --config configs/rastrigin_config.yaml \
    --output results/continuous/ \
    --parallel \
    --n-runs 20
```

#### Configuration Files (YAML format)

**Example: `configs/rastrigin_config.yaml`**

```yaml
problem:
  name: rastrigin
  dimensions: [10, 30]
  
algorithms:
  pso:
    pop_size: 50
    max_iter: 1000
    w: 0.7
    c1: 1.5
    c2: 1.5
    
  abc:
    pop_size: 50
    max_iter: 1000
    limit: 100
    
  # ... other algorithms

experiment:
  n_runs: 20
  seeds: [42, 43, 44, ...]  # Auto-generate if not provided
  visualize: true
  save_animations: false
```

---

## Coding Standards

### 1. Type Hints

```python
from typing import Callable, Optional, List, Dict, Tuple, Union
import numpy as np
from numpy.typing import NDArray

def example_function(
    x: NDArray[np.float64],
    dim: int,
    bounds: Optional[NDArray[np.float64]] = None
) -> Tuple[NDArray[np.float64], float]:
    """Function with proper type hints."""
    pass
```

### 2. Docstrings (Google Style)

```python
def optimize(self, verbose: bool = True) -> Dict[str, any]:
    """Run the optimization process.
    
    This method initializes the population and iteratively updates
    it until the maximum number of iterations is reached.
    
    Args:
        verbose: If True, print progress information during optimization.
        
    Returns:
        A dictionary containing:
            - best_solution (np.ndarray): The best solution found
            - best_fitness (float): The fitness of the best solution
            - fitness_history (List[float]): Best fitness at each iteration
            - iterations (int): Total number of iterations performed
            
    Example:
        >>> optimizer = PSO(func, dim=10, bounds=bounds)
        >>> result = optimizer.optimize(verbose=True)
        >>> print(f"Best fitness: {result['best_fitness']}")
    """
    pass
```

### 3. Error Handling

```python
class InvalidBoundsError(Exception):
    """Raised when bounds are invalid."""
    pass

def validate_bounds(bounds: np.ndarray, dim: int):
    """Validate bounds array."""
    if bounds.shape != (2, dim):
        raise InvalidBoundsError(
            f"Bounds shape must be (2, {dim}), got {bounds.shape}"
        )
    if np.any(bounds[0] >= bounds[1]):
        raise InvalidBoundsError(
            "Lower bounds must be less than upper bounds"
        )
```

### 4. NumPy Best Practices

```python
# Good: Vectorized operations
fitness = np.sum(population ** 2, axis=1)

# Bad: Loops
fitness = np.array([np.sum(ind ** 2) for ind in population])

# Good: In-place operations when possible
population += velocities

# Good: Use np.random.Generator for reproducibility
rng = np.random.default_rng(seed=42)
random_values = rng.random(size=(pop_size, dim))
```

### 5. Performance Optimization

```python
# Use numba for critical loops if needed (optional)
from numba import jit

@jit(nopython=True)
def compute_distances(particles: np.ndarray) -> np.ndarray:
    """Compute pairwise distances efficiently."""
    n = particles.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((particles[i] - particles[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    return distances
```

---

## Testing Strategy

### Unit Tests Structure

```python
# tests/test_algorithms/test_pso.py

import unittest
import numpy as np
from swarm_intelligence.algorithms.swarm import PSO
from swarm_intelligence.problems.continuous import Sphere

class TestPSO(unittest.TestCase):
    """Test cases for PSO algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = Sphere(dim=10)
        self.optimizer = PSO(
            objective_func=self.problem.evaluate,
            dim=10,
            bounds=self.problem.get_bounds(),
            pop_size=20,
            max_iter=100,
            seed=42
        )
    
    def test_initialization(self):
        """Test population initialization."""
        self.optimizer.initialize_population()
        self.assertEqual(self.optimizer.population.shape, (20, 10))
        
    def test_bounds_enforcement(self):
        """Test that solutions stay within bounds."""
        result = self.optimizer.optimize(verbose=False)
        solution = result['best_solution']
        bounds = self.problem.get_bounds()
        self.assertTrue(np.all(solution >= bounds[0]))
        self.assertTrue(np.all(solution <= bounds[1]))
        
    def test_convergence(self):
        """Test that fitness improves over iterations."""
        result = self.optimizer.optimize(verbose=False)
        history = result['fitness_history']
        # Check that fitness generally decreases
        self.assertLess(history[-1], history[0])
        
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        opt1 = PSO(self.problem.evaluate, 10, self.problem.get_bounds(), seed=42)
        opt2 = PSO(self.problem.evaluate, 10, self.problem.get_bounds(), seed=42)
        result1 = opt1.optimize(verbose=False)
        result2 = opt2.optimize(verbose=False)
        np.testing.assert_array_almost_equal(
            result1['best_solution'], 
            result2['best_solution']
        )
```

---

## Documentation

### README.md Structure

```markdown
# AI Swarm Intelligence Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]
[![NumPy](https://img.shields.io/badge/numpy-1.20+-orange.svg)]
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]

## 🎯 Overview

A comprehensive implementation of 8 optimization algorithms with real-time visualization for solving continuous and discrete optimization problems.

### Implemented Algorithms
- **Swarm Intelligence**: PSO, ABC, FA, CS, ACO
- **Evolutionary**: Genetic Algorithm
- **Local Search**: Hill Climbing, Simulated Annealing

### Test Problems
- **Continuous**: Sphere, Rastrigin, Rosenbrock, Ackley
- **Discrete**: Knapsack, TSP, Graph Coloring

## 🚀 Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage

```python
from swarm_intelligence.algorithms.swarm import PSO
from swarm_intelligence.problems.continuous import RastriginFunction

problem = RastriginFunction(dim=10)
optimizer = PSO(
    objective_func=problem.evaluate,
    dim=problem.dim,
    bounds=problem.get_bounds(),
    pop_size=50,
    max_iter=1000
)
result = optimizer.optimize()
```

## 📊 Features

- Real-time visualization during optimization
- Comprehensive benchmarking framework
- Statistical analysis tools
- Jupyter notebook tutorials
- Export results to CSV/JSON/LaTeX

## 📚 Documentation

See `docs/` for detailed documentation and API reference.

## 🎓 Course Information

**Course**: Introduction to Artificial Intelligence (CSC14003)  
**Institution**: VNUHCM - University of Science  
**Project**: Swarm Intelligence Algorithms

```

### API Documentation Structure
```markdown
# API Reference

## Core Classes

### BaseOptimizer

Base class for all optimization algorithms.

**Constructor Parameters:**
- `objective_func` (Callable): Function to minimize
- `dim` (int): Problem dimensionality
- `bounds` (np.ndarray): Search space bounds, shape (2, dim)
- `pop_size` (int): Population size (default: 50)
- `max_iter` (int): Maximum iterations (default: 1000)
- `seed` (Optional[int]): Random seed (default: None)
- `visualizer` (Optional[RealtimeVisualizer]): Visualization object

**Methods:**
- `optimize(verbose: bool = True) -> dict`: Run optimization
- `initialize_population() -> None`: Initialize population (abstract)
- `update_population() -> None`: Update for one iteration (abstract)

**Returns from optimize():**
```python
{
    'best_solution': np.ndarray,  # Best solution found
    'best_fitness': float,         # Best fitness value
    'fitness_history': List[float], # Fitness per iteration
    'iterations': int,             # Total iterations
    'time': float                  # Execution time
}
```

### Example Usage

```python
optimizer = PSO(
    objective_func=lambda x: np.sum(x**2),
    dim=10,
    bounds=np.array([[-5.12]*10, [5.12]*10]),
    pop_size=30,
    max_iter=500,
    seed=42
)
result = optimizer.optimize(verbose=True)
print(f"Best fitness: {result['best_fitness']:.6f}")
```

```

---

## Implementation Priority Order

### Phase 1: Core Foundation (Week 1)
1. **Day 1-2**: Base classes
   - [ ] `base_algorithm.py` - Complete BaseOptimizer
   - [ ] `base_problem.py` - Complete OptimizationProblem
   - [ ] `solution.py` - Solution representation
   - [ ] `population.py` - Population management

2. **Day 3-4**: First algorithm + problem
   - [ ] `pso.py` - Complete PSO implementation
   - [ ] `rastrigin.py` - Rastrigin function
   - [ ] Test PSO on Rastrigin
   - [ ] Basic convergence plot

3. **Day 5-7**: Basic visualization
   - [ ] `convergence_plot.py` - Convergence curves
   - [ ] `realtime_plotter.py` - Basic real-time plotting
   - [ ] Test with PSO

### Phase 2: Complete Algorithms (Week 2)
4. **Day 8-9**: Swarm algorithms
   - [ ] `abc.py` - Artificial Bee Colony
   - [ ] `fa.py` - Firefly Algorithm
   - [ ] `cs.py` - Cuckoo Search
   - [ ] Test all on Rastrigin

5. **Day 10-11**: Other algorithms
   - [ ] `ga.py` - Genetic Algorithm
   - [ ] `hc.py` - Hill Climbing
   - [ ] `sa.py` - Simulated Annealing
   - [ ] `aco.py` - Ant Colony (discrete only)

6. **Day 12-14**: All continuous problems
   - [ ] `sphere.py`
   - [ ] `rosenbrock.py`
   - [ ] `ackley.py`
   - [ ] Test all algorithms on all problems

### Phase 3: Discrete Problems & Visualization (Week 3)
7. **Day 15-16**: Discrete problems
   - [ ] `knapsack.py` - 0/1 Knapsack
   - [ ] `tsp.py` - Traveling Salesman
   - [ ] `graph_coloring.py` - Graph Coloring
   - [ ] Adapt algorithms for discrete problems

8. **Day 17-18**: Advanced visualization
   - [ ] `landscape_plot.py` - 3D surfaces
   - [ ] `swarm_animation.py` - Animated visualizations
   - [ ] `comparison_plot.py` - Multi-algorithm comparison
   - [ ] Real-time updates for all algorithms

9. **Day 19-21**: Benchmarking framework
   - [ ] `runner.py` - Experiment runner
   - [ ] `metrics.py` - Performance metrics
   - [ ] `statistical_tests.py` - Statistical analysis
   - [ ] Run comprehensive experiments

### Phase 4: Documentation & Polish (Week 4)
10. **Day 22-23**: Jupyter notebooks
    - [ ] Create all 7 notebooks
    - [ ] Add visualizations to notebooks
    - [ ] Test notebook execution

11. **Day 24-25**: Documentation
    - [ ] Complete README.md
    - [ ] API documentation
    - [ ] Algorithm descriptions with math
    - [ ] Usage examples

12. **Day 26-28**: Final polish
    - [ ] Unit tests for all components
    - [ ] Performance optimization
    - [ ] Bug fixes
    - [ ] Demo video creation
    - [ ] Final report generation

---

## File Templates

### Algorithm Template

```python
"""
Particle Swarm Optimization (PSO) Algorithm.

This module implements the PSO algorithm for continuous optimization problems.

Mathematical Formulation:
    Velocity update:
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
    
    Position update:
        x_i(t+1) = x_i(t) + v_i(t+1)
    
    where:
        w: inertia weight
        c1, c2: cognitive and social coefficients
        r1, r2: random numbers in [0,1]
        pbest_i: personal best position of particle i
        gbest: global best position

References:
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95 - International Conference on Neural Networks,
    4, 1942-1948.
"""

import numpy as np
from typing import Optional, Callable
from numpy.typing import NDArray

from ..core.base_algorithm import BaseOptimizer


class PSO(BaseOptimizer):
    """Particle Swarm Optimization algorithm.
    
    PSO is a population-based stochastic optimization technique inspired by
    the social behavior of bird flocking or fish schooling.
    
    Attributes:
        w (float): Inertia weight controlling previous velocity influence.
        c1 (float): Cognitive coefficient (personal best attraction).
        c2 (float): Social coefficient (global best attraction).
        v_max (Optional[float]): Maximum velocity magnitude.
        velocities (NDArray): Current velocities of all particles.
        personal_best_positions (NDArray): Best positions found by each particle.
        personal_best_fitness (NDArray): Fitness values of personal bests.
        global_best_position (NDArray): Best position found by entire swarm.
        global_best_fitness (float): Fitness of global best position.
    
    Example:
        >>> from swarm_intelligence.problems.continuous import Sphere
        >>> problem = Sphere(dim=10)
        >>> optimizer = PSO(
        ...     objective_func=problem.evaluate,
        ...     dim=10,
        ...     bounds=problem.get_bounds(),
        ...     pop_size=50,
        ...     max_iter=1000,
        ...     w=0.7,
        ...     c1=1.5,
        ...     c2=1.5
        ... )
        >>> result = optimizer.optimize()
        >>> print(f"Best fitness: {result['best_fitness']:.6f}")
    """
    
    def __init__(
        self,
        objective_func: Callable[[NDArray[np.float64]], float],
        dim: int,
        bounds: NDArray[np.float64],
        pop_size: int = 50,
        max_iter: int = 1000,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: Optional[float] = None,
        seed: Optional[int] = None,
        visualizer: Optional['RealtimeVisualizer'] = None
    ):
        """Initialize PSO optimizer.
        
        Args:
            objective_func: Function to minimize, takes array and returns float.
            dim: Dimensionality of the search space.
            bounds: Array of shape (2, dim) with [lower_bounds, upper_bounds].
            pop_size: Number of particles in the swarm.
            max_iter: Maximum number of iterations.
            w: Inertia weight (typical range: 0.4-0.9).
            c1: Cognitive coefficient (typical range: 1.5-2.0).
            c2: Social coefficient (typical range: 1.5-2.0).
            v_max: Maximum velocity (if None, set to 10% of search range).
            seed: Random seed for reproducibility.
            visualizer: Optional visualizer for real-time plotting.
        
        Raises:
            ValueError: If bounds shape is incorrect or parameters are invalid.
        """
        super().__init__(
            objective_func=objective_func,
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter,
            seed=seed,
            visualizer=visualizer
        )
        
        # PSO-specific parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Set maximum velocity
        if v_max is None:
            search_range = bounds[1] - bounds[0]
            self.v_max = 0.1 * search_range
        else:
            self.v_max = v_max
        
        # Initialize PSO-specific attributes
        self.velocities: Optional[NDArray[np.float64]] = None
        self.personal_best_positions: Optional[NDArray[np.float64]] = None
        self.personal_best_fitness: Optional[NDArray[np.float64]] = None
        self.global_best_position: Optional[NDArray[np.float64]] = None
        self.global_best_fitness: float = float('inf')
        
    def initialize_population(self) -> None:
        """Initialize particle positions and velocities.
        
        Particles are initialized uniformly within bounds.
        Velocities are initialized randomly in range [-v_max, v_max].
        """
        # Initialize positions uniformly within bounds
        self.population = np.random.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.dim)
        )
        
        # Initialize velocities
        self.velocities = np.random.uniform(
            low=-self.v_max,
            high=self.v_max,
            size=(self.pop_size, self.dim)
        )
        
        # Evaluate initial population
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        
        # Initialize personal bests
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = fitness.copy()
        
        # Initialize global best
        best_idx = np.argmin(fitness)
        self.global_best_position = self.population[best_idx].copy()
        self.global_best_fitness = fitness[best_idx]
        
        # Update tracking
        self.best_solution = self.global_best_position.copy()
        self.best_fitness = self.global_best_fitness
        self.fitness_history.append(self.best_fitness)
        
    def update_population(self) -> None:
        """Perform one iteration of PSO updates.
        
        Updates velocities and positions according to PSO equations,
        then updates personal and global bests.
        """
        # Generate random coefficients
        r1 = np.random.random((self.pop_size, self.dim))
        r2 = np.random.random((self.pop_size, self.dim))
        
        # Update velocities
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.population)
        social = self.c2 * r2 * (self.global_best_position - self.population)
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Clip velocities to v_max
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        
        # Update positions
        self.population += self.velocities
        
        # Enforce bounds
        self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
        
        # Evaluate new positions
        fitness = np.array([self.evaluate(ind) for ind in self.population])
        
        # Update personal bests
        improved = fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.population[improved]
        self.personal_best_fitness[improved] = fitness[improved]
        
        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.global_best_fitness:
            self.global_best_position = self.population[best_idx].copy()
            self.global_best_fitness = fitness[best_idx]
            self.best_solution = self.global_best_position.copy()
            self.best_fitness = self.global_best_fitness
        
        # Track fitness
        self.fitness_history.append(self.best_fitness)
        
    def get_algorithm_name(self) -> str:
        """Return the algorithm name.
        
        Returns:
            str: "PSO"
        """
        return "PSO"
```

### Problem Template

```python
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
    - Difficulty: Hard due to large number of local minima

References:
    Rastrigin, L. A. (1974). Systems of extremal control.
    Mir, Moscow.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ..core.base_problem import OptimizationProblem


class RastriginFunction(OptimizationProblem):
    """Rastrigin function for continuous optimization.
    
    A highly multimodal benchmark function commonly used to test
    optimization algorithms' ability to escape local optima.
    
    Attributes:
        dim (int): Problem dimensionality.
        A (float): Amplitude parameter (default: 10).
        domain (tuple): Search space domain (-5.12, 5.12).
    
    Example:
        >>> problem = RastriginFunction(dim=10)
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
        return self.A * n + sum_term
    
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
            'optimal_solution': self.get_optimal_solution(),
            'characteristics': [
                'Highly multimodal',
                'Many regularly distributed local minima',
                'Global optimum at origin',
                'Continuous and differentiable'
            ],
            'difficulty': 'Hard',
            'formula': 'f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]'
        }
```

---

## Common Pitfalls to Avoid

### 1. Bounds Handling

```python
# ❌ Bad: Direct clipping without velocity adjustment
self.population = np.clip(self.population, lower, upper)

# ✅ Good: Reflect or absorb at boundaries
mask = (self.population < lower) | (self.population > upper)
self.velocities[mask] *= -0.5  # Absorb
self.population = np.clip(self.population, lower, upper)
```

### 2. Fitness Evaluation

```python
# ❌ Bad: Repeated evaluations
for iter in range(max_iter):
    for i in range(pop_size):
        fitness[i] = evaluate(population[i])  # Evaluates every iteration

# ✅ Good: Cache and only re-evaluate when necessary
fitness_cache = {}
def evaluate_cached(x):
    key = tuple(x)
    if key not in fitness_cache:
        fitness_cache[key] = evaluate(x)
    return fitness_cache[key]
```

### 3. Random Number Generation

```python
# ❌ Bad: Using global numpy random state
np.random.seed(42)
x = np.random.random()

# ✅ Good: Use Generator for reproducibility and thread safety
rng = np.random.default_rng(seed=42)
x = rng.random()
```

### 4. Memory Management

```python
# ❌ Bad: Storing entire population history
population_history = []
for iter in range(10000):
    population_history.append(population.copy())  # Memory explosion!

# ✅ Good: Store only necessary information
best_solutions = []
for iter in range(10000):
    best_solutions.append(population[best_idx].copy())
```

### 5. Visualization Performance

```python
# ❌ Bad: Updating plot every iteration
for iter in range(1000):
    plt.clf()
    plt.plot(fitness_history)
    plt.draw()
    plt.pause(0.001)  # Very slow!

# ✅ Good: Update plot periodically
line, = plt.plot([], [])
for iter in range(1000):
    if iter % 10 == 0:  # Update every 10 iterations
        line.set_data(range(len(fitness_history)), fitness_history)
        plt.draw()
        plt.pause(0.001)
```

---

## Integration Examples

### Example 1: Using with Real-time Visualization

```python
from swarm_intelligence.algorithms.swarm import PSO
from swarm_intelligence.problems.continuous import RastriginFunction
from swarm_intelligence.visualization import RealtimeVisualizer

# Setup problem
problem = RastriginFunction(dim=2)

# Create visualizer
visualizer = RealtimeVisualizer(
    problem=problem,
    mode='2d',
    update_interval=5  # Update every 5 iterations
)

# Create optimizer with visualizer
optimizer = PSO(
    objective_func=problem.evaluate,
    dim=2,
    bounds=problem.get_bounds(),
    pop_size=30,
    max_iter=200,
    visualizer=visualizer
)

# Run optimization with live visualization
result = optimizer.optimize(verbose=True)

# Finalize and save
visualizer.finalize(result)
visualizer.save_animation('pso_rastrigin.gif')
```

### Example 2: Batch Experiments

```python
from swarm_intelligence.benchmarks import ExperimentRunner
from swarm_intelligence.algorithms.swarm import PSO, ABC, FA
from swarm_intelligence.problems.continuous import Rastrigin, Rosenbrock

# Define algorithms to test
algorithms = [PSO, ABC, FA]

# Define problems
problems = [
    RastriginFunction(dim=10),
    RastriginFunction(dim=30),
    RosenbrockFunction(dim=10)
]

# Create runner
runner = ExperimentRunner(
    algorithms=algorithms,
    problems=problems,
    n_runs=20
)

# Run all experiments (can be parallelized)
results = runner.run_all(parallel=True, n_jobs=4)

# Save results
runner.save_results('results/experiment_results.json')

# Generate comparison plots
from swarm_intelligence.visualization import plot_algorithm_comparison
plot_algorithm_comparison(results, save_path='results/comparison.png')
```

### Example 3: Custom Problem

```python
import numpy as np
from swarm_intelligence.core import OptimizationProblem
from swarm_intelligence.algorithms.swarm import PSO

class CustomProblem(OptimizationProblem):
    """My custom optimization problem."""
    
    def __init__(self):
        super().__init__(dim=5, problem_type='continuous', name='Custom')
    
    def evaluate(self, x):
        # Your custom objective function
        return np.sum(x**2) + np.prod(np.sin(x))
    
    def get_bounds(self):
        return np.array([[-10]*5, [10]*5])
    
    def get_optimal_value(self):
        return None  # Unknown
    
    def get_optimal_solution(self):
        return None  # Unknown

# Use it
problem = CustomProblem()
optimizer = PSO(problem.evaluate, problem.dim, problem.get_bounds())
result = optimizer.optimize()
```

---

## Performance Benchmarks (Target)

After implementation, aim for these performance targets:

### Computation Time (10D Rastrigin, 1000 iterations, 50 particles)

- PSO: ~2-3 seconds
- ABC: ~3-4 seconds
- GA: ~3-4 seconds
- FA: ~4-5 seconds
- CS: ~3-4 seconds

### Memory Usage

- Maximum: <500MB for all experiments
- Per algorithm instance: <100MB

### Solution Quality (10D Rastrigin, 20 runs)

- Target: Mean fitness < 10.0
- Best algorithms should achieve: < 5.0

---

## Final Checklist

Before considering the project complete:

- [ ] All 8 algorithms implemented and tested
- [ ] All 7 benchmark problems implemented
- [ ] Real-time visualization working
- [ ] All 7 Jupyter notebooks completed
- [ ] Comprehensive README with examples
- [ ] Unit tests with >80% coverage
- [ ] Documentation for all public APIs
- [ ] Experiment results generated
- [ ] Comparison plots created
- [ ] Demo video recorded
- [ ] Code follows PEP 8 style guide
- [ ] No hardcoded paths or magic numbers
- [ ] All dependencies in requirements.txt
- [ ] setup.py configured correctly
- [ ] .gitignore includes results/, **pycache**, etc.

---

## GitHub Copilot Usage Tips

To get the best code generation from Copilot:

1. **Start with function signatures and docstrings**
   - Write the function signature with type hints
   - Write a detailed docstring
   - Copilot will generate implementation

2. **Use descriptive variable names**
   - `personal_best_positions` instead of `pbest`
   - Copilot understands context better

3. **Comment your algorithm steps**
   - Write comments describing each step
   - Copilot will implement the steps

4. **Provide examples in docstrings**
   - Examples help Copilot understand expected behavior

5. **Use TODO comments**
   - `# TODO: Implement velocity update according to PSO formula`
   - Copilot will suggest implementation

---

## Additional Resources

### Mathematical References

- Optimization algorithms: <https://www.swarmintelligence.org/>
- Benchmark functions: <https://www.al-roomi.org/benchmarks>

### Python Best Practices

- PEP 8: <https://peps.python.org/pep-0008/>
- Type hints: <https://docs.python.org/3/library/typing.html>
- NumPy best practices: <https://numpy.org/doc/stable/user/basics.performance.html>

### Visualization

- Matplotlib animations: <https://matplotlib.org/stable/api/animation_api.html>
- Real-time plotting: <https://matplotlib.org/stable/users/explain/animations/blitting.html>

---

## Success Criteria

The project is successful when:

1. All algorithms converge to near-optimal solutions
2. Real-time visualization runs smoothly
3. Code is clean, documented, and testable
4. Notebooks are educational and runnable
5. Results are reproducible
6. Project structure is professional and maintainable

Good luck with the implementation! 🚀
