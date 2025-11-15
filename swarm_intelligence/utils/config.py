"""
Configuration management using YAML.

Handles loading, saving, and validation of experiment configurations.

Author: Group 3
Date: 2024
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AlgorithmConfig:
    """Algorithm configuration."""
    name: str
    pop_size: int = 50
    max_iter: int = 1000
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemConfig:
    """Problem configuration."""
    name: str
    dim: int = 10
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    num_runs: int = 30
    seed: Optional[int] = 42
    parallel: bool = False
    save_results: bool = True
    output_dir: str = "results"


@dataclass
class Config:
    """
    Main configuration class.
    
    Attributes:
        algorithms: List of algorithm configurations
        problems: List of problem configurations
        experiment: Experiment settings
    """
    algorithms: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    experiment: Optional[ExperimentConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        config = cls()
        
        # Parse algorithms
        if 'algorithms' in data:
            config.algorithms = [
                AlgorithmConfig(**alg) for alg in data['algorithms']
            ]
        
        # Parse problems
        if 'problems' in data:
            config.problems = [
                ProblemConfig(**prob) for prob in data['problems']
            ]
        
        # Parse experiment settings
        if 'experiment' in data:
            config.experiment = ExperimentConfig(**data['experiment'])
        
        return config


def load_config(filepath: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Path to YAML configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Config.from_dict(data)


def save_config(config: Config, filepath: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        filepath: Output file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to: {filepath}")


def create_default_config() -> Config:
    """
    Create a default configuration.
    
    Returns:
        Default Config object
    """
    config = Config()
    
    # Default algorithms
    config.algorithms = [
        AlgorithmConfig(name="PSO", pop_size=50, max_iter=1000, params={"w": 0.7, "c1": 1.5, "c2": 1.5}),
        AlgorithmConfig(name="ABC", pop_size=50, max_iter=1000, params={"limit": 100}),
        AlgorithmConfig(name="GA", pop_size=50, max_iter=1000, params={"crossover_rate": 0.8, "mutation_rate": 0.1}),
    ]
    
    # Default problems
    config.problems = [
        ProblemConfig(name="Sphere", dim=10),
        ProblemConfig(name="Rastrigin", dim=10),
        ProblemConfig(name="Rosenbrock", dim=10),
    ]
    
    # Default experiment settings
    config.experiment = ExperimentConfig(
        name="default_experiment",
        num_runs=30,
        seed=42,
        parallel=False,
        save_results=True,
        output_dir="results"
    )
    
    return config


def validate_config(config: Config) -> bool:
    """
    Validate configuration.
    
    Args:
        config: Config object to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not config.algorithms:
        raise ValueError("At least one algorithm must be specified")
    
    if not config.problems:
        raise ValueError("At least one problem must be specified")
    
    if config.experiment is None:
        raise ValueError("Experiment configuration is required")
    
    # Validate algorithm configs
    for alg in config.algorithms:
        if alg.pop_size <= 0:
            raise ValueError(f"Invalid pop_size for {alg.name}: {alg.pop_size}")
        if alg.max_iter <= 0:
            raise ValueError(f"Invalid max_iter for {alg.name}: {alg.max_iter}")
    
    # Validate problem configs
    for prob in config.problems:
        if prob.dim <= 0:
            raise ValueError(f"Invalid dim for {prob.name}: {prob.dim}")
    
    # Validate experiment config
    if config.experiment.num_runs <= 0:
        raise ValueError(f"Invalid num_runs: {config.experiment.num_runs}")
    
    return True
