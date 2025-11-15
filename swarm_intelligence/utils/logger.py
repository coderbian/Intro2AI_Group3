"""
Logging utilities for optimization experiments.

Provides consistent logging across the framework with file and console handlers.

Author: Group 3
Date: 2024
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "swarm_intelligence",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with file and/or console handlers.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "swarm_intelligence") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger for optimization experiments with structured output.
    
    Attributes:
        logger: Underlying logger instance
        experiment_name: Name of the experiment
        start_time: Experiment start timestamp
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        level: int = logging.INFO
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Create log file with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=f"exp_{experiment_name}",
            level=level,
            log_file=log_file,
            console=True
        )
    
    def log_start(self, config: dict):
        """Log experiment start with configuration."""
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Started at: {self.start_time}")
        self.logger.info("=" * 70)
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 70)
    
    def log_iteration(self, iteration: int, fitness: float, algorithm: str):
        """Log iteration progress."""
        self.logger.debug(
            f"[{algorithm}] Iteration {iteration}: fitness = {fitness:.6f}"
        )
    
    def log_result(self, algorithm: str, problem: str, fitness: float, time: float):
        """Log optimization result."""
        self.logger.info(
            f"Result - {algorithm} on {problem}: "
            f"fitness={fitness:.6f}, time={time:.3f}s"
        )
    
    def log_end(self):
        """Log experiment end."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment completed at: {end_time}")
        self.logger.info(f"Total duration: {duration:.2f}s")
        self.logger.info("=" * 70)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
