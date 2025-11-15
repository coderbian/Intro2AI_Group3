"""
Real-time Plotter for Optimization Algorithms

Provides live visualization of algorithm convergence during execution.

Features:
- Non-blocking real-time updates
- Multiple subplot support
- Customizable update frequency
- Thread-safe operation

Author: Group 3
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
import threading
import time


class RealtimePlotter:
    """
    Real-time plotter for monitoring optimization progress.
    
    Displays convergence curves and optionally population diversity or other metrics
    during algorithm execution.
    
    Attributes:
        fig: Matplotlib figure
        axes: List of axes for subplots
        histories: Dictionary storing plot data
        update_interval: Minimum time between updates (seconds)
        is_running: Flag indicating if plotting is active
    """
    
    def __init__(
        self,
        n_subplots: int = 1,
        titles: Optional[List[str]] = None,
        ylabels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        update_interval: float = 0.1
    ):
        """
        Initialize real-time plotter.
        
        Args:
            n_subplots: Number of subplots (default: 1)
            titles: List of subplot titles (optional)
            ylabels: List of y-axis labels (optional)
            figsize: Figure size (width, height)
            update_interval: Minimum time between updates in seconds
        """
        self.n_subplots = n_subplots
        self.update_interval = update_interval
        self.is_running = False
        self.last_update = 0
        self.lock = threading.Lock()
        
        # Create figure and subplots
        if n_subplots == 1:
            self.fig, ax = plt.subplots(figsize=figsize)
            self.axes = [ax]
        else:
            self.fig, self.axes = plt.subplots(1, n_subplots, figsize=figsize)
        
        # Initialize histories for each subplot
        self.histories = {i: {} for i in range(n_subplots)}
        
        # Set titles and labels
        for i, ax in enumerate(self.axes):
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            
            if ylabels and i < len(ylabels):
                ax.set_ylabel(ylabels[i], fontsize=10)
            else:
                ax.set_ylabel('Fitness', fontsize=10)
            
            ax.grid(True, alpha=0.3)
        
        # Enable interactive mode
        plt.ion()
        plt.show()
        
        self.is_running = True
    
    def add_point(self, subplot_idx: int, series_name: str, x: float, y: float):
        """
        Add a data point to a specific series.
        
        Args:
            subplot_idx: Index of the subplot
            series_name: Name of the data series
            x: X coordinate (iteration number)
            y: Y coordinate (fitness value)
        """
        with self.lock:
            if series_name not in self.histories[subplot_idx]:
                self.histories[subplot_idx][series_name] = {'x': [], 'y': []}
            
            self.histories[subplot_idx][series_name]['x'].append(x)
            self.histories[subplot_idx][series_name]['y'].append(y)
    
    def update(self, force: bool = False):
        """
        Update the plot with current data.
        
        Args:
            force: Force update even if interval hasn't elapsed
        """
        current_time = time.time()
        
        # Check if enough time has passed since last update
        if not force and (current_time - self.last_update) < self.update_interval:
            return
        
        with self.lock:
            for subplot_idx, ax in enumerate(self.axes):
                ax.clear()
                
                # Reapply labels and grid
                ax.set_xlabel('Iteration', fontsize=10)
                ax.set_ylabel('Fitness', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Plot all series in this subplot
                for series_name, data in self.histories[subplot_idx].items():
                    if len(data['x']) > 0:
                        ax.plot(data['x'], data['y'], linewidth=2, label=series_name)
                
                # Add legend if there are multiple series
                if len(self.histories[subplot_idx]) > 1:
                    ax.legend(loc='best', fontsize=9)
            
            plt.tight_layout()
            
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                # Handle case where window was closed
                self.is_running = False
        
        self.last_update = current_time
    
    def close(self):
        """Close the plotter and cleanup resources."""
        self.is_running = False
        plt.close(self.fig)
    
    def save(self, filepath: str, dpi: int = 300):
        """
        Save the current plot to file.
        
        Args:
            filepath: Output file path
            dpi: Image resolution (default: 300)
        """
        with self.lock:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MultiAlgorithmPlotter:
    """
    Real-time plotter for comparing multiple algorithms simultaneously.
    
    Useful for benchmarking and visual comparison of convergence behavior.
    """
    
    def __init__(
        self,
        algorithm_names: List[str],
        figsize: Tuple[int, int] = (14, 8),
        update_interval: float = 0.5
    ):
        """
        Initialize multi-algorithm plotter.
        
        Args:
            algorithm_names: List of algorithm names to track
            figsize: Figure size (width, height)
            update_interval: Update frequency in seconds
        """
        self.algorithm_names = algorithm_names
        self.update_interval = update_interval
        self.is_running = False
        self.last_update = 0
        self.lock = threading.Lock()
        
        # Create figure with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Initialize data storage
        self.best_fitness = {name: [] for name in algorithm_names}
        self.current_fitness = {name: [] for name in algorithm_names}
        self.iterations = {name: [] for name in algorithm_names}
        
        # Setup axes
        self.ax1.set_title('Best Fitness Over Time', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Iteration', fontsize=10)
        self.ax1.set_ylabel('Best Fitness', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Current Fitness Distribution', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Algorithm', fontsize=10)
        self.ax2.set_ylabel('Fitness', fontsize=10)
        self.ax2.grid(True, alpha=0.3, axis='y')
        
        # Enable interactive mode
        plt.ion()
        plt.show()
        
        self.is_running = True
    
    def update_algorithm(self, algorithm_name: str, iteration: int, 
                        best_fitness: float, current_fitness: float):
        """
        Update data for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            iteration: Current iteration number
            best_fitness: Best fitness found so far
            current_fitness: Current iteration fitness
        """
        with self.lock:
            if algorithm_name in self.algorithm_names:
                self.iterations[algorithm_name].append(iteration)
                self.best_fitness[algorithm_name].append(best_fitness)
                self.current_fitness[algorithm_name].append(current_fitness)
    
    def update_display(self, force: bool = False):
        """
        Update the visualization.
        
        Args:
            force: Force update regardless of interval
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_update) < self.update_interval:
            return
        
        with self.lock:
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            
            # Reapply formatting
            self.ax1.set_title('Best Fitness Over Time', fontsize=12, fontweight='bold')
            self.ax1.set_xlabel('Iteration', fontsize=10)
            self.ax1.set_ylabel('Best Fitness', fontsize=10)
            self.ax1.grid(True, alpha=0.3)
            
            # Plot convergence curves
            for name in self.algorithm_names:
                if len(self.iterations[name]) > 0:
                    self.ax1.plot(self.iterations[name], self.best_fitness[name], 
                                linewidth=2, label=name)
            
            self.ax1.legend(loc='best', fontsize=9)
            
            # Plot current fitness comparison
            self.ax2.set_title('Current Best Fitness', fontsize=12, fontweight='bold')
            self.ax2.set_ylabel('Fitness', fontsize=10)
            self.ax2.grid(True, alpha=0.3, axis='y')
            
            current_best = []
            labels = []
            for name in self.algorithm_names:
                if len(self.best_fitness[name]) > 0:
                    current_best.append(self.best_fitness[name][-1])
                    labels.append(name)
            
            if len(current_best) > 0:
                colors = plt.cm.viridis(np.linspace(0, 1, len(current_best)))
                self.ax2.bar(labels, current_best, color=colors, alpha=0.7)
                self.ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                self.is_running = False
        
        self.last_update = current_time
    
    def close(self):
        """Close the plotter."""
        self.is_running = False
        plt.close(self.fig)
    
    def save(self, filepath: str, dpi: int = 300):
        """Save the plot to file."""
        with self.lock:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Comparison plot saved to: {filepath}")
