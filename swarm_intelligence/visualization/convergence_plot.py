"""
Convergence Plot Visualization

Plots fitness evolution over iterations to visualize algorithm convergence behavior.

Features:
- Single algorithm convergence curve
- Multiple algorithm comparison
- Log scale support for wide fitness ranges
- Statistical confidence intervals
- Customizable styling

Author: Group 3
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import os


def plot_convergence(
    history: List[float],
    title: str = "Convergence Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Fitness",
    log_scale: bool = False,
    show_best: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot convergence curve for a single optimization run.
    
    Args:
        history: List of fitness values per iteration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic scale for y-axis
        show_best: Show horizontal line at best fitness
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(1, len(history) + 1)
    
    # Plot convergence curve
    ax.plot(iterations, history, 'b-', linewidth=2, label='Fitness')
    
    # Show best fitness line
    if show_best:
        best_fitness = min(history)
        ax.axhline(y=best_fitness, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Best: {best_fitness:.6f}')
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    return fig


def plot_multiple_convergence(
    histories: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Fitness",
    log_scale: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot convergence curves for multiple algorithms.
    
    Args:
        histories: Dictionary mapping algorithm names to fitness histories
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic scale for y-axis
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        colors: List of colors for each algorithm (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default color palette
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each algorithm
    for idx, (name, history) in enumerate(histories.items()):
        iterations = np.arange(1, len(history) + 1)
        color = colors[idx % len(colors)]
        
        ax.plot(iterations, history, linewidth=2, label=name, color=color)
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_convergence_with_std(
    mean_history: List[float],
    std_history: List[float],
    title: str = "Convergence with Confidence Interval",
    xlabel: str = "Iteration",
    ylabel: str = "Fitness",
    log_scale: bool = False,
    confidence: float = 0.95,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot convergence curve with confidence interval from multiple runs.
    
    Args:
        mean_history: Mean fitness values per iteration
        std_history: Standard deviation per iteration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic scale for y-axis
        confidence: Confidence level (default: 0.95)
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(1, len(mean_history) + 1)
    mean = np.array(mean_history)
    std = np.array(std_history)
    
    # Calculate confidence interval (assuming normal distribution)
    # For 95% confidence: ±1.96 * std
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
    margin = z_score * std
    
    # Plot mean line
    ax.plot(iterations, mean, 'b-', linewidth=2, label='Mean')
    
    # Plot confidence interval
    ax.fill_between(iterations, mean - margin, mean + margin, 
                     alpha=0.3, color='blue', 
                     label=f'{int(confidence*100)}% CI')
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot with CI saved to: {save_path}")
    
    return fig


def plot_convergence_subplots(
    histories: Dict[str, List[float]],
    problems: Optional[List[str]] = None,
    title: str = "Algorithm Convergence on Multiple Problems",
    log_scale: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot convergence curves in subplots for better comparison.
    
    Args:
        histories: Nested dictionary {problem: {algorithm: history}}
        problems: List of problem names (extracted from histories if None)
        title: Overall plot title
        log_scale: Use logarithmic scale for y-axis
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if problems is None:
        problems = list(histories.keys())
    
    n_problems = len(problems)
    n_cols = 2 if n_problems > 1 else 1
    n_rows = (n_problems + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_problems == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, problem in enumerate(problems):
        ax = axes[idx]
        
        for jdx, (alg_name, history) in enumerate(histories[problem].items()):
            iterations = np.arange(1, len(history) + 1)
            color = colors[jdx % len(colors)]
            ax.plot(iterations, history, linewidth=2, label=alg_name, color=color)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Fitness', fontsize=10)
        ax.set_title(problem, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_problems, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subplot comparison saved to: {save_path}")
    
    return fig
