"""
Data export utilities for results.

Exports optimization results to various formats: CSV, JSON, LaTeX tables.

Author: Group 3
Date: 2024
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


def export_to_csv(
    data: List[Dict[str, Any]],
    filepath: str,
    columns: Optional[List[str]] = None
):
    """
    Export data to CSV file.
    
    Args:
        data: List of dictionaries containing data
        filepath: Output CSV file path
        columns: Column names to include (all if None)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        print("Warning: No data to export")
        return
    
    # Determine columns
    if columns is None:
        columns = list(data[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data exported to CSV: {filepath}")


def export_to_json(
    data: Any,
    filepath: str,
    indent: int = 2
):
    """
    Export data to JSON file.
    
    Args:
        data: Data to export (dict, list, etc.)
        filepath: Output JSON file path
        indent: Indentation level
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    data = convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"Data exported to JSON: {filepath}")


def export_to_latex(
    data: List[Dict[str, Any]],
    filepath: str,
    caption: str = "Results",
    label: str = "tab:results",
    column_format: Optional[str] = None
):
    """
    Export data to LaTeX table.
    
    Args:
        data: List of dictionaries containing data
        filepath: Output .tex file path
        caption: Table caption
        label: Table label for references
        column_format: LaTeX column format (e.g., 'llrr')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        print("Warning: No data to export")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Determine column format
    if column_format is None:
        column_format = 'l' * len(df.columns)
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(f"\\begin{{tabular}}{{{column_format}}}")
    latex.append("\\hline")
    
    # Header
    header = " & ".join(df.columns) + " \\\\"
    latex.append(header)
    latex.append("\\hline")
    
    # Rows
    for _, row in df.iterrows():
        row_str = " & ".join(str(val) for val in row.values) + " \\\\"
        latex.append(row_str)
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Data exported to LaTeX: {filepath}")


class ResultsExporter:
    """
    Comprehensive results exporter.
    
    Handles exporting optimization results to multiple formats.
    """
    
    def __init__(self, output_dir: str = "results/exports"):
        """
        Initialize results exporter.
        
        Args:
            output_dir: Output directory for exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_benchmark_results(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        experiment_name: str
    ):
        """
        Export benchmark results in all formats.
        
        Args:
            results: Nested dict {algorithm: {problem: {metrics}}}
            experiment_name: Name of the experiment
        """
        # Flatten results for CSV
        flat_data = []
        for alg_name, problems in results.items():
            for prob_name, metrics in problems.items():
                row = {
                    'Algorithm': alg_name,
                    'Problem': prob_name,
                    **metrics
                }
                flat_data.append(row)
        
        # Export to CSV
        csv_path = self.output_dir / f"{experiment_name}_results.csv"
        export_to_csv(flat_data, str(csv_path))
        
        # Export to JSON
        json_path = self.output_dir / f"{experiment_name}_results.json"
        export_to_json(results, str(json_path))
        
        # Export to LaTeX
        latex_path = self.output_dir / f"{experiment_name}_results.tex"
        export_to_latex(
            flat_data,
            str(latex_path),
            caption=f"Optimization Results: {experiment_name}",
            label=f"tab:{experiment_name}"
        )
        
        print(f"\nAll results exported to: {self.output_dir}")
    
    def export_convergence_data(
        self,
        histories: Dict[str, List[float]],
        experiment_name: str
    ):
        """
        Export convergence histories to CSV.
        
        Args:
            histories: Dict mapping algorithm names to fitness histories
            experiment_name: Name of the experiment
        """
        # Create DataFrame with iterations as rows
        max_len = max(len(h) for h in histories.values())
        data = {'Iteration': list(range(1, max_len + 1))}
        
        for alg_name, history in histories.items():
            # Pad with last value if needed
            padded = history + [history[-1]] * (max_len - len(history))
            data[alg_name] = padded
        
        df = pd.DataFrame(data)
        
        csv_path = self.output_dir / f"{experiment_name}_convergence.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Convergence data exported to: {csv_path}")
    
    def export_statistical_tests(
        self,
        test_results: Dict[str, Any],
        experiment_name: str
    ):
        """
        Export statistical test results.
        
        Args:
            test_results: Dictionary of test results
            experiment_name: Name of the experiment
        """
        json_path = self.output_dir / f"{experiment_name}_statistical_tests.json"
        export_to_json(test_results, str(json_path))
        
        print(f"Statistical test results exported to: {json_path}")
    
    def create_summary_report(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        experiment_name: str,
        output_format: str = "markdown"
    ):
        """
        Create a summary report.
        
        Args:
            results: Benchmark results
            experiment_name: Name of the experiment
            output_format: 'markdown' or 'txt'
        """
        lines = []
        lines.append(f"# Optimization Results Summary: {experiment_name}\n")
        lines.append("=" * 70)
        lines.append("")
        
        # Overall statistics
        lines.append("## Overall Statistics\n")
        
        all_algorithms = set()
        all_problems = set()
        for alg_name, problems in results.items():
            all_algorithms.add(alg_name)
            all_problems.update(problems.keys())
        
        lines.append(f"- **Algorithms tested**: {len(all_algorithms)}")
        lines.append(f"- **Problems tested**: {len(all_problems)}")
        lines.append("")
        
        # Results by problem
        lines.append("## Results by Problem\n")
        
        for prob_name in sorted(all_problems):
            lines.append(f"### {prob_name}\n")
            lines.append("| Algorithm | Mean | Std | Best | Worst |")
            lines.append("|-----------|------|-----|------|-------|")
            
            for alg_name in sorted(all_algorithms):
                if prob_name in results.get(alg_name, {}):
                    metrics = results[alg_name][prob_name]
                    lines.append(
                        f"| {alg_name} | "
                        f"{metrics.get('mean_fitness', 0):.6f} | "
                        f"{metrics.get('std_fitness', 0):.6f} | "
                        f"{metrics.get('best_fitness', 0):.6f} | "
                        f"{metrics.get('worst_fitness', 0):.6f} |"
                    )
            
            lines.append("")
        
        # Save report
        ext = ".md" if output_format == "markdown" else ".txt"
        report_path = self.output_dir / f"{experiment_name}_summary{ext}"
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Summary report created: {report_path}")
