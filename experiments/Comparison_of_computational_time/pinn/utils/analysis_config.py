#!/usr/bin/env python3
"""
Parameter sensitivity analysis configuration module - Energy-based Timoshenko Beam PINNs

Author: Yang
Creation date: 2024-09-22
Version: 1.0

This module defines the configuration class for parameter sensitivity analysis,
providing unified management of analysis parameters, boundary conditions, and
output settings. It avoids duplicating the same configuration logic across
different analysis scripts.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class AnalysisConfig:
    """
    Parameter sensitivity analysis configuration class

    Provides unified management of all analysis-related configuration parameters,
    including baseline parameters, sweep ranges, boundary conditions, distribution
    types, and parallel configuration.

    Attributes:
        baseline_params: Baseline parameter configuration dictionary
        parameter_ranges: Parameter sweep range dictionary
        boundary_conditions: List of boundary condition types
        distribution_types: List of material distribution types
        base_output_dir: Output root directory
        script_name: Script name, used for output directory hierarchy
        max_workers: Number of parallel worker processes (parallel version only)
        use_gpu: Whether to use GPU acceleration (parallel version only)
        gpu_memory_fraction: GPU memory usage fraction (parallel version only)
        resource_limit: System resource usage fraction limit (parallel version only)
        min_free_memory_gb: Minimum free memory to reserve (parallel version only)
        min_free_cores: Minimum number of free cores to reserve (parallel version only)
        enable_resource_monitor: Whether to enable resource monitoring (parallel version only)
        cpu_threshold: CPU utilization threshold (parallel version only)
        memory_threshold: Memory utilization threshold (parallel version only)
    """

    # Basic configuration parameters
    baseline_params: Dict[str, float] = None
    parameter_ranges: Dict[str, List[float]] = None
    boundary_conditions: List[str] = None
    distribution_types: List[str] = None
    base_output_dir: str = "results"
    script_name: str = "sensitivity_analysis"

    # Parallel configuration parameters (parallel version only)
    max_workers: Optional[int] = None
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    resource_limit: float = 0.7
    min_free_memory_gb: float = 4.0
    min_free_cores: int = 2
    enable_resource_monitor: bool = True
    cpu_threshold: int = 80
    memory_threshold: int = 85

    def __post_init__(self):
        """Initialize the default configuration parameters"""

        # Set the default baseline parameters
        if self.baseline_params is None:
            self.baseline_params = {
                'W_Gr': 0.025,      # Graphene mass fraction
                'H_Gr': 0.8,        # Graphene shape factor
                'T': 300,           # Temperature (K)
                'q': -0.08,         # Dimensionless distributed load
                'L_factor': 20,     # Beam length factor (L/h)
                'h': 0.1,           # Beam thickness (m) - fixed
                'num_layers': 10,   # Number of material layers - fixed
                # Elastic foundation parameters
                'k1': 0.0,          # Elastic foundation Winkler stiffness coefficient
                'k2': 0.0,          # Elastic foundation Pasternak stiffness coefficient
                # Boundary conditions
                'bc_type': 'C-C',   # Boundary condition type
                'distribution': 'X' # Material distribution type
            }

        # Set the default parameter sweep ranges
        if self.parameter_ranges is None:
            self.parameter_ranges = {
                'W_Gr': [0, 0.005, 0.010, 0.015, 0.020, 0.025],  # 6 values
                'H_Gr': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 11 values
                'T': [300, 325, 350, 375, 400],  # 5 values
                'q': [0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12, -0.13, -0.14, -0.15, -0.16, -0.17, -0.18, -0.19, -0.20],  # 21 values
                'L_factor': [10, 20, 30, 40, 50],  # 5 values, L/h ratio
                'k1': [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],  # 11 values
                'k2': [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]   # 11 values
            }

        # Set the default boundary conditions
        if self.boundary_conditions is None:
            self.boundary_conditions = ['C-C', 'C-H', 'H-H', 'S-S', 'C-F']  # Five boundary conditions

        # Set the default distribution types
        if self.distribution_types is None:
            self.distribution_types = ['X', 'U', 'O']  # Three distribution types

        # Set the default parallel configuration (only when max_workers is None)
        if self.max_workers is None:
            self.max_workers = max(1, int(mp.cpu_count() * 0.7))

    def get_total_cases(self) -> int:
        """
        Compute the total number of analysis cases

        Returns:
            int: Total number of cases = number of boundary conditions x number of distribution types x sum of all parameter values
        """
        total_params = sum(len(values) for values in self.parameter_ranges.values())
        total_combinations = len(self.boundary_conditions) * len(self.distribution_types)
        return total_params * total_combinations

    def validate(self) -> bool:
        """
        Validate the validity of the configuration parameters

        Returns:
            bool: Whether the configuration is valid
        """
        # Check the basic parameters
        if not self.baseline_params or not self.parameter_ranges:
            return False

        # Check the boundary conditions and distribution types
        if not self.boundary_conditions or not self.distribution_types:
            return False

        # Check the parallel configuration
        if self.max_workers is not None and self.max_workers < 1:
            return False

        return True

    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("Parameter Sensitivity Analysis Configuration Summary")
        print("="*60)
        print(f"Script name: {self.script_name}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"Boundary conditions: {len(self.boundary_conditions)} types {self.boundary_conditions}")
        print(f"Distribution types: {len(self.distribution_types)} types {self.distribution_types}")
        print(f"Analysis parameters: {len(self.parameter_ranges)} params {list(self.parameter_ranges.keys())}")
        print(f"Total cases: {self.get_total_cases()}")

        if self.max_workers is not None:
            print(f"\nParallel configuration:")
            print(f"  Workers: {self.max_workers}")
            print(f"  Resource limit: {self.resource_limit:.1%}")
            print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
            if self.enable_resource_monitor:
                print(f"  Resource monitor: Enabled (CPU<{self.cpu_threshold}%, Memory<{self.memory_threshold}%)")

        print("="*60)


# Parameter label mapping constants (avoid duplicate definitions)
PARAM_LABELS = {
    'W_Gr': ('Graphene Mass Fraction W_Gr', 'W_Gr'),
    'H_Gr': ('Graphene Shape Factor H_Gr', 'H_Gr'),
    'T': ('Temperature (K)', 'Temperature'),
    'q': ('Load Magnitude |q|', 'Load'),
    'L_factor': ('Length Ratio L/h', 'Length Ratio'),
    'k1': ('Winkler Foundation Stiffness k1', 'k1'),
    'k2': ('Pasternak Foundation Stiffness k2', 'k2')
}


def create_standard_config(script_name: str = "sensitivity_analysis",
                          enable_parallel: bool = False) -> AnalysisConfig:
    """
    Create a standard configuration instance

    Args:
        script_name: Script name, used for the output directory
        enable_parallel: Whether to enable the parallel configuration

    Returns:
        AnalysisConfig: Configuration instance
    """
    config = AnalysisConfig(script_name=script_name)

    if not enable_parallel:
        # The serial version does not need the parallel configuration
        config.max_workers = None
        config.enable_resource_monitor = False

    return config
