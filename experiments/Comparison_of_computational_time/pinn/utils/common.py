"""
General-purpose utility functions module for the PINN

Author: Yang
Version: 1.0

Responsibilities:
- Random seed setting: ensure full reproducibility of PINN training experiments
- Compute device selection: automatically detect and select the optimal CUDA/CPU compute device
- Experiment configuration display: format and print training parameters and system information
- Numerical computation helpers: provide commonly used mathematical computation and validation functions

Core features:
- Cross-library seed synchronization: uniformly set the random seeds of Python random, NumPy, and PyTorch
- GPU environment detection: intelligently detect CUDA availability and select the optimal device
- Experiment management: facilitate parameter recording, result reproduction, and performance comparison of research experiments
- Error handling: includes complete exception handling and state validation mechanisms

Technical implementation:
- Ensure consistency of PINN network initialization
- Ensure numerical stability of automatic differentiation computations
- Support device selection strategies for multi-GPU environments
- Provide configuration information tracking during training

Application scenarios:
- Environment configuration before Timoshenko beam PINN training
- Unified parameter setting in sensitivity analysis
- Result consistency assurance across multiple experiment runs
- Foundational tools for debugging and performance analysis
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set the random seed to ensure experiment reproducibility

    Description:
    Synchronously set the random seeds of Python, NumPy, PyTorch, and CUDA, ensuring that
    each run of an experiment with the same parameters yields fully identical results.

    Technical points:
    - Python random module: controls Python built-in random functions
    - NumPy random generation: affects all functions in numpy.random
    - PyTorch CPU randomness: controls the randomness of tensor operations on the CPU
    - PyTorch CUDA randomness: controls all random operations on the GPU
    - CuDNN determinism: disable non-deterministic algorithms to ensure consistent GPU computation

    Research significance:
    For PINN training, a deterministic random seed is the key to reproducible results,
    especially in parameter sensitivity analysis, where the influence of randomness on the results must be excluded.

    Parameters:
    - seed: random seed value, default 42 (a commonly used research-experiment seed)
    """
    random.seed(seed)                    # Python built-in random functions
    np.random.seed(seed)                 # NumPy random number generator
    torch.manual_seed(seed)              # PyTorch CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # current GPU device random seed
        torch.cuda.manual_seed_all(seed)    # all GPU devices random seed
        torch.backends.cudnn.deterministic = True   # CuDNN deterministic mode
        torch.backends.cudnn.benchmark = False      # disable performance optimization (affects determinism)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Automatically detect and return the optimal compute device

    Description:
    Detect whether the system supports CUDA, prefer the GPU for accelerated computation,
    and automatically fall back to CPU mode if the GPU is unavailable.

    Device selection strategy:
    - Priority: CUDA GPU > CPU
    - Automatic detection: no manual configuration required
    - Graceful degradation: automatically use the CPU when the GPU is unavailable

    Returns:
    - torch.device: 'cuda' (if available) or 'cpu'

    Application scenarios:
    PINN training typically involves a large amount of automatic differentiation computation; GPU acceleration can
    significantly speed up training, especially for large-scale parameter-sweep tasks.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_config(params_dict: dict):
    """Format and print configuration parameters

    Description:
    Display experiment configuration parameters in a clean tabular form, for easy experiment recording and result traceability.

    Output format:
    ============================================================
    Config:
    ------------------------------------------------------------
      parameter name: parameter value
      ...
    ============================================================

    Parameters:
    - params_dict: configuration parameter dictionary, with keys as parameter names and values as parameter values

    Research value:
    Detailed parameter records are an important part of research experiments, and help with:
    - Reproducibility of experiment results
    - Comparison in parameter sensitivity analysis
    - Writing experiment reports and papers
    """
    print("=" * 60)
    print("Configuration:")
    print("-" * 60)
    for key, value in params_dict.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def parse_activation_from_path(path_or_name: str) -> Dict[str, any]:
    """Parse activation function parameters from a file name or path

    Description:
    According to the project's naming convention, parse the activation function type
    and its related parameters (such as SIREN's omega parameters) from the model file name or folder path.

    Naming convention:
    - Tanh activation: file name contains '_Tanh' or '-Tanh'
    - Sin activation: file name contains '_Sin' or '-Sin'
    - SIREN activation: file name contains '_SIREN_w{omega0}_{omegah}' or '-SIREN_w{omega0}_{omegah}'
      For example: '_SIREN_w30.0_30.0' denotes omega_0=30.0, omega_hidden=30.0
    - If not specified in the file name, Tanh is used by default

    Parameters:
    - path_or_name: file path or file name string

    Returns:
    - Dict: a dictionary containing the following keys:
        - 'activation_type': str, activation function type ('Tanh', 'Sin', 'SIREN')
        - 'siren_omega_0': float, SIREN first-layer frequency factor (only valid for SIREN)
        - 'siren_omega_hidden': float, SIREN hidden-layer frequency factor (only valid for SIREN)

    Usage example:
    >>> parse_activation_from_path("Linearw_W_0.025_T_300_H_0.8_qn0.08_Tanh.pth")
    {'activation_type': 'Tanh', 'siren_omega_0': 30.0, 'siren_omega_hidden': 30.0}

    >>> parse_activation_from_path("model_SIREN_w30.0_15.0.pth")
    {'activation_type': 'SIREN', 'siren_omega_0': 30.0, 'siren_omega_hidden': 15.0}
    """
    import re
    from pathlib import Path

    # Default parameters
    result = {
        'activation_type': 'Tanh',
        'siren_omega_0': 30.0,
        'siren_omega_hidden': 30.0,
    }

    # Convert the path to a string for searching
    path_str = str(path_or_name)

    # Try to match the SIREN format: _SIREN_w{omega0}_{omegah} or -SIREN_w{omega0}_{omegah}
    siren_pattern = r'[_-]SIREN_w([\d.]+)_([\d.]+)'
    siren_match = re.search(siren_pattern, path_str)
    if siren_match:
        result['activation_type'] = 'SIREN'
        result['siren_omega_0'] = float(siren_match.group(1))
        result['siren_omega_hidden'] = float(siren_match.group(2))
        return result

    # Try to match the simple SIREN format (no parameters, use default omega)
    if re.search(r'[_-]SIREN(?![_\w])', path_str):
        result['activation_type'] = 'SIREN'
        return result

    # Try to match the Sin activation function
    if re.search(r'[_-]Sin(?![_\w])', path_str):
        result['activation_type'] = 'Sin'
        return result

    # Try to match the Tanh activation function (explicitly specified)
    if re.search(r'[_-]Tanh(?![_\w])', path_str):
        result['activation_type'] = 'Tanh'
        return result

    # No activation function marker found, use the default value Tanh
    # Note: file names in the old format may not contain activation function information
    return result


def max_deflection(w: np.ndarray) -> float:
    """Compute the maximum absolute value of the deflection field

    Description:
    Compute the maximum absolute deformation of the Timoshenko beam deflection field w(x). This is an
    important indicator in structural analysis, used to evaluate the degree of beam deformation and its safety.

    Physical meaning:
    - The maximum deflection reflects the overall deformation level of the beam
    - It is a key constraint indicator in structural design
    - It is used to compare the difference between linear and nonlinear solutions

    Parameters:
    - w: deflection array, typically the discrete point values along the beam length direction

    Returns:
    - float: maximum absolute deflection value max |w(x)|

    Application:
    Commonly used in sensitivity analysis to compare the structural response under different parameter combinations.
    """
    return float(np.abs(w).max())


def safe_mkdir(path: str | Path) -> None:
    """Create a directory with support for Windows long paths

    Description:
    Safely create a directory, automatically handling the Windows long-path limit (260 characters).
    If the path length exceeds the limit, the '\\\\?\\' prefix is added automatically.

    Parameters:
    - path: directory path (string or Path object)
    """
    import os
    from pathlib import Path

    path_str = str(Path(path).resolve())
    if os.name == 'nt' and len(path_str) > 200:
        if not path_str.startswith("\\\\?\\"):
            path_str = "\\\\?\\" + path_str

    os.makedirs(path_str, exist_ok=True)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R-squared coefficient of determination

    Description:
    Compute the R-squared coefficient of determination between predicted and true values,
    used to evaluate the accuracy of the model's predictions.

    Mathematical formula:
        R^2 = 1 - SS_res / SS_tot
        where:
        - SS_res = sum((y_true - y_pred)^2)  (residual sum of squares)
        - SS_tot = sum((y_true - y_mean)^2)  (total sum of squares)

    Parameters:
    - y_true: array of true values
    - y_pred: array of predicted values

    Returns:
    - float: R-squared value, typically in the range [0, 1]; the closer to 1, the more accurate the prediction

    Notes:
    - If the input contains non-finite values (NaN/Inf), return NaN
    - If the true values are almost constant (SS_tot is approximately 0), return 1.0 or 0.0
    """
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        return float('nan')
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    
    return float(1.0 - ss_res / ss_tot)
