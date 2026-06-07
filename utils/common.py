"""
General-purpose utility functions module
Contains common functionality such as device configuration and random seed setup
"""

import torch
import random
import numpy as np
from typing import Optional


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the computing device

    Parameters:
        use_cuda: whether to use CUDA (if available)

    Returns:
        a torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for computation")
    return device


def setup_torch_defaults(device: Optional[torch.device] = None, 
                        dtype: torch.dtype = torch.float64):
    """
    Set the PyTorch default device and data type

    Parameters:
        device: computing device, automatically selected if None
        dtype: data type, double precision by default
    """
    if device is None:
        device = get_device()
    
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    
    return device


def set_seed(seed: int = 42):
    """
    Set all random seeds to ensure reproducible results

    Parameters:
        seed: random seed value
    """
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        # Ensure deterministic behavior of CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed has been set to: {seed}")


# Convenience initialization function
def initialize_computing_environment(seed: Optional[int] = None,
                                    use_cuda: bool = True,
                                    dtype: torch.dtype = torch.float64,
                                    verbose: bool = True):
    """
    One-click initialization of the computing environment

    Parameters:
        seed: random seed, not set if None
        use_cuda: whether to use CUDA
        dtype: data type
        verbose: whether to print information

    Returns:
        a device object
    """
    # Set the device and default type
    device = get_device(use_cuda) if verbose else \
             torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    setup_torch_defaults(device, dtype)

    # Set the random seed
    if seed is not None:
        set_seed(seed) if verbose else _set_seed_quiet(seed)

    if verbose:
        print(f"Computing environment initialized: device={device}, dtype={dtype}")

    return device


def _set_seed_quiet(seed: int):
    """Set the random seed silently (without printing information)"""
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False