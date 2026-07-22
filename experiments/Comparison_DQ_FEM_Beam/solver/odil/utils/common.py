

import torch
import random
import numpy as np
from typing import Optional


def get_device(use_cuda: bool = True) -> torch.device:
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for computation")
    return device


def setup_torch_defaults(device: Optional[torch.device] = None, 
                        dtype: torch.dtype = torch.float64):
    
    if device is None:
        device = get_device()
    
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    
    return device


def set_seed(seed: int = 42):
    
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed has been set to: {seed}")



def initialize_computing_environment(seed: Optional[int] = None,
                                    use_cuda: bool = True,
                                    dtype: torch.dtype = torch.float64,
                                    verbose: bool = True):
    
    
    device = get_device(use_cuda) if verbose else \
             torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    setup_torch_defaults(device, dtype)

    
    if seed is not None:
        set_seed(seed) if verbose else _set_seed_quiet(seed)

    if verbose:
        print(f"Computing environment initialized: device={device}, dtype={dtype}")

    return device


def _set_seed_quiet(seed: int):
    
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False