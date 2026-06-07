"""
Tensor operation utility functions

Provides common tensor operations and helper functions.
"""

import torch
from typing import Optional, List, Dict


def create_boundary_mask(N: int, device: torch.device, 
                        dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Create boundary mask (exclude boundary points)

    Parameters:
        N: number of nodes
        device: computation device
        dtype: data type

    Returns:
        boundary mask tensor
    """
    mask = torch.ones(N, device=device, dtype=dtype)
    mask[0] = 0.0   # exclude left boundary
    mask[-1] = 0.0  # exclude right boundary
    return mask


def create_optimizer(params: List[torch.Tensor], optim_name: str, lr: float, 
                    max_iter: Optional[int] = None) -> torch.optim.Optimizer:
    """
    Unified interface for creating an optimizer

    Parameters:
        params: parameter list
        optim_name: optimizer name
        lr: learning rate
        max_iter: maximum number of iterations (for L-BFGS)

    Returns:
        optimizer instance
    """
    optim_name = optim_name.lower()
    
    if optim_name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif optim_name == "lbfgs":
        return torch.optim.LBFGS(
            params, 
            lr=lr, 
            max_iter=max_iter if max_iter else 200,
            line_search_fn="strong_wolfe",
            history_size=200,
            tolerance_grad=1e-16,
            tolerance_change=1e-16
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")


def apply_gradient_clipping(params: List[torch.Tensor], max_norm: float = 100.0) -> float:
    """
    Apply gradient clipping

    Parameters:
        params: parameter list
        max_norm: maximum gradient norm

    Returns:
        gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def initialize_parameters(N: int, bc_type: str, device: torch.device, 
                         initial_guess: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """
    Initialize optimization parameters

    Parameters:
        N: number of nodes
        bc_type: boundary condition type
        device: computation device
        initial_guess: initial guess

    Returns:
        parameter dictionary
    """
    if bc_type == 'C-C':
        # Hard constraint: interior node parameters
        u_param = torch.zeros(size=(N-2,), device=device, dtype=torch.float64, requires_grad=True)
        w_param = torch.zeros(size=(N-2,), device=device, dtype=torch.float64, requires_grad=True)
        phi_param = torch.zeros(size=(N-2,), device=device, dtype=torch.float64, requires_grad=True)
    else:
        # Soft constraint: all-node parameters
        u_param = torch.zeros(size=(N,), device=device, dtype=torch.float64, requires_grad=True)
        w_param = torch.zeros(size=(N,), device=device, dtype=torch.float64, requires_grad=True)
        phi_param = torch.zeros(size=(N,), device=device, dtype=torch.float64, requires_grad=True)
    
    # Add initialization
    with torch.no_grad():
        if initial_guess is not None and bc_type == 'C-C':
            # Use initial guess (C-C boundary only). Clone so the parameters own
            # their storage; otherwise in-place optimizer updates would write
            # back into the caller's initial_guess and contaminate reuse of it.
            u_param.data = initial_guess.get('u_inner', torch.randn_like(u_param) * 0.01).clone()
            w_param.data = initial_guess.get('w_inner', torch.randn_like(w_param) * 0.01).clone()
            phi_param.data = initial_guess.get('phi_inner', torch.randn_like(phi_param) * 0.01).clone()
        else:
            # Random initialization
            u_param.data = torch.randn_like(u_param) * 0.01
            w_param.data = torch.randn_like(w_param) * 0.01
            phi_param.data = torch.randn_like(phi_param) * 0.01
    
    return {
        'u_param': u_param,
        'w_param': w_param,
        'phi_param': phi_param
    }