

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_torch_tensor(value: Any) -> bool:
    
    return torch is not None and isinstance(value, torch.Tensor)


def _zeros(shape: Sequence[int], *, like: Any = None, device=None, dtype=None):
    
    if _is_torch_tensor(like):
        return torch.zeros(*shape, device=like.device if device is None else device, dtype=like.dtype if dtype is None else dtype)
    if torch is not None and device is not None:
        return torch.zeros(*shape, device=device, dtype=torch.float64 if dtype is None else dtype)
    return np.zeros(shape, dtype=np.float64 if dtype is None else dtype)


def _eye(n: int, *, like: Any = None, device=None, dtype=None):
    
    if _is_torch_tensor(like):
        return torch.eye(n, device=like.device if device is None else device, dtype=like.dtype if dtype is None else dtype)
    if torch is not None and device is not None:
        return torch.eye(n, device=device, dtype=torch.float64 if dtype is None else dtype)
    return np.eye(n, dtype=np.float64 if dtype is None else dtype)


def _stack_blocks(block_rows: Sequence[Sequence[Any]]):
    
    first = block_rows[0][0]
    if _is_torch_tensor(first):
        return torch.cat([torch.cat(row, dim=1) for row in block_rows], dim=0)
    return np.block([list(row) for row in block_rows])


def assemble_linear_stiffness_matrix(
    N: int,
    a11: float,
    a55: float,
    b11: float,
    d11: float,
    n_xT: float,
    lambda_val: float,
    A,
    B,
    k1: float = 0.0,
    k2: float = 0.0,
    *,
    device=None,
    dtype=None,
):
    
    identity = _eye(N, like=A, device=device, dtype=dtype)
    zeros = _zeros((N, N), like=A, device=device, dtype=dtype)

    
    kl11 = a11 * B                                          
    kl12 = zeros
    kl13 = b11 * B                                          
    
    kl21 = zeros
    kl22 = (a55 + n_xT + k2) * B - k1 * identity
    kl23 = lambda_val * a55 * A                             
    
    kl31 = b11 * B                                          
    kl32 = -lambda_val * a55 * A                            
    kl33 = d11 * B - a55 * (lambda_val ** 2) * identity     

    return _stack_blocks(((kl11, kl12, kl13), (kl21, kl22, kl23), (kl31, kl32, kl33)))


def assemble_nonlinear_stiffness_matrix(
    N: int,
    a11: float,
    b11: float,
    lambda_val: float,
    A,
    B,
    d_vec,
    *,
    device=None,
    dtype=None,
):
    
    backend_like = A
    zeros = _zeros((N, N), like=backend_like, device=device, dtype=dtype)

    
    u = d_vec[:N]             
    w = d_vec[N : 2 * N]     
    phi = d_vec[2 * N : 3 * N]  

    
    knl11 = zeros
    knl12 = _zeros((N, N), like=backend_like, device=device, dtype=dtype)
    knl13 = zeros
    knl21 = _zeros((N, N), like=backend_like, device=device, dtype=dtype)
    knl22 = _zeros((N, N), like=backend_like, device=device, dtype=dtype)
    knl23 = _zeros((N, N), like=backend_like, device=device, dtype=dtype)
    knl31 = zeros
    knl32 = _zeros((N, N), like=backend_like, device=device, dtype=dtype)
    knl33 = zeros
    b11_sign = 1.0  

    
    for i in range(N):
        ai = A[i, :]       
        bi = B[i, :]       
        bi_w = bi @ w       
        ai_w = ai @ w       
        bi_u = bi @ u       
        ai_u = ai @ u       
        bi_phi = bi @ phi   
        ai_phi = ai @ phi   
        for j in range(N):
            aij = A[i, j]   
            bij = B[i, j]   
            
            knl12[i, j] = 0.5 * a11 / lambda_val * (aij * bi_w + ai_w * bij)
            
            knl21[i, j] = 0.5 * a11 / lambda_val * (bij * ai_w + aij * bi_w)
            
            knl22[i, j] = (
                0.5 * a11 / lambda_val * (bi_u * aij + ai_u * bij)                
                + b11_sign * 0.5 * b11 / lambda_val * (bi_phi * aij + ai_phi * bij)  
                + 0.5 * a11 / (lambda_val ** 2) * ai_w * ai_w * bij               
                + a11 / (lambda_val ** 2) * bi_w * ai_w * aij                      
            )
            
            knl23[i, j] = b11_sign * 0.5 * b11 / lambda_val * (bij * ai_w + aij * bi_w)
            
            knl32[i, j] = b11_sign * 0.5 * b11 / lambda_val * (bij * ai_w + aij * bi_w)

    return _stack_blocks(((knl11, knl12, knl13), (knl21, knl22, knl23), (knl31, knl32, knl33)))


def get_boundary_dofs(bc_type: str, N: int) -> List[int]:
    
    if bc_type == "C-C":
        return [0, N - 1, N, 2 * N - 1, 2 * N, 3 * N - 1]
    if bc_type in {"H-H", "S-S"}:
        return [0, N - 1, N, 2 * N - 1]
    if bc_type in {"C-H", "C-S"}:
        return [0, N - 1, N, 2 * N - 1, 2 * N]
    if bc_type == "C-F":
        return [0, N, 2 * N]
    raise ValueError(f"Unsupported boundary type: {bc_type}")
