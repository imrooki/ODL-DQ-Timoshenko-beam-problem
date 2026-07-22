

import torch


def inject_dirichlet(u_inner: torch.Tensor, bc0: float, bc1: float) -> torch.Tensor:
    
    N = u_inner.numel() + 2
    u = torch.empty(N, dtype=u_inner.dtype, device=u_inner.device)
    u[0] = bc0   
    u[-1] = bc1  
    u[1:-1] = u_inner  
    return u


def compute_moment(ux: torch.Tensor, phix: torch.Tensor, wx: torch.Tensor,
                  b11: float, d11: float, lambda_val: float,
                  index: int, is_nonlinear: bool = False,
                  m_xT: float = 0.0) -> torch.Tensor:
    
    M = b11 * ux[index] + d11 * phix[index]

    if is_nonlinear:
        M = M + b11 / (2 * lambda_val) * wx[index]**2

    
    M = M - m_xT

    return M


