"""
Solver utility functions module

This module contains common utility functions for the solver.
"""

import torch


def inject_dirichlet(u_inner: torch.Tensor, bc0: float, bc1: float) -> torch.Tensor:
    """
    Combine interior node values with boundary conditions into a complete solution (hard constraint)

    Parameters:
        u_inner: [N-2] interior node values
        bc0: left boundary value (x=0)
        bc1: right boundary value (x=L)

    Returns:
        u: [N] complete solution

    Example:
        >>> u_inner = torch.tensor([1.0, 2.0, 3.0])  # 3 interior nodes
        >>> u = inject_dirichlet(u_inner, 0.0, 0.0)  # boundary value is 0
        >>> u  # tensor([0.0, 1.0, 2.0, 3.0, 0.0])  # 5 nodes
    """
    N = u_inner.numel() + 2
    u = torch.empty(N, dtype=u_inner.dtype, device=u_inner.device)
    u[0] = bc0   # left boundary
    u[-1] = bc1  # right boundary
    u[1:-1] = u_inner  # interior nodes
    return u


def compute_moment(ux: torch.Tensor, phix: torch.Tensor, wx: torch.Tensor,
                  b11: float, d11: float, lambda_val: float,
                  index: int, is_nonlinear: bool = False,
                  m_xT: float = 0.0) -> torch.Tensor:
    """
    Compute the bending moment at a specified location (including the thermal stress term)

    Parameters:
        ux, phix, wx: first-order derivatives
        b11, d11: material parameters
        lambda_val: slenderness ratio parameter of the beam
        index: index of the computation location (0 for left end, -1 for right end)
        is_nonlinear: whether it is a nonlinear problem (default False)
        m_xT: thermal stress bending moment (default 0)

    Returns:
        M: bending moment value

    Bending moment formula:
        Linear: M = b11*du/dx + d11*dphi/dx - m_xT
        Nonlinear: M = b11*du/dx + d11*dphi/dx + (b11/2λ)*(dw/dx)² - m_xT
    """
    # Base bending moment term (sign corrected: +d11)
    M = b11 * ux[index] + d11 * phix[index]

    # Nonlinear term (nonlinear case only)
    if is_nonlinear:
        M = M + b11 / (2 * lambda_val) * wx[index]**2

    # Thermal stress term
    M = M - m_xT

    return M


