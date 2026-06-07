"""
Timoshenko beam PDE residual computation module
=============================

This module implements the PDE residual computation for the Timoshenko beam, a core component of the ODIL framework.
It supports residual computation for both linear and nonlinear cases, and is compatible with all discretization methods (DQ, Taylor, Spline).

Main features:
1. Residual computation for the linear Timoshenko beam equations
2. Residual computation for the nonlinear Timoshenko beam equations (von Kármán geometric nonlinearity)
3. Full compatibility with the ODIL optimization framework
4. Support for arbitrary derivative matrix formats

This module is independent of the specific discretization method, requiring only the first- and second-order derivative matrices as input.
"""

import torch
from typing import Dict, Tuple


class TimoshenkoBeamResiduals:
    """
    Timoshenko beam PDE residual computation class

    Used to compute the governing equation residuals of the Timoshenko beam, supporting both linear and nonlinear cases.
    """
    
    def __init__(self, material_params: Dict, q: float = 0.0, k1: float = 0.0, k2: float = 0.0):
        """
        Initialize the residual calculator

        Parameters:
            material_params: material parameter dictionary, containing a11, b11, d11, a55, lambda_val, n_xT, etc.
            q: uniformly distributed load
            k1: Winkler foundation stiffness (dimensionless)
            k2: Pasternak foundation stiffness (dimensionless)
        """
        self.a11 = material_params['a11']
        self.b11 = material_params['b11']
        self.d11 = material_params['d11']
        self.a55 = material_params['a55']
        self.lambda_val = material_params['lambda_val']
        self.n_xT = material_params['n_xT']
        self.m_xT = material_params.get('m_xT', 0.0)
        self.q = q
        self.k1 = k1  # Winkler foundation stiffness
        self.k2 = k2  # Pasternak foundation stiffness
    
    def compute_linear(self, u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
                      A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the linear PDE residual (with elastic foundation)

        Parameters:
            u, w, phi: axial displacement, transverse displacement, rotation
            A, B: first- and second-order derivative matrices

        Returns:
            R1, R2, R3: the residuals of the three PDEs
        """
        # Compute derivatives
        ux = A @ u      # du/dx
        wx = A @ w      # dw/dx
        phix = A @ phi  # dphi/dx

        uxx = B @ u     # d2u/dx2
        wxx = B @ w     # d2w/dx2
        phixx = B @ phi # d2phi/dx2

        # Determine whether in interior-node solving mode (e.g. spline natural BC)
        interior_mode = uxx.shape[0] != u.shape[0]

        # The spline interior operator returns an (N-2)×N derivative matrix, so the original field variables must be sliced to the interior nodes
        if interior_mode:
            field_slice = slice(1, -1)
            u_field = u[field_slice]
            w_field = w[field_slice]
            phi_field = phi[field_slice]
        else:
            u_field = u
            w_field = w
            phi_field = phi

        # Linear equation residuals (dimensionless FSDT PDE with elastic foundation)

        # Equation 1: a11*u_xx + b11*phi_xx = 0
        R1 = self.a11 * uxx + self.b11 * phixx

        # Equation 2: a55*(w_xx + λ*phi_x) + q - n_xT*w_xx - k1*w + k2*w_xx
        R2 = self.a55 * (wxx + self.lambda_val * phix) + self.q
        # Thermal stress term: -n_xT * ∂²w/∂x²
        R2 = R2 - self.n_xT * wxx
        # Add elastic foundation terms
        R2 = R2 - self.k1 * w_field + self.k2 * wxx

        # Equation 3: b11*u_xx + d11*phi_xx - a55*λ*(w_x + λ*phi) = 0
        R3 = self.b11 * uxx + self.d11 * phixx - self.a55 * self.lambda_val * (wx + self.lambda_val * phi_field)

        return R1, R2, R3
    
    def compute_nonlinear(self, u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
                         A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the nonlinear PDE residual (von Kármán geometric nonlinearity + elastic foundation)

        Parameters:
            u, w, phi: axial displacement, transverse displacement, rotation
            A, B: first- and second-order derivative matrices

        Returns:
            R1, R2, R3: the residuals of the three PDEs
        """
        # Compute derivatives
        ux = A @ u
        wx = A @ w
        phix = A @ phi

        uxx = B @ u
        wxx = B @ w
        phixx = B @ phi

        # Determine whether in interior-node solving mode
        interior_mode = uxx.shape[0] != u.shape[0]

        if interior_mode:
            field_slice = slice(1, -1)
            u_field = u[field_slice]
            w_field = w[field_slice]
            phi_field = phi[field_slice]
        else:
            u_field = u
            w_field = w
            phi_field = phi

        # Nonlinear equation residuals (dimensionless FSDT PDE with elastic foundation)

        # Equation 1: a11*u_xx + b11*phi_xx + (a11/λ)*w_x*w_xx = 0
        R1 = self.a11 * uxx + self.b11 * phixx + (self.a11 / self.lambda_val) * wx * wxx

        # Equation 2: the full nonlinear equation (von Kármán)
        # Base von Kármán nonlinear terms
        term_a = (self.a11 / self.lambda_val) * (
            uxx * wx + ux * wxx + (3.0 / (2.0 * self.lambda_val)) * wxx * (wx ** 2)
        )
        term_b = (self.b11 / self.lambda_val) * (phixx * wx + phix * wxx)

        R2 = term_a + term_b + self.a55 * (wxx + self.lambda_val * phix) + self.q
        # Thermal stress term: -n_xT * ∂²w/∂x²
        R2 = R2 - self.n_xT * wxx
        # Add elastic foundation terms
        R2 = R2 - self.k1 * w_field + self.k2 * wxx

        # Equation 3: b11*u_xx + d11*phi_xx + (b11/λ)*w_x*w_xx - a55*λ*(w_x + λ*phi) = 0
        R3 = self.b11 * uxx + self.d11 * phixx + (self.b11 / self.lambda_val) * wx * wxx - \
             self.a55 * self.lambda_val * (wx + self.lambda_val * phi_field)

        return R1, R2, R3
    
    def compute(self, u: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
               A: torch.Tensor, B: torch.Tensor, is_nonlinear: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the PDE residual (automatically selecting linear or nonlinear)

        Parameters:
            u, w, phi: displacements and rotation
            A, B: derivative matrices
            is_nonlinear: whether to use the nonlinear formulation

        Returns:
            R1, R2, R3: the residuals of the three PDEs
        """
        if is_nonlinear:
            return self.compute_nonlinear(u, w, phi, A, B)
        else:
            return self.compute_linear(u, w, phi, A, B)
