"""
Timoshenko beam PINNs boundary condition handling module

Author: Yang
Version: 1.0

Responsibilities:
- Implement hard-constraint lifting functions for Dirichlet boundary conditions, hard-coding endpoint values into the network output
- Provide a boundary condition penalty term (BoundaryConditionPenalty), imposing u/w/φ/M constraints at the endpoints
- Support multiple boundary condition types: combinations such as C-C, S-S, H-H, C-S, C-H
- Implement a mixed boundary handling strategy combining hard and soft constraints

======================================================================
                        Boundary condition handling strategy overview
======================================================================
┌─────────┬───────────────────────┬──────────────────────┬────────────┐
│ BC type │ Hard (Lifting func.)  │ Soft (penalty method)│ Physical   │
│         │                       │                      │ meaning    │
├─────────┼───────────────────────┼──────────────────────┼────────────┤
│ C-C     │ u=w=φ=0 (both ends)  │ none                 │ fully fixed│
│         │ fully hard-coded      │                      │ (clamped)  │
├─────────┼───────────────────────┼──────────────────────┼────────────┤
│ S-S     │ u=w=0 (both ends)    │ M=0 (both ends)     │ simply     │
│         │ displ. hard-coded     │ moment soft constr.  │ supported  │
├─────────┼───────────────────────┼──────────────────────┼────────────┤
│ H-H     │ u=w=0 (both ends)    │ M=0 (both ends)     │ hinged     │
│         │ same as S-S           │ same as S-S          │ (= simple) │
├─────────┼───────────────────────┼──────────────────────┼────────────┤
│ C-S     │ left:u=w=φ=0         │ M=0 (right end)     │ clamped-   │
│         │ right:u=w=0           │                      │ simple mix │
├─────────┼───────────────────────┼──────────────────────┼────────────┤
│ C-H     │ left:u=w=φ=0         │ M=0 (right end)     │ clamped-   │
│         │ right:u=w=0           │                      │ hinged mix │
└─────────┴───────────────────────┴──────────────────────┴────────────┘

Handling flow:
1. Hard constraint: the lifting() function analytically satisfies the displacement boundary conditions, ensuring exactness
2. Soft constraint: the BoundaryConditionPenalty class computes a penalty term measuring the degree of violation
3. Combination: total loss = energy loss + boundary weight × penalty loss

Key points:
- The boundary loss is evaluated by sampling only at the endpoints (x=0, 1), and combined with a weight into the total loss.
- Natural boundaries (such as M=0 at a simply-supported end) are implemented via penalty terms; derivatives are obtained from autograd.
======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

# Compatibility for running the script directly: fall back to absolute imports when relative imports fail
try:
    from .data_types import BoundaryConditions, BoundaryConditionType, MaterialCoeffs, PhysicalParams
    from .numerics import safe_divide, as_shape
except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from modules.data_types import BoundaryConditions, BoundaryConditionType, MaterialCoeffs, PhysicalParams
    from modules.numerics import safe_divide, as_shape



def lifting(
    x_norm: torch.Tensor, raw_u: torch.Tensor, raw_w: torch.Tensor, raw_phi: torch.Tensor, bc: BoundaryConditions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dirichlet boundary condition lifting function (polynomial/power-function type - poly)

    Description:
    Transforms the raw output of the neural network into a displacement field that satisfies the Dirichlet boundary conditions.
    This is an analytical boundary condition handling method that ensures the network output automatically satisfies the given endpoint displacement constraints.

    Mathematical principle and concrete examples:
    For given boundary values, a constraint-satisfying solution is constructed using interpolation and correction terms:

    1. Both ends specified (e.g. C-C boundary condition, u=w=φ=0):
       v(x) = vL + x*(vR-vL) + x*(1-x)*v̂(x)

       Example: clamped-clamped beam, u(0)=u(1)=0
       - vL = 0, vR = 0
       - u(x) = 0 + x*(0-0) + x*(1-x)*û(x) = x*(1-x)*û(x)
       - Verification: u(0) = 0*(1-0)*û(0) = 0
       - Verification: u(1) = 1*(1-1)*û(1) = 0
       - Feature: the x*(1-x) term is 0 at the endpoints, ensuring the boundary conditions are satisfied exactly

    2. One end specified (e.g. left end clamped, right end free):
       - Left end specified: v(x) = vL + x*v̂(x)
       - Right end specified: v(x) = vR + (1-x)*v̂(x)

       Example: cantilever beam, u(0)=0, u(1)=free
       - vL = 0, vR = None
       - u(x) = 0 + x*û(x) = x*û(x)
       - Verification: u(0) = 0*û(0) = 0
       - Verification: u(1) = 1*û(1) = û(1) (learned by the network)

    3. Both ends free (e.g. the rotation φ of a simply-supported beam):
       v(x) = v̂(x)

       Example: rotation of a simply-supported beam, φ(0)=φ(1)=free
       - vL = None, vR = None
       - φ(x) = φ̂(x) (determined entirely by network learning and energy minimization)

    Technical advantages:
    - Hard constraint: the boundary conditions are satisfied analytically, independently of the optimization process
    - Reduced penalty: lowers the weight requirement of the boundary condition penalty function
    - Improved convergence: the network focuses on learning the structure of the interior solution
    - Numerical stability: avoids numerical problems caused by an overly large penalty weight

    Mathematical verification:
    For any combination of boundary conditions, the lifting function guarantees:
    - If vL ≠ None, then v(0) = vL (left-end constraint satisfied)
    - If vR ≠ None, then v(1) = vR (right-end constraint satisfied)
    - The interior solution is determined jointly by the network output v̂(x) and the energy variational principle

    Parameters:
    - x_norm: normalized coordinate tensor, range [0,1], shape=(N,1)
    - raw_u/raw_w/raw_phi: raw outputs of the neural network, corresponding to the three displacement components u, w, φ
    - bc: boundary condition specification, containing the Dirichlet value at each endpoint (None means free)

    Returns:
    - (u, w, phi): tuple of lifted displacement fields satisfying the boundary conditions
    """

    xi = x_norm

    def lift_one(vhat: torch.Tensor, vL: Optional[float], vR: Optional[float]) -> torch.Tensor:
        """Boundary condition lifting for a single physical quantity

        Lifting strategy:
        1. Both-end constraint: linear interpolation base + network correction term
        2. Single-end constraint: fixed boundary + network contribution at the free end
        3. No constraint: use the network output directly
        """
        if vL is not None and vR is not None:
            # Both ends specified: v = vL + xi*(vR-vL) + xi*(1-xi)*vhat
            # Linear interpolation ensures the endpoint values; the xi*(1-xi) term is 0 at the endpoints and does not affect the boundary conditions
            return vL + xi * (vR - vL) + xi * (1.0 - xi) * vhat
        elif vL is not None and vR is None:
            # Only left end specified: v = vL + xi*vhat (ensures v(0)=vL, right end learned by the network)
            return vL + xi * vhat
        elif vL is None and vR is not None:
            # Only right end specified: v = vR + (1-xi)*vhat (ensures v(1)=vR, left end learned by the network)
            return vR + (1.0 - xi) * vhat
        else:
            # Both ends free: use the network output directly, constrained by energy minimization or penalty terms
            return vhat

    u = lift_one(raw_u, bc.u_left, bc.u_right)
    w = lift_one(raw_w, bc.w_left, bc.w_right)
    phi = lift_one(raw_phi, bc.phi_left, bc.phi_right)
    return u, w, phi


def lifting_trig(
    x_norm: torch.Tensor, raw_u: torch.Tensor, raw_w: torch.Tensor, raw_phi: torch.Tensor, bc: BoundaryConditions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dirichlet boundary condition lifting function (trigonometric/Galerkin style - trig)

    Description:
    Uses trigonometric functions (sin/cos) to construct a lifting function satisfying the Dirichlet boundary conditions.
    This form is consistent with the trial functions of the Galerkin method and has better periodicity and smoothness.

    Mathematical principle:
    For given boundary values, a constraint-satisfying solution is constructed using sine functions:

    1. Both ends specified (e.g. C-C boundary condition, u=w=φ=0):
       v(x) = vL + x*(vR-vL) + sin(π*x)*v̂(x)

       Example: clamped-clamped beam, u(0)=u(1)=0
       - vL = 0, vR = 0
        - u(x) = 0 + sin(π*x)*û(x)
        - Verification: u(0) = sin(0)*û(0) = 0
        - Verification: u(1) = sin(π)*û(1) = 0
        - Feature: sin(π*x) is 0 at the endpoints and reaches a maximum of 1 in the middle

    2. One end specified (e.g. left end clamped, right end free):
       - Left end specified: v(x) = vL + sin(π*x/2)*v̂(x)
       - Right end specified: v(x) = vR + cos(π*x/2)*v̂(x)

       Example: cantilever beam, u(0)=0, u(1)=free
       - vL = 0, vR = None
       - u(x) = 0 + sin(π*x/2)*û(x)
       - Verification: u(0) = sin(0)*û(0) = 0
       - Verification: u(1) = sin(π/2)*û(1) = û(1) (learned by the network)

    3. Both ends free (e.g. the rotation φ of a simply-supported beam):
       v(x) = v̂(x)

    Technical advantages (compared with poly):
    - Smoother gradient variation, beneficial for network optimization
    - Consistent with the trial-function form of the Galerkin method
    - Performs better on periodic problems
    - May better capture oscillatory features

    Parameters:
    - x_norm: normalized coordinate tensor, range [0,1], shape=(N,1)
    - raw_u/raw_w/raw_phi: raw outputs of the neural network, corresponding to the three displacement components u, w, φ
    - bc: boundary condition specification, containing the Dirichlet value at each endpoint (None means free)

    Returns:
    - (u, w, phi): tuple of lifted displacement fields satisfying the boundary conditions
    """
    import math
    pi = math.pi
    xi = x_norm

    def lift_one_trig(vhat: torch.Tensor, vL: Optional[float], vR: Optional[float]) -> torch.Tensor:
        """Trigonometric boundary condition lifting for a single physical quantity"""
        if vL is not None and vR is not None:
            # Both ends specified: v = vL + xi*(vR-vL) + sin(π*xi)*vhat
            # sin(π*xi) is 0 at xi=0,1 and reaches a maximum of 1 at xi=0.5
            sin_term = torch.sin(pi * xi)
            return vL + xi * (vR - vL) + sin_term * vhat
        elif vL is not None and vR is None:
            # Only left end specified: v = vL + sin(π*xi/2)*vhat
            # sin(π*xi/2) is 0 at xi=0 and 1 at xi=1
            sin_half = torch.sin(pi * xi / 2.0)
            return vL + sin_half * vhat
        elif vL is None and vR is not None:
            # Only right end specified: v = vR + cos(π*xi/2)*vhat
            # cos(π*xi/2) is 0 at xi=1 and 1 at xi=0
            cos_half = torch.cos(pi * xi / 2.0)
            return vR + cos_half * vhat
        else:
            # Both ends free: use the network output directly
            return vhat

    u = lift_one_trig(raw_u, bc.u_left, bc.u_right)
    w = lift_one_trig(raw_w, bc.w_left, bc.w_right)
    phi = lift_one_trig(raw_phi, bc.phi_left, bc.phi_right)
    return u, w, phi


def lifting_none(
    x_norm: torch.Tensor, raw_u: torch.Tensor, raw_w: torch.Tensor, raw_phi: torch.Tensor, bc: BoundaryConditions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """No lifting function (pure soft-constraint mode)

    Description:
    Returns the raw output of the neural network directly, without performing any boundary condition lifting.
    In this mode, all boundary conditions are satisfied entirely through penalty terms (soft constraints).

    Use cases:
    - For comparison experiments, evaluating the effect of hard constraints (lifting) vs soft constraints (penalty-only)
    - When the boundary conditions are complex and difficult to express in analytical form
    - Studying the effect of different boundary handling strategies on PINN training

    Notes:
    - When using this mode, it is recommended to increase bc_weight to ensure the boundary conditions are satisfied
    - Convergence may be slower than when using a lifting function
    - The accuracy with which the boundary conditions are satisfied depends on the choice of penalty weight

    Parameters:
    - x_norm: normalized coordinate tensor, range [0,1], shape=(N,1)
    - raw_u/raw_w/raw_phi: raw outputs of the neural network
    - bc: boundary condition specification (not used in this mode, kept only for interface consistency)

    Returns:
    - (u, w, phi): returns the raw network outputs directly
    """
    return raw_u, raw_w, raw_phi


def get_lifting_function(lifting_basis: str = "poly"):
    """Get the corresponding lifting function by basis-function type

    Parameters:
        lifting_basis: lifting basis-function family
            - 'poly': polynomial form x*(1-x)*NN(x)
            - 'trig': trigonometric form sin²(πx)*NN(x)
            - 'none': no lifting (pure soft-constraint mode)

    Returns:
        The corresponding lifting function
    """
    basis_norm = str(lifting_basis).lower().strip()

    if basis_norm in ("poly", "polynomial"):
        return lifting
    elif basis_norm in ("trig", "sin", "sincos", "galerkin"):
        return lifting_trig
    elif basis_norm in ("none", "identity", "soft", "raw"):
        return lifting_none
    else:
        raise ValueError(
            f"Unsupported lifting_basis: {lifting_basis}. "
            "Available options: ['poly', 'trig', 'none']"
        )


class BoundaryConditionPenalty:
    """Boundary condition penalty calculator

    Description:
    Computes the penalty loss for different types of boundary conditions, mainly handling Neumann boundary conditions (such as M=0 at a free end)
    and reinforcing the constraint strength of Dirichlet boundary conditions. This is a soft-constraint method, complementary to the hard constraints of the lifting function.

    Handling logic and strategy:

    1. Mixed-constraint strategy:
       - Hard constraint (lifting): handles displacement boundary conditions (u=0, w=0, φ=0), satisfied exactly
       - Soft constraint (penalty): handles force/moment boundary conditions (M=0) and reinforces displacement constraints, satisfied approximately
       - Combination advantage: reduces the penalty weight requirement and improves numerical stability

    2. Boundary-condition-specific handling:

       a) C-C (clamped-clamped):
          - Hard constraint: the lifting function already handles u=w=φ=0
          - Soft constraint: an additional penalty ensures constraint strength and prevents numerical drift
          - Physical meaning: both ends fully clamped, all degrees of freedom restricted

       b) S-S/H-H (simply-supported/hinged-hinged):
          - Hard constraint: the lifting function handles u=w=0 (displacements fixed)
          - Soft constraint: the penalty handles M=0 (zero bending moment, a natural boundary condition)
          - φ free: determined jointly by energy minimization and moment balance
          - Physical meaning: the end can rotate freely but cannot translate

       c) C-S/C-H (clamped-simply-supported/clamped-hinged):
          - Left end: lifting handles u=w=φ=0, the penalty reinforces the constraint
          - Right end: lifting handles u=w=0, the penalty handles M=0
          - Physical meaning: a mixed constraint with one end clamped and one end simply supported

    3. Bending moment formula:

       Linear boundary condition:
       M = b11*∂u/∂x + d11*∂φ/∂x - m_xT

       Nonlinear boundary condition:
       M = b11*∂u/∂x + d11*∂φ/∂x + (b11/2λ)*(∂w/∂x)² - m_xT

       Key difference:
       - Nonlinear: additionally includes the geometric nonlinear term (b11/2λ)*(∂w/∂x)²

    4. Implementation features:

       a) Endpoint evaluation:
          - The boundary condition residual is computed only at x=0 and x=1
          - Avoids unnecessary constraints at interior points, reducing the computational cost

       b) Penalty construction:
          - Uses a squared-error form: loss = Σ(residual²)
          - Automatically differentiable: the required derivatives are computed via autograd
          - Weight adjustment: the bc_weight parameter balances the PDE loss and the boundary loss

       c) Numerical stability:
          - The safe_divide function prevents division-by-zero errors
          - Supports GPU-accelerated computation

    5. Workflow:
       step1: Initialization → set material parameters, physical parameters, device type
       step2: Boundary evaluation → _evaluate_at_boundary() computes field variables and derivatives at the endpoints
       step3: Moment computation → compute the bending moment M according to linear/nonlinear theory
       step4: Residual computation → compute() computes the total penalty loss according to the boundary type
       step5: Gradient backpropagation → the penalty loss participates in the optimization of the total loss
    """

    def __init__(self, bc: BoundaryConditions, coeffs: MaterialCoeffs, params: PhysicalParams, device: Optional[torch.device] = None, *, is_nonlinear: bool = False) -> None:
        self.bc = bc
        self.coeffs = coeffs
        self.params = params
        self.lambda_val = float(params.lambda_val)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_nonlinear = bool(is_nonlinear)

    def _evaluate_at_boundary(self, x_val: float, field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        x = torch.tensor([[x_val]], device=self.device, dtype=torch.float32)
        x.requires_grad_(True)
        fields = field_eval(x)

        # Compute the endpoint bending moment M (used for simply-supported/hinged boundary conditions)
        # Boundary condition formulas:
        # Linear boundary: M = b11*∂u/∂x + d11*∂φ/∂x - m_x^T = 0
        # Nonlinear boundary: M = b11*∂u/∂x + d11*∂φ/∂x + (b11/2λ)*(∂w/∂x)² - m_x^T = 0
        # where: ux = ∂u/∂x, phix = ∂φ/∂x, wx = ∂w/∂x
        # Note: the d11 term has a positive sign (not negative)

        b11 = as_shape(self.coeffs.b11, x)
        d11 = as_shape(self.coeffs.d11, x)

        # Basic bending moment term (the d11 term has a positive sign)
        moment = b11 * fields["ux"] + d11 * fields["phix"]

        # Nonlinear term (included by the nonlinear theory only)
        if self.is_nonlinear:
            moment = moment + safe_divide(b11 * (fields["wx"] ** 2), 2.0 * self.lambda_val)

        # Thermal bending moment term (if present)
        moment = moment - self.params.m_xT

        fields["M"] = moment
        return fields

    def compute(self, field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute the boundary penalty at both endpoints."""

        left = self._evaluate_at_boundary(0.0, field_eval)
        right = self._evaluate_at_boundary(1.0, field_eval)

        loss = torch.tensor(0.0, device=self.device)
        bc_type = self.bc.type

        # Dirichlet penalties (soft constraints) are applied only when the corresponding
        # boundary value is specified (i.e., not None). This keeps existing BC behavior
        # unchanged while allowing free-end conditions such as C-F.
        if self.bc.u_left is not None:
            loss += ((left["u"] - self.bc.u_left) ** 2).mean()
        if self.bc.u_right is not None:
            loss += ((right["u"] - self.bc.u_right) ** 2).mean()
        if self.bc.w_left is not None:
            loss += ((left["w"] - self.bc.w_left) ** 2).mean()
        if self.bc.w_right is not None:
            loss += ((right["w"] - self.bc.w_right) ** 2).mean()

        # Add the specific penalty term (φ or M) according to the boundary condition type
        if bc_type == BoundaryConditionType.CLAMPED_CLAMPED.value:
            # C-C: φ=0 at both ends
            loss += (left["phi"] ** 2).mean() + (right["phi"] ** 2).mean()
        elif bc_type in (BoundaryConditionType.SIMPLE_SIMPLE.value, BoundaryConditionType.HINGED_HINGED.value):
            # S-S/H-H: M=0 at both ends
            loss += (left["M"] ** 2).mean() + (right["M"] ** 2).mean()
        elif bc_type in (BoundaryConditionType.CLAMPED_SIMPLE.value, BoundaryConditionType.CLAMPED_HINGED.value):
            # C-S/C-H: φ=0 at the left end, M=0 at the right end
            loss += (left["phi"] ** 2).mean() + (right["M"] ** 2).mean()
        elif bc_type == BoundaryConditionType.CLAMPED_FREE.value:
            # C-F: left end clamped (φ=0), right end free (M=0)
            loss += (left["phi"] ** 2).mean() + (right["M"] ** 2).mean()
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")

        # Return the total loss (already a scalar)
        return loss


__all__ = ["lifting", "lifting_trig", "lifting_none", "get_lifting_function", "BoundaryConditionPenalty"]

# ===============
# Convenience constructors
# ===============

def make_bc_spec(bc_type: str) -> BoundaryConditions:
    """Construct a standard boundary condition specification by boundary type.

    Description:
    This is a convenience constructor that automatically generates the corresponding
    BoundaryConditions object from a standard engineering boundary condition type, simplifying user input and ensuring consistency of the boundary condition setup.

    Physical meaning and engineering applications of the boundary conditions:

    1. C-C (Clamped-Clamped):
       - Physical constraint: u=w=φ=0 (both ends)
       - Engineering meaning: both ends fully clamped, similar to the end fixity of a reinforced-concrete beam
       - Application scenarios: bridge girders, building frame beams, fixed connections of mechanical parts
       - Features: maximum stiffness and the strongest load-bearing capacity, but significant stress concentration at the ends

    2. S-S (Simply-Supported):
       - Physical constraint: u=w=0, φ=free, M=0 (both ends)
       - Engineering meaning: both ends hinged-supported, constraining only displacement and allowing free rotation
       - Application scenarios: simply-supported beam bridges, precast slab beams, bearing supports of mechanical transmission shafts
       - Features: simple to compute, uniform stress distribution; a classic model for theoretical analysis

    3. H-H (Hinged-Hinged):
       - Physical constraint: same as S-S, u=w=0, φ=free, M=0 (both ends)
       - Engineering meaning: both ends hinged; equivalent to simply-supported in Timoshenko theory
       - Application scenarios: the same as a simply-supported beam, with more emphasis on the hinged nature of the connection
       - Features: mathematically equivalent to S-S, emphasizing the rotational freedom at the ends

    4. C-S (Clamped-Simply-Supported):
       - Physical constraint: left end u=w=φ=0; right end u=w=0, φ=free, M=0
       - Engineering meaning: an asymmetric constraint with one end clamped and one end simply supported
       - Application scenarios: cantilever-simply-supported combined beams, mechanical structures with asymmetric supports
       - Features: structurally asymmetric, stress concentration at the clamped end, relatively larger deformation at the simply-supported end

    5. C-H (Clamped-Hinged):
       - Physical constraint: same as C-S, mathematically equivalent
       - Engineering meaning: one end clamped, one end hinged-supported
       - Application scenarios: the same as C-S, emphasizing the nature of the hinged connection
       - Features: fully equivalent to C-S in Timoshenko theory

    6. C-F (Clamped-Free):
       - Physical constraint: left end u=w=φ=0; right end u, w, φ free (None)
       - Engineering meaning: a typical cantilever beam (left end clamped, right end free)
       - Application scenarios: cantilever beams, structures with free ends such as blades/thin beams
       - Features: the mechanical boundary conditions at the free end (such as M=0, shear force=0, axial force=0) should arise naturally from the energy variation, and can be reinforced with penalty terms if necessary

    Boundary condition selection guidance:
    - Maximum stiffness requirement → choose C-C
    - Economy and simplicity → choose S-S/H-H
    - Special constraint conditions → choose C-S/C-H
    - Theoretical study and comparison → it is recommended to analyze multiple boundary conditions simultaneously

    Implementation strategy:
    - Dirichlet conditions (u=w=φ=value): implemented via hard-coding in the lifting function
    - Neumann conditions (M=0): implemented via the BoundaryConditionPenalty penalty term
    - Free conditions (φ=None): determined naturally by the energy variational principle

    Conventions (consistent with the existing implementation):
    - C-C: u=w=φ=0 at both ends, fully hard-constrained
    - S-S/H-H: u=0, w=0 at both ends, φ free (None); M=0 implemented via the penalty term
    - C-S/C-H: left end u=w=φ=0; right end u=0, w=0, φ free, M=0 implemented via the penalty term
    - C-F: left end u=w=φ=0; right end u, w, φ free (None); the free-end mechanical boundary conditions can be reinforced via penalty terms
    """
    if bc_type == BoundaryConditionType.CLAMPED_CLAMPED.value:
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=0.0,
            w_left=0.0,
            w_right=0.0,
            phi_left=0.0,
            phi_right=0.0,
        )
    elif bc_type in (BoundaryConditionType.SIMPLE_SIMPLE.value, BoundaryConditionType.HINGED_HINGED.value):
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=0.0,
            w_left=0.0,
            w_right=0.0,
            phi_left=None,
            phi_right=None,
        )
    elif bc_type in (BoundaryConditionType.CLAMPED_SIMPLE.value, BoundaryConditionType.CLAMPED_HINGED.value):
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=0.0,
            w_left=0.0,
            w_right=0.0,
            phi_left=0.0,
            phi_right=None,
        )
    elif bc_type == BoundaryConditionType.CLAMPED_FREE.value:
        return BoundaryConditions(
            type=bc_type,
            u_left=0.0,
            u_right=None,
            w_left=0.0,
            w_right=None,
            phi_left=0.0,
            phi_right=None,
        )
    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")

__all__.extend(["make_bc_spec"])
