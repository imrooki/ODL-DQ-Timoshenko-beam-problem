"""
Timoshenko beam physical formulas and energy loss computation module

Author: Yang
Version: 1.0

Responsibilities:
- Implement the physical-formula computation for the Timoshenko beam based on the energy variational principle
- Provide computation of linear/nonlinear strain energy density, thermal strain energy density, and external-force work density
- Compute the analytical expressions of internal force components such as the bending moment and axial force
- Provide a complete energy loss function class and a total-loss assembler

Core physical formulas:
- Linear strain energy: based on classical Timoshenko beam theory
- Nonlinear strain energy: includes the large-deformation geometric nonlinear term (∂w/∂x)²
- Thermal strain energy: accounts for temperature changes and the thermal expansion effect of the material
- Elastic foundation energy: includes the Winkler foundation and the Pasternak foundation
- External-force work: the work done by the distributed load q(x) on the transverse displacement w

Technical features:
- Strictly follows the energy variational principle: δΠ = 0
- Supports spatially varying material properties: a11(x), b11(x), d11(x), a55(x)
- Numerically stable loss function design, suitable for PINNs training
- Supports unified handling of multiple boundary conditions

Input/output conventions:
- Displacement field dictionary: {'u', 'w', 'phi', 'ux', 'wx', 'phix'}
- Material coefficients: in functional form Callable[[torch.Tensor], torch.Tensor]
- Returned tensors: preserve computational graph integrity, supporting automatic differentiation

Design principles:
- Separate physical formulas from numerical implementation to ensure theoretical accuracy
- All energy components can be computed and verified independently
- A unified interface compatible with both linear and nonlinear problems
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Type-checking path (in-package import)
    from .bc import BoundaryConditionPenalty

# Compatibility for running the script directly: fall back to absolute imports when relative imports fail
try:  # Prefer in-package relative imports
    from .data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from .numerics import safe_divide, mean_integral, as_shape, quad_nodes_weights
except ImportError:  # Compatibility for running this file directly
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from modules.data_types import MaterialCoeffs, PhysicalParams, BoundaryConditions
    from modules.numerics import safe_divide, mean_integral, as_shape, quad_nodes_weights


# ========== Energy loss function class ==========

class EnergyLoss:
    """Physical loss calculator based on the energy variational principle — the theoretical core of the project

    ================================================================================
                        Theoretical basis and implementation strategy of the energy method
    ================================================================================

    [Theoretical basis: Hamilton's variational principle]

    Fundamental principle of structural mechanics: when the system is in equilibrium, the total potential energy takes a stationary value
    Mathematical expression: δΠ = 0

    For the Timoshenko beam:
    Π_total = Π_str + Π_w - Π_e

    where:
    - Π_str: elastic strain energy (the energy stored by material deformation)
    - Π_w: elastic foundation energy (the energy stored by the foundation support)
    - Π_e: external-force potential energy (the work potential of the external load)

    [Advantages of the energy method compared with the strong-form PDE]

    1. Theoretical advantages:
       - Automatically satisfies the equilibrium equations: the energy stationarity condition naturally yields all equilibrium equations
       - Clear physical meaning: each term has a well-defined physical meaning
       - Natural boundary conditions automatically satisfied: natural boundary conditions are automatically satisfied through the variation

    2. Numerical advantages:
       - The integral form is more stable: avoids the numerical error of high-order derivatives
       - Global constraint: the energy is a domain-wide integral, a stronger constraint than a pointwise PDE constraint
       - Good convergence: energy minimization is a convex optimization problem (in the linear case)

    3. Implementation advantages:
       - Unified framework: linear/nonlinear problems only require changing the energy expression
       - Easy to extend: adding new energy terms (such as damping, inertial forces) is convenient
       - Physical consistency: energy conservation is automatically satisfied

    [Exact expression of the strain energy density]

    Full formula implementation:

    Linear theory:
    π_str = [first term][second term] + [third term][∂φ/∂x] + a55(∂w/∂x + λφ)²

    Nonlinear theory (adding geometric nonlinear terms):
    - First term adds: (a11/2λ)(∂w/∂x)²
    - Second term adds: (1/2λ)(∂w/∂x)²
    - Third term adds: (b11/2λ)(∂w/∂x)²

    Key innovations:
    - Thermal strain contribution accounting for the temperature effect
    - Unified handling of linear/nonlinear problems

    [Key techniques of the numerical implementation]

    1. Automatic differentiation ensures accuracy:
       - All derivatives are computed via PyTorch autograd
       - Avoids the truncation error of finite differences
       - Supports exact computation of arbitrary-order derivatives

    2. Numerical stability guarantees:
       - safe_divide avoids division by zero
       - NaN/Inf detection and handling
       - Negative-energy warning (physical anomaly detection)

    3. Efficient integration strategy:
       - Supports multiple integration methods (MC, Gauss, AGQ)
       - Adaptive integration adjusts according to the variation of the function
       - Integration weights are precomputed to optimize performance

    ================================================================================
    """

    def __init__(
        self,
        coeffs: MaterialCoeffs,
        params: PhysicalParams,
        bc: BoundaryConditions,
        device: Optional[torch.device] = None,
        *,
        is_nonlinear: bool = False,
    ) -> None:
        self.coeffs = coeffs
        self.params = params
        self.bc = bc
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_nonlinear = bool(is_nonlinear)
        if self.params.lambda_val <= 0:
            raise ValueError(f"lambda must be greater than 0, current: {self.params.lambda_val}")

    def compute_elastic_foundation_energy_density(self, x: torch.Tensor, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the elastic foundation energy density

        Theoretical formula:
        π_w = k1*w² + k2*(∂w/∂x)²

        where:
        - k1: Winkler elastic foundation stiffness coefficient (point support)
        - k2: Pasternak elastic foundation stiffness coefficient (continuous support)
        - w: deflection
        - ∂w/∂x: derivative of the deflection

        Parameters:
            x: normalized coordinate tensor
            fields: displacement field dictionary, containing 'w' and 'wx'

        Returns:
            Elastic foundation energy density tensor
        """
        k1, k2 = self.params.k1, self.params.k2
        w, wx = fields["w"], fields["wx"]

        # Compute the elastic foundation energy density
        foundation_density = k1 * (w ** 2) + k2 * (wx ** 2)

        return foundation_density

    def compute_strain_energy_density(self, x: torch.Tensor, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the strain energy density of the Timoshenko beam, implemented according to the full formula form.

        Full formula, the strain energy density is:

        Linear problem:
        [a₁₁∂u/∂x + b₁₁∂φ/∂x - λn_xᵀ]
        × [∂u/∂x - λα∆T]
        + [b₁₁∂u/∂x + d₁₁∂φ/∂x - λm_xᵀ] × ∂φ/∂x
        + a₅₅(∂w/∂x + λφ)²

        Nonlinear problem: on top of the above, the corresponding nonlinear term is added inside each bracket.

        Parameters:
            x: normalized coordinate tensor
            fields: dictionary of the displacement field and its derivatives

        Returns:
            Strain energy density tensor
        """
        import warnings

        # Input validity check
        for field_name, field_tensor in fields.items():
            if torch.isnan(field_tensor).any():
                raise ValueError(f"Displacement field '{field_name}' contains NaN values")
            if torch.isinf(field_tensor).any():
                warnings.warn(f"Displacement field '{field_name}' contains infinite values; continuing computation but use caution")

        # Get the material coefficients
        a11 = as_shape(self.coeffs.a11, x)
        b11 = as_shape(self.coeffs.b11, x)
        d11 = as_shape(self.coeffs.d11, x)
        a55 = as_shape(self.coeffs.a55, x)

        # Extract the displacement field and derivatives
        ux, wx, phix = fields["ux"], fields["wx"], fields["phix"]
        phi = fields["phi"]
        lambda_val = self.params.lambda_val

        # Compute each term according to the full formula
        # First bracket term: [a₁₁∂u/∂x + b₁₁∂φ/∂x - λn_xᵀ]
        term1 = (a11 * ux
                + b11 * phix
                - lambda_val * self.params.n_xT)

        # Second bracket term: [∂u/∂x - λα∆T]
        term2 = (ux
                - lambda_val * self.params.alpha_t * self.params.DeltaT)

        # Third bracket term: [b₁₁∂u/∂x + d₁₁∂φ/∂x - λm_xᵀ]
        term3 = (b11 * ux
                + d11 * phix
                - lambda_val * self.params.m_xT)

        # Shear term: a₅₅(∂w/∂x + λφ)²
        shear_term = a55 * (wx + lambda_val * phi) ** 2

        # Nonlinear terms (added only when is_nonlinear=True)
        if self.is_nonlinear:
            # Add the nonlinear term to each bracket term
            # First bracket adds: + (a₁₁/2λ)(∂w/∂x)²
            term1 = term1 + safe_divide(a11 * (wx ** 2), 2.0 * lambda_val)

            # Second bracket adds: + (1/2λ)(∂w/∂x)²
            term2 = term2 + safe_divide(wx ** 2, 2.0 * lambda_val)

            # Third bracket adds: + (b₁₁/2λ)(∂w/∂x)²
            term3 = term3 + safe_divide(b11 * (wx ** 2), 2.0 * lambda_val)

        # Assemble the full strain energy density
        strain_energy_density = term1 * term2 + term3 * phix + shear_term

        # Result validity check
        if torch.isnan(strain_energy_density).any():
            raise RuntimeError("Strain energy density computation produced NaN values")
        if torch.isinf(strain_energy_density).any():
            warnings.warn("Strain energy density contains infinite values - possible numerical instability")
        if (strain_energy_density < 0).any():
            warnings.warn("Negative strain energy detected - please check physical parameters and boundary conditions")

        return strain_energy_density

    def compute_total_energy(self, x: torch.Tensor, fields: Dict[str, torch.Tensor], weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute the total potential energy and each energy component based on Timoshenko beam theory.

        Unified definition:
        Π_all = Π_str + Π_w - Π_e

        Physical meaning of each component:
        - Π_str: elastic strain energy (the energy stored by material deformation, including the thermal strain term)
          ∫[1/2 * strain energy density] dx, computed by compute_strain_energy_density
          Note: the thermal strain energy is integrated into the strain energy density (not a separate term)
        - Π_w: elastic foundation energy (the energy stored by the foundation support)
          ∫[1/2 * (k1*w² + k2*(∂w/∂x)²)] dx
        - Π_e: external-force work (the work done by the external load)
          ∫[q * w] dx, where q is the distributed load and w is the deflection
        - Π_all: total potential energy (the objective function of the PINN, to be minimized)

        Note: the Pi_str_T return value is kept as 0.0 for interface compatibility; the actual thermal strain contribution is already included in Pi_str.

        Parameters:
            x: normalized coordinate tensor [0,1]
            fields: displacement field and its derivatives {'u','w','phi','ux','wx','phix'}
            weights: integration weights (optional)

        Returns:
            A dictionary containing all energy components
        """
        # Compute the strain energy density (already includes all strain terms, nonlinear terms, and the thermal strain term)
        strain_density = self.compute_strain_energy_density(x, fields)

        # Compute the elastic foundation energy density
        foundation_density = self.compute_elastic_foundation_energy_density(x, fields)

        # Integral of each energy component
        Pi_str = 0.5 * mean_integral(strain_density, weights=weights)
        Pi_w = 0.5 * mean_integral(foundation_density, weights=weights)

        # External-force work density: q * w
        # Note: according to the principle of virtual work, the external-force work does not need a 1/2 coefficient
        Pi_e = mean_integral(self.params.q * fields["w"], weights=weights)

        # Compute the total potential energy
        Pi_all = Pi_str + Pi_w - Pi_e

        return {
            "Pi_str": Pi_str,
            "Pi_w": Pi_w,
            "Pi_str_T": torch.tensor(0.0, device=x.device),  # Legacy interface kept; thermal strain contribution already integrated into Pi_str
            "Pi_e": Pi_e,
            "Pi_all": Pi_all,
            "strain_density": strain_density,
            "foundation_density": foundation_density,
        }


# Linear/nonlinear selection is unified through EnergyLoss via is_nonlinear.


class WeightedEnergyLoss:
    """Total-loss assembler: total = Π_all + boundary_weight * BC_penalty."""

    def __init__(
        self,
        energy_loss: "EnergyLoss",
        bc_penalty: BoundaryConditionPenalty,
        bc_weight: float = 1000.0,
        adaptive_weights: bool = False,
    ) -> None:
        self.energy_loss = energy_loss
        self.bc_penalty = bc_penalty
        self.bc_weight = float(bc_weight)
        self.adaptive_weights = bool(adaptive_weights)
        # Use collections.deque to limit the history length and prevent memory leaks
        if adaptive_weights:
            from collections import deque
            self._history = {
                "energy": deque(maxlen=100),  # Keep at most 100 history records
                "bc": deque(maxlen=100)
            }
        else:
            self._history = None
        # Integrator ('mc' or 'gauss' or 'clenshaw' or 'agq') - passed from params.py
        self.integrator: str = 'mc'

        # AGQ configuration parameters - passed from params.py
        # Important: these initial values serve only as defaults; the actual values are overridden by solver.py during initialization
        # See lines 193-201 of solver.py, which uses the actual configuration values from params.py
        self.agq_rule: str = 'G10K21'        # AGQ integration rule (high-order/low-order pair)
        self.agq_abs_tol: float = 1e-6       # Absolute tolerance
        self.agq_rel_tol: float = 1e-4       # Relative tolerance
        self.agq_max_points: int = 4096      # Maximum number of integration points
        self.agq_max_depth: int = 100        # Maximum subdivision depth
        self.agq_refine_every: int = 0       # Refinement frequency (0 means no automatic refinement)
        self.agq_fail_policy: str = 'use_partial'  # Failure policy

        # AGQ cache and state
        self._agq_nodes: Optional[torch.Tensor] = None
        self._agq_weights: Optional[torch.Tensor] = None
        self._agq_build_count: int = 0
        self._agq_last_nodes: int = 0
        self._agq_last_intervals: int = 0
        self._agq_last_hit_limit: bool = False
        self._agq_info_printed: bool = False

    def _adaptive_w(self, energy: torch.Tensor, bc_loss: torch.Tensor) -> float:
        """Adaptive weight computation, using a safer numerical method"""
        bc_loss_val = abs(bc_loss.item())
        energy_val = abs(energy.item())

        # Use a stricter threshold to avoid extreme ratios
        if bc_loss_val > 1e-10 and energy_val > 1e-15:
            ratio = energy_val / bc_loss_val
            # Constrain the ratio within a reasonable range
            ratio = max(min(ratio, self.bc_weight * 10.0), self.bc_weight * 0.1)
            return float(ratio)
        return float(self.bc_weight)

    def compute_total_loss(self, x: torch.Tensor, field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Select the integrator
        integ_norm = (
            (self.integrator or 'mc')
            .lower()
            .replace('_', '')
            .replace('-', '')
            .replace('–', '')  # en dash
            .replace('—', '')  # em dash
            .replace(' ', '')
        )
        quad_x: Optional[torch.Tensor] = None
        weights: Optional[torch.Tensor] = None
        if integ_norm == 'agq':
            # Cache or rebuild the AGQ nodes
            need_build = self._agq_nodes is None or self._agq_weights is None
            if not need_build and self.agq_refine_every and self.agq_refine_every > 0:
                self._agq_build_count += 1
                if self._agq_build_count % int(self.agq_refine_every) == 0:
                    need_build = True
            if need_build:
                try:
                    quad_x, weights = self._build_agq_nodes_and_weights(field_eval, device=x.device, dtype=x.dtype)
                    self._agq_nodes, self._agq_weights = quad_x, weights
                    # Print statistics once after the first build
                    if not self._agq_info_printed:
                        try:
                            print(
                                f"AGQ nodes built: n_nodes={self._agq_last_nodes}, "
                                f"n_intervals={self._agq_last_intervals}, hit_limit={self._agq_last_hit_limit}"
                            )
                        except Exception:
                            pass
                        self._agq_info_printed = True
                except Exception as e:
                    # Hard-error fallback policy
                    if (self.agq_fail_policy or 'use_partial').lower() == 'fallback_gauss':
                        quad_x, weights = quad_nodes_weights('gauss', int(x.shape[0]), device=x.device, dtype=x.dtype)
                    else:
                        raise e
            else:
                quad_x, weights = self._agq_nodes, self._agq_weights
        else:
            # Original integrators (mc/gauss/clenshaw)
            n = int(x.shape[0])
            if integ_norm in ('gauss', 'gausslegendre', 'legendre'):
                quad_x, weights = quad_nodes_weights('gauss', n, device=x.device, dtype=x.dtype)
            elif integ_norm in ('clenshaw', 'clenshawcurtis', 'cc'):
                quad_x, weights = quad_nodes_weights('clenshaw', n, device=x.device, dtype=x.dtype)

        # Compute the fields at the selected nodes
        x_use = quad_x if quad_x is not None else x
        fields = field_eval(x_use)
        energy = self.energy_loss.compute_total_energy(x_use, fields, weights=weights)
        bc_loss = self.bc_penalty.compute(field_eval)
        boundary_weight = self._adaptive_w(energy["Pi_all"], bc_loss) if self.adaptive_weights else self.bc_weight
        total = energy["Pi_all"] + boundary_weight * bc_loss
        if self._history is not None:
            # The deque handles the length limit automatically; no manual truncation needed
            self._history["energy"].append(float(energy["Pi_all"].item()))
            self._history["bc"].append(float(bc_loss.item()))
        return {"total": total, "bc": bc_loss, "bc_weight": boundary_weight, **energy}

    def _build_agq_nodes_and_weights(
        self,
        field_eval: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build 1D AGQ (joint adaptive quadrature) nodes and weights.

        - Dynamically parse the high-order/low-order Gauss-Legendre point counts according to self.agq_rule;
        - Supported format: 'G{n_lo}K{n_hi}', e.g. 'G10K21', 'G7K15', 'G15K31', etc.;
        - Subdivision strategy: accept if the error e <= max(abs_tol, rel_tol * |Q_hi|), otherwise bisect;
        - Limit control: when max_points or max_depth is reached, accept the current interval (using high-order nodes) and mark that the tolerance was not met.
        """
        import numpy as _np

        def leggauss(n: int):
            xi, w = _np.polynomial.legendre.leggauss(int(n))
            return xi.astype(_np.float64), w.astype(_np.float64)

        def map_to(a: float, b: float, xi: _np.ndarray, w: _np.ndarray):
            c = 0.5 * (b - a)
            m = 0.5 * (a + b)
            x = m + c * xi
            ww = c * w
            return x, ww

        # Parse agq_rule (e.g. "G10K21" → n_lo=10, n_hi=21)
        import re
        rule = getattr(self, 'agq_rule', 'G10K21')
        match = re.match(r'G(\d+)K(\d+)', rule, re.IGNORECASE)
        if match:
            n_lo, n_hi = int(match.group(1)), int(match.group(2))
        else:
            n_lo, n_hi = 10, 21
            import warnings
            warnings.warn(f"Unable to parse AGQ rule '{rule}', using default G10K21")

        xi_hi, w_hi = leggauss(n_hi)
        xi_lo, w_lo = leggauss(n_lo)

        abs_tol = float(self.agq_abs_tol)
        rel_tol = float(self.agq_rel_tol)
        max_pts = max(1, int(self.agq_max_points))
        max_depth = max(0, int(self.agq_max_depth))

        stack: list[tuple[float, float, int]] = [(0.0, 1.0, 0)]
        X_list: list[_np.ndarray] = []
        W_list: list[_np.ndarray] = []
        total_pts = 0
        hit_limit = False
        # Recursion converted to an iterative stack
        while stack:
            a, b, depth = stack.pop()
            # Map nodes onto the subinterval
            xh_np, wh_np = map_to(a, b, xi_hi, w_hi)
            xl_np, wl_np = map_to(a, b, xi_lo, w_lo)

            # Evaluate the integrand density of the total potential energy (composite term)
            # High order
            xh = torch.tensor(xh_np, device=device, dtype=dtype).reshape(-1, 1)
            xh.requires_grad_(True)
            fields_h = field_eval(xh)
            strain_h = self.energy_loss.compute_strain_energy_density(xh, fields_h)
            foundation_h = self.energy_loss.compute_elastic_foundation_energy_density(xh, fields_h)
            integrand_h = 0.5 * strain_h + 0.5 * foundation_h - self.energy_loss.params.q * fields_h["w"]
            q_hi = float((torch.tensor(wh_np, device=device, dtype=dtype).view(-1) * integrand_h.view(-1)).sum().detach().cpu().item())

            # Low order
            xl = torch.tensor(xl_np, device=device, dtype=dtype).reshape(-1, 1)
            xl.requires_grad_(True)
            fields_l = field_eval(xl)
            strain_l = self.energy_loss.compute_strain_energy_density(xl, fields_l)
            foundation_l = self.energy_loss.compute_elastic_foundation_energy_density(xl, fields_l)
            integrand_l = 0.5 * strain_l + 0.5 * foundation_l - self.energy_loss.params.q * fields_l["w"]
            q_lo = float((torch.tensor(wl_np, device=device, dtype=dtype).view(-1) * integrand_l.view(-1)).sum().detach().cpu().item())

            err = abs(q_hi - q_lo)
            tol = max(abs_tol, rel_tol * abs(q_hi))

            force_accept = False
            if depth >= max_depth:
                force_accept = True
            if total_pts + n_hi > max_pts:
                force_accept = True

            if (err <= tol) or force_accept:
                X_list.append(xh_np)
                W_list.append(wh_np)
                total_pts += n_hi
                if total_pts >= max_pts:
                    hit_limit = True
                    # Force-accept all remaining intervals on the stack (using the low-order rule to save points)
                    while stack:
                        a_rem, b_rem, _ = stack.pop()
                        x_rem, w_rem = map_to(a_rem, b_rem, xi_lo, w_lo)
                        X_list.append(x_rem)
                        W_list.append(w_rem)
                    break
            else:
                m = 0.5 * (a + b)
                # Push the left and right subintervals onto the stack (right first, then left, so the left is popped first)
                if depth + 1 <= max_depth or total_pts + 2 * n_hi <= max_pts:
                    stack.append((m, b, depth + 1))
                    stack.append((a, m, depth + 1))
                else:
                    # Cannot subdivide further, force-accept
                    X_list.append(xh_np)
                    W_list.append(wh_np)
                    total_pts += n_hi
                    if total_pts >= max_pts:
                        hit_limit = True
                        # Force-accept all remaining intervals on the stack (using the low-order rule to save points)
                        while stack:
                            a_rem, b_rem, _ = stack.pop()
                            x_rem, w_rem = map_to(a_rem, b_rem, xi_lo, w_lo)
                            X_list.append(x_rem)
                            W_list.append(w_rem)
                        break

        if not X_list:
            # Fallback: return at least one set of Gauss nodes
            xi, w = xi_hi, w_hi
            x_np, w_np = map_to(0.0, 1.0, xi, w)
            X_list.append(x_np)
            W_list.append(w_np)

        X_all = _np.concatenate(X_list, axis=0)
        W_all = _np.concatenate(W_list, axis=0)
        x_all = torch.tensor(X_all, device=device, dtype=dtype).reshape(-1, 1)
        w_all = torch.tensor(W_all, device=device, dtype=dtype).reshape(-1)
        # Reset the AGQ build count
        self._agq_build_count = 0
        # Record statistics
        self._agq_last_nodes = int(w_all.numel())
        self._agq_last_intervals = int(len(X_list))
        self._agq_last_hit_limit = bool(hit_limit)

        # Weight-sum validation: check the completeness of the integration domain
        weights_sum = float(w_all.sum().item())
        domain_min, domain_max = float(X_all.min()), float(X_all.max())
        if abs(weights_sum - 1.0) > 0.01 or domain_min > 0.001 or domain_max < 0.999:
            import warnings
            warnings.warn(
                f"[AGQ] Integration domain may be incomplete: weights_sum={weights_sum:.4f} (expected~1.0), "
                f"coverage=[{domain_min:.4f}, {domain_max:.4f}] (expected [0,1]), "
                f"hit_limit={hit_limit}, n_nodes={w_all.numel()}, n_intervals={len(X_list)}"
            )

        return x_all.requires_grad_(True), w_all


def create_loss_function(
    problem_type: str,
    coeffs: MaterialCoeffs,
    params: PhysicalParams,
    bc: BoundaryConditions,
    bc_weight: float = 1000.0,
    adaptive_weights: bool = False,
    device: Optional[torch.device] = None,
) -> WeightedEnergyLoss:
    """Factory function: create the complete total-loss system (energy + boundary penalty)."""

    # Use a function-level lazy import to avoid a module-level circular import
    def _get_boundary_penalty_class():
        """Lazily import the boundary condition penalty class to avoid circular imports"""
        try:
            from .bc import BoundaryConditionPenalty
            return BoundaryConditionPenalty
        except ImportError:
            # Compatibility fallback: use an absolute import
            from modules.bc import BoundaryConditionPenalty
            return BoundaryConditionPenalty

    p = problem_type.lower()
    if p not in ("linear", "nonlinear"):
        raise ValueError(f"Unsupported problem type: {problem_type}")

    # Create the energy loss part
    energy = EnergyLoss(coeffs, params, bc, device, is_nonlinear=(p == "nonlinear"))

    # Lazily obtain and create the boundary condition penalty
    BoundaryConditionPenalty = _get_boundary_penalty_class()
    bc_pen = BoundaryConditionPenalty(bc, coeffs, params, device, is_nonlinear=(p == "nonlinear"))

    return WeightedEnergyLoss(energy, bc_pen, bc_weight, adaptive_weights)


__all__ = [
    "EnergyLoss",
    "WeightedEnergyLoss",
    "create_loss_function",
]
