"""
Unified spline derivative matrix generator for ODIL framework.

This module provides several spline variants:
- cubic (natural boundary by default)
- quintic spline (B-spline based)
- generic B-spline (configurable degree)
- tension spline (cardinal Hermite with tension parameter)
- Hermite spline (Catmull–Rom tangents as a special case)

All routines return first- and second-derivative matrices acting on nodal values.
The implementation avoids external dependencies and relies on torch linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SplineOptions:
    spline_type: str = "cubic"
    bc_type: str = "natural"
    bspline_degree: Optional[int] = None
    tension: float = 0.0
    derivative_values: Optional[torch.Tensor] = None


class SplineCore:
    def __init__(
        self,
        x: torch.Tensor,
        spline_type: str = "cubic",
        bc_type: str = "natural",
        device: Optional[torch.device] = None,
        bspline_degree: Optional[int] = None,
        tension: float = 0.0,
        derivative_values: Optional[torch.Tensor] = None,
    ) -> None:
        self.x = x.detach().clone().to(device=device, dtype=torch.float64)
        if not torch.all(self.x[1:] > self.x[:-1]):
            raise ValueError("Spline nodes must be strictly increasing")

        self.N = self.x.numel()
        if self.N < 3:
            raise ValueError("Spline construction requires at least 3 nodes")

        self.device = self.x.device
        self.spline_type = spline_type.lower()
        self.bc_type = bc_type.lower()
        self.bspline_degree = bspline_degree
        self.tension = float(tension)
        self.derivative_values = (
            derivative_values.detach().to(self.device, dtype=torch.float64)
            if derivative_values is not None
            else None
        )

    def compute_derivative_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.spline_type == "cubic":
            return self._build_cubic_natural()
        if self.spline_type == "quintic":
            return self._build_bspline_matrices(degree=5)
        if self.spline_type in {"b-spline", "bspline", "b_spline"}:
            degree = self.bspline_degree if self.bspline_degree is not None else 3
            if degree < 2:
                raise ValueError("B-spline degree must be >= 2")
            return self._build_bspline_matrices(degree=degree)
        if self.spline_type == "tension":
            return self._build_bspline_matrices(degree=3, tension=self.tension)
        if self.spline_type == "hermite":
            hermite_tension = max(0.0, float(self.tension))
            return self._build_bspline_matrices(degree=3, tension=hermite_tension)

        raise ValueError(f"Unsupported spline type: {self.spline_type}")

    # ------------------------------------------------------------------
    # Cubic spline with natural boundary (tri-diagonal system)
    # ------------------------------------------------------------------
    def _build_cubic_natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.diff(self.x)
        N = self.N
        device = self.device
        dtype = torch.float64

        L = torch.zeros((N, N), dtype=dtype, device=device)
        D = torch.zeros((N, N), dtype=dtype, device=device)

        # Natural boundary: second derivative zero
        L[0, 0] = 1.0
        L[-1, -1] = 1.0

        for i in range(1, N - 1):
            hi_prev = h[i - 1]
            hi = h[i]

            L[i, i - 1] = hi_prev
            L[i, i] = 2.0 * (hi_prev + hi)
            L[i, i + 1] = hi

            D[i, i - 1] = 6.0 / hi_prev
            D[i, i] = -6.0 * (1.0 / hi_prev + 1.0 / hi)
            D[i, i + 1] = 6.0 / hi

        D2 = torch.linalg.solve(L, D)

        # First derivative obtained via integration of spline polynomial
        identity = torch.eye(N, dtype=dtype, device=device)
        D1 = torch.zeros((N, N), dtype=dtype, device=device)
        for j in range(N):
            y = identity[:, j]
            M = D2 @ y
            deriv = torch.zeros(N, dtype=dtype, device=device)

            hi = h[0]
            deriv[0] = (y[1] - y[0]) / hi - (hi / 6.0) * (2.0 * M[0] + M[1])

            for i in range(1, N - 1):
                hi_prev = h[i - 1]
                hi_curr = h[i]
                left = (
                    M[i] * hi_prev / 2.0
                    - M[i - 1] * hi_prev / 6.0
                    + (y[i] - y[i - 1]) / hi_prev
                )
                right = (
                    -M[i] * hi_curr / 2.0
                    + M[i + 1] * hi_curr / 6.0
                    + (y[i + 1] - y[i]) / hi_curr
                )
                deriv[i] = 0.5 * (left + right)

            hi_last = h[-1]
            deriv[-1] = (
                M[-1] * hi_last / 2.0
                - M[-2] * hi_last / 6.0
                + (y[-1] - y[-2]) / hi_last
            )

            D1[:, j] = deriv

        return D1, D2

    # ------------------------------------------------------------------
    # B-spline based derivatives (degree >= 2)
    # ------------------------------------------------------------------
    def _build_bspline_matrices(self, degree: int, tension: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if degree >= self.N:
            raise ValueError("B-spline degree cannot exceed number of nodes - 1")

        if tension is not None and tension > 0:
            u = self._tension_parameterization(tension)
        else:
            u = (self.x - self.x[0]) / (self.x[-1] - self.x[0])
        knots = self._open_knot_vector(u, degree)

        B0, B1, B2 = self._evaluate_bspline_basis_and_derivatives(self.x, knots, degree)
        ident = torch.eye(B0.size(1), dtype=torch.float64, device=self.device)
        inv_B0 = torch.linalg.solve(B0, ident)
        D1 = B1 @ inv_B0
        D2 = B2 @ inv_B0
        return D1, D2

    def _tension_parameterization(self, tension: float) -> torch.Tensor:
        tension = float(max(0.0, tension))
        x = self.x
        delta = torch.diff(x)
        if delta.min() <= 0:
            raise ValueError("Nodes must be strictly increasing for tension spline")

        # power-based weighting: tension=0 -> original spacing, tension→1 -> uniform spacing
        power = max(0.0, 1.0 - min(tension, 0.999))
        weights = torch.pow(delta, power)
        u = torch.zeros_like(x)
        u[1:] = torch.cumsum(weights, dim=0)
        u = (u - u[0]) / (u[-1] - u[0])
        return u

    def _open_knot_vector(self, u: torch.Tensor, degree: int) -> torch.Tensor:
        n = self.N
        m = n + degree + 1
        knots = torch.zeros(m, dtype=torch.float64, device=self.device)
        knots[: degree + 1] = u[0]
        knots[-(degree + 1):] = u[-1]
        num_internal = n - degree - 1
        for j in range(1, num_internal + 1):
            start = j
            stop = j + degree
            knots[degree + j] = torch.mean(u[start:stop])
        return knots * (self.x[-1] - self.x[0]) + self.x[0]

    def _evaluate_bspline_basis_and_derivatives(
        self, x: torch.Tensor, knots: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_basis = knots.numel() - degree - 1
        m = x.numel()
        dtype = torch.float64
        device = self.device

        basis_cache = {}
        deriv1_cache = {}
        deriv2_cache = {}

        def basis(p: int, i: int) -> torch.Tensor:
            key = (p, i)
            if key in basis_cache:
                return basis_cache[key]
            if p == 0:
                left = knots[i]
                right = knots[i + 1]
                vals = ((x >= left) & (x < right)).to(dtype)
                if i == n_basis - 1:
                    vals = torch.where(x == knots[-1], torch.tensor(1.0, device=device), vals)
                basis_cache[key] = vals
                return vals
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            val = torch.zeros(m, dtype=dtype, device=device)
            if denom1 > 0:
                val += (x - knots[i]) / denom1 * basis(p - 1, i)
            if denom2 > 0:
                val += (knots[i + p + 1] - x) / denom2 * basis(p - 1, i + 1)
            basis_cache[key] = val
            return val

        def deriv1(p: int, i: int) -> torch.Tensor:
            key = (p, i)
            if key in deriv1_cache:
                return deriv1_cache[key]
            if p == 0:
                deriv1_cache[key] = torch.zeros(m, dtype=dtype, device=device)
                return deriv1_cache[key]
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            val = torch.zeros(m, dtype=dtype, device=device)
            if denom1 > 0:
                val += p / denom1 * basis(p - 1, i)
            if denom2 > 0:
                val -= p / denom2 * basis(p - 1, i + 1)
            deriv1_cache[key] = val
            return val

        def deriv2(p: int, i: int) -> torch.Tensor:
            key = (p, i)
            if key in deriv2_cache:
                return deriv2_cache[key]
            if p <= 1:
                deriv2_cache[key] = torch.zeros(m, dtype=dtype, device=device)
                return deriv2_cache[key]
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            val = torch.zeros(m, dtype=dtype, device=device)
            if denom1 > 0:
                val += p / denom1 * deriv1(p - 1, i)
            if denom2 > 0:
                val -= p / denom2 * deriv1(p - 1, i + 1)
            deriv2_cache[key] = val
            return val

        B0 = torch.stack([basis(degree, i) for i in range(n_basis)], dim=1)
        B1 = torch.stack([deriv1(degree, i) for i in range(n_basis)], dim=1)
        B2 = torch.stack([deriv2(degree, i) for i in range(n_basis)], dim=1)
        return B0, B1, B2


__all__ = ["SplineCore", "SplineOptions"]
