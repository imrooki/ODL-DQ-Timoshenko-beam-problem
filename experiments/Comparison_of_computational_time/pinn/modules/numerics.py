"""
PINNs numerical computation and automatic differentiation tools module

Author: Yang
Version: 1.0

Responsibilities:
- Provide high-precision automatic differentiation functions based on PyTorch autograd
- Implement numerically stable safe division operations, avoiding division by zero and numerical instability
- Provide arbitrary-order derivative computation, supporting high-order PDE solving
- Include numerical safety checks and exception handling mechanisms

Core functionality:
- d_dx(): first-order automatic differentiation
- compute_derivatives(): arbitrary-order derivative computation
- safe_divide(): numerically stable safe division operation

Technical features:
- Implements automatic differentiation via the PyTorch computational graph, ensuring derivative accuracy
- Supports backpropagation and retention of the gradient computational graph
- Built-in NaN/Inf detection and warning mechanisms
- Suitable for the complex systems of differential equations in Timoshenko beam theory

Design principles:
- This module focuses on numerical computation and does not involve specific physical formulas
- All functions preserve computational graph integrity, supporting higher-order differentiation
- Prioritizes numerical stability and computational accuracy
"""

from __future__ import annotations

from typing import Optional,  Union, Callable

import torch



def d_dx(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute the first-order derivative dy/dx (using autograd).

    Parameters:
        y: output tensor
        x: input tensor
        create_graph: whether to create the computational graph (set True only when a second-order derivative is needed)

    Memory optimization:
        - retain_graph=False avoids retaining an unnecessary computational graph
        - only_inputs=True computes gradients with respect to inputs only
    """
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=False,  # Do not retain the graph, to reduce memory
        only_inputs=True     # Compute gradients with respect to inputs only
    )[0]


def compute_derivatives(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute the arbitrary-order derivative d^n y / dx^n (chained autograd calls).

    Memory-optimized version:
        - The last-order derivative does not need create_graph
        - Do not retain the computational graph after each differentiation
    """
    grad = y
    for i in range(order):
        # create_graph is needed only when this is not the last order
        need_graph = (i < order - 1)
        grad = torch.autograd.grad(
            grad, x,
            grad_outputs=torch.ones_like(grad),
            create_graph=need_graph,  # Needed for intermediate steps only
            retain_graph=False,        # Do not retain the graph
            only_inputs=True           # Compute gradients with respect to inputs only
        )[0]
    return grad


def safe_divide(numerator: torch.Tensor, denominator: Union[float, torch.Tensor], eps: float = 1e-10) -> torch.Tensor:
    """Enhanced safe division, avoiding instability caused by division by zero or an overly small denominator, and including NaN detection.

    - If the denominator is a float: use max(|den|, eps) as a floor.
    - If the denominator is a torch.Tensor: clamp to [-inf, -eps]∪[eps, +inf], preserving the sign.
    - Adds NaN/Inf detection and handling
    """
    import warnings

    # Input validity check
    if torch.isnan(numerator).any():
        raise ValueError("Numerator contains NaN values")
    if torch.isinf(numerator).any():
        warnings.warn("Numerator contains Inf values, results may be unstable")

    if isinstance(denominator, float):
        if abs(denominator) < eps:
            warnings.warn(f"Small denominator detected: {denominator}, using eps protection")
        denom = max(abs(denominator), eps)
        result = numerator / denom
    else:
        # Check the denominator tensor
        if torch.isnan(denominator).any():
            raise ValueError("Denominator contains NaN values")
        if torch.isinf(denominator).any():
            warnings.warn("Denominator contains Inf values, results may be unstable")

        # Safe division handling
        safe_denom = torch.clamp(torch.abs(denominator), min=eps) * torch.sign(denominator)
        # Handle zero denominator
        zero_mask = torch.abs(denominator) < eps
        safe_denom = torch.where(zero_mask, eps, safe_denom)
        result = numerator / safe_denom

    # Result check
    if torch.isnan(result).any():
        raise RuntimeError("Safe divide produced NaN results - numerical instability detected")
    if torch.isinf(result).any():
        warnings.warn("Safe divide produced Inf results - potential numerical issues")
    
    return result


def mean_integral(density: torch.Tensor, factor: float = 1.0, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Integral approximation:
    - If no weights are provided: mean approximation ∫ f dx ≈ mean(f)*factor (default factor=1 means interval length=1).
    - If weights are provided: weighted sum ∫ f dx ≈ Σ w_i f(x_i), ignoring factor.
    """

    if weights is None:
        return torch.mean(density) * factor
    # Broadcast to a compatible shape, then compute the weighted sum
    if density.dim() == 2 and density.size(1) == 1:
        density = density.view(-1)
    w = weights.view(-1).to(dtype=density.dtype, device=density.device)
    return torch.sum(w * density.view(-1))


__all__ = ["d_dx", "compute_derivatives", "safe_divide", "mean_integral"]


def as_shape(
    value: Union[float, int, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    like: torch.Tensor,
) -> torch.Tensor:
    """Return ``value`` as a tensor shaped like ``like``.

    - If ``value`` is callable, evaluate it as ``value(like)``.
    - If ``value`` is a scalar or small tensor, broadcast to ``like``'s shape.
    - Always match dtype and device to ``like``.
    """

    if callable(value):
        out = value(like)
        return out.to(dtype=like.dtype, device=like.device)

    if isinstance(value, torch.Tensor):
        out = value.to(dtype=like.dtype, device=like.device)
        # Use broadcasting; adding zeros materializes a tensor with ``like``'s shape
        return out + torch.zeros_like(like)

    # Scalar path
    return torch.full_like(like, float(value))


__all__.append("as_shape")


def quad_nodes_weights(method: str, n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate quadrature nodes and weights on the interval [0,1].

    - method: 'gauss' (Gauss-Legendre), 'clenshaw' (Clenshaw-Curtis)
    - n: number of nodes (positive integer)
    Returns: x ∈ [0,1]^{n×1}, w ∈ R^n, satisfying Σ w_i ≈ 1
    """
    if n <= 0:
        raise ValueError("n must be positive")
    m = (method or "").lower()
    import numpy as np

    if m in ("gauss", "gauss_legendre", "legendre"):
        # Gauss-Legendre on [-1,1], then mapped to [0,1]
        t, w = np.polynomial.legendre.leggauss(int(n))
        x = 0.5 * (t + 1.0)
        w = 0.5 * w
        x_t = torch.tensor(x, device=device, dtype=dtype).reshape(-1, 1)
        w_t = torch.tensor(w, device=device, dtype=dtype).reshape(-1)
        return x_t.requires_grad_(True), w_t

    if m in ("clenshaw", "clenshaw_curtis", "cc"):
        if n == 1:
            x = np.array([0.5], dtype=np.float64)
            w = np.array([1.0], dtype=np.float64)
        else:
            N = n - 1
            k = np.arange(0, n, dtype=np.float64)
            theta = np.pi * k / N
            x_std = np.cos(theta)  # [-1,1]
            # Weights (Waldvogel formula)
            w = np.ones(n, dtype=np.float64)
            jmax = N // 2
            j = np.arange(1, jmax + 1, dtype=np.float64)
            coeff = 2.0 / (4.0 * j * j - 1.0)
            cos_term = np.cos(np.outer(2.0 * j, theta))
            w -= (coeff[:, None] * cos_term).sum(axis=0)
            if N % 2 == 0 and N > 0:
                jN = N // 2
                w -= (1.0 / (4.0 * jN * jN - 1.0)) * np.cos(N * theta)
            w *= 2.0 / N
            # Map to [0,1]
            x = 0.5 * (x_std + 1.0)
            w *= 0.5
        x_t = torch.tensor(x, device=device, dtype=dtype).reshape(-1, 1)
        w_t = torch.tensor(w, device=device, dtype=dtype).reshape(-1)
        return x_t.requires_grad_(True), w_t

    raise ValueError(f"Unknown quadrature method: {method}")


__all__.extend(["quad_nodes_weights"])


def sample_1d(N: int, device: torch.device, *, sampler: str = "uniform", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate 1D samples in [0,1] of shape (N,1) with gradients.

    samplers:
    - "uniform": torch.rand independent Uniform(0,1) (Monte Carlo)
    - "lhs": Latin Hypercube (pyDOE2 or pyDOE). Requires one of these packages.
    - "sobol": Low-discrepancy Sobol sequence (torch.quasirandom).
    """
    s = (sampler or "uniform").lower()
    if s == "uniform":
        x = torch.rand((N, 1), device=device, dtype=dtype)
        return x.requires_grad_(True)
    elif s == "sobol":
        try:
            from torch.quasirandom import SobolEngine
        except Exception as e:
            raise ImportError("Sobol sampler requires torch.quasirandom.SobolEngine") from e
        # Use current torch RNG to derive a reproducible seed
        seed = int(torch.initial_seed() & 0xFFFFFFFF)
        engine = SobolEngine(dimension=1, scramble=True, seed=seed)
        X = engine.draw(N)
        x = X.to(device=device, dtype=dtype)
        return x.requires_grad_(True)
    elif s == "lhs":
        # Prefer pyDOE2, fallback to pyDOE; if both unavailable, use a local 1D LHS implementation
        try:
            from pyDOE2 import lhs as _lhs  # type: ignore
            X = _lhs(1, samples=int(N), criterion="maximin", iterations=10)
        except Exception:
            try:
                from pyDOE import lhs as _lhs  # type: ignore
                X = _lhs(1, samples=int(N), criterion="maximin", iterations=10)
            except Exception:
                # Local lightweight 1D LHS with optional maximin selection
                import numpy as np

                def _lhs_1d_local(samples: int, iterations: int = 10) -> np.ndarray:
                    samples = int(samples)
                    if samples <= 0:
                        return np.zeros((0, 1), dtype=np.float64)
                    best = None
                    best_score = -1.0
                    # Precompute bin starts
                    bin_width = 1.0 / samples
                    starts = np.linspace(0.0, 1.0 - bin_width, samples)
                    rng = np.random
                    for _ in range(max(1, int(iterations))):
                        u = rng.rand(samples)
                        x = starts + u * bin_width
                        rng.shuffle(x)
                        xs = np.sort(x)
                        # include edges for fairness
                        diffs = np.diff(np.concatenate(([0.0], xs, [1.0])))
                        score = float(diffs.min())
                        if score > best_score:
                            best_score = score
                            best = x.copy()
                    return best.reshape(-1, 1)

                X = _lhs_1d_local(int(N), iterations=10)
        x = torch.tensor(X, device=device, dtype=dtype)
        return x.requires_grad_(True)
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Choose from 'uniform', 'lhs', 'sobol'.")


__all__.append("sample_1d")
