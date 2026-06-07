"""
DQ (Differential Quadrature) core functionality module

This module implements the core functionality of the Differential Quadrature method:
1. Chebyshev-Gauss-Lobatto node generation
2. DQ weighting coefficient computation
3. Derivative matrix construction
4. Numerical stability enhancement

Designed based on the ODIL framework (Optimizing a Discrete Loss)
"""

import torch
import math
import warnings
from typing import Tuple, Optional

# Set default precision
torch.set_default_dtype(torch.float64)

def cheb_lobatto_nodes(N: int, a: float = 0.0, b: float = 1.0, 
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate standard Chebyshev-Gauss-Lobatto nodes

    Parameters:
        N: number of nodes
        a, b: mapping interval [a, b]
        device: compute device

    Returns:
        x: node coordinates (increasing)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure N is an integer type (fix torch.arange argument error)
    if not isinstance(N, int):
        N = int(N.item() if torch.is_tensor(N) else N)

    # Standard CGL nodes
    # x = 0.5 * (1.0 - cos(i * pi / (N-1))), i=0:N-1
    i = torch.arange(N, dtype=torch.float64, device=device)
    x = 0.5 * (1.0 - torch.cos(i * math.pi / (N - 1)))

    # Ensure endpoints are exactly 0 and 1 (avoid floating-point error)
    x[0] = 0.0
    x[-1] = 1.0

    # Map to physical domain [a,b]
    xx = a + (b - a) * x
    return xx

def weighting_coefficients(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                     torch.Tensor, torch.Tensor]:
    """
    Weighting coefficient computation
    DQ weighting matrix based on Lagrange interpolation

    Parameters:
        X: node coordinates

    Returns:
        A: first-order derivative matrix
        B: second-order derivative matrix
        C: third-order derivative matrix
        D: fourth-order derivative matrix
    """
    N = len(X)
    device = X.device
    dtype = X.dtype

    # Step 1: Compute the Y vector (Y(I) = ∏(X(I)-X(J)), J≠I)
    Y = torch.ones(N, device=device, dtype=dtype)
    for I in range(N):
        for J in range(N):
            if I != J:
                Y[I] = Y[I] * (X[I] - X[J])

    # Step 2: Compute the first-order derivative matrix A
    A = torch.zeros((N, N), device=device, dtype=dtype)
    for I in range(N):
        sum_diag = 0.0
        for J in range(N):
            if I != J:
                # Off-diagonal element
                A[I, J] = Y[I] / (X[I] - X[J]) / Y[J]
                # Accumulate diagonal element
                sum_diag += 1.0 / (X[I] - X[J])
        # Set diagonal element
        A[I, I] = sum_diag

    # Step 3: Compute higher-order derivative matrices
    B = A @ A  # second-order derivative
    C = A @ B  # third-order derivative
    D = B @ B  # fourth-order derivative

    # Check numerical stability
    cond_A = torch.linalg.cond(A).item()
    if cond_A > 1e10:
        warnings.warn(f"First derivative matrix condition number too large: {cond_A:.2e}")
    
    return A, B, C, D

def weighting_coefficients_negsum(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                            torch.Tensor, torch.Tensor]:
    """
    Weighting coefficient computation using the negative-sum rule
    The diagonal elements use the negative-sum rule: A[i,i] = -Σ(A[i,j])

    Parameters:
        X: node coordinates

    Returns:
        A: first-order derivative matrix
        B: second-order derivative matrix
        C: third-order derivative matrix
        D: fourth-order derivative matrix
    """
    N = len(X)
    device = X.device
    dtype = X.dtype

    # Step 1: Compute the Y vector - direct product Y[i] = ∏(X[i]-X[j])
    Y = torch.ones(N, device=device, dtype=dtype)
    for I in range(N):
        for J in range(N):
            if I != J:
                Y[I] = Y[I] * (X[I] - X[J])

    # Step 2: Compute the first-order derivative matrix A - off-diagonal elements
    A = torch.zeros((N, N), device=device, dtype=dtype)
    for I in range(N):
        for J in range(N):
            if I != J:
                # Off-diagonal element
                A[I, J] = Y[I] / (X[I] - X[J]) / Y[J]

    # Step 3: Diagonal element computation - negative-sum rule A[i,i] = -Σ(A[i,j])
    for I in range(N):
        A[I, I] = -torch.sum(A[I, :])

    # Step 4: Compute higher-order derivative matrices
    B = A @ A  # second-order derivative
    C = A @ B  # third-order derivative
    D = B @ B  # fourth-order derivative

    # Check numerical stability
    cond_A = torch.linalg.cond(A).item()
    if cond_A > 1e10:
        warnings.warn(f"First derivative matrix condition number too large: {cond_A:.2e}")
    
    return A, B, C, D


def compute_derivatives(u: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                        C: Optional[torch.Tensor] = None, 
                        D: Optional[torch.Tensor] = None) -> dict:
    """
    Compute the derivatives of various orders of a function using DQ matrices

    Parameters:
        u: function values at the nodes
        A: first-order derivative matrix
        B: second-order derivative matrix
        C: third-order derivative matrix (optional)
        D: fourth-order derivative matrix (optional)

    Returns:
        Dictionary containing the derivatives of various orders
    """
    derivatives = {
        'u': u,
        'ux': A @ u,    # first-order derivative
        'uxx': B @ u,   # second-order derivative
    }

    if C is not None:
        derivatives['uxxx'] = C @ u  # third-order derivative

    if D is not None:
        derivatives['uxxxx'] = D @ u  # fourth-order derivative

    return derivatives

def check_matrix_condition(A: torch.Tensor, B: torch.Tensor, 
                          threshold: float = 1e10) -> dict:
    """
    Check the condition number of the DQ matrices to assess numerical stability
    Following the DQ expert's suggestion, lower the threshold from 1e12 to 1e10

    Parameters:
        A: first-order derivative matrix
        B: second-order derivative matrix
        threshold: condition number threshold (default 1e10)

    Returns:
        Condition number information dictionary
    """
    cond_A = torch.linalg.cond(A).item()
    cond_B = torch.linalg.cond(B).item()

    info = {
        'cond_A': cond_A,
        'cond_B': cond_B,
        'stable': cond_A < threshold and cond_B < threshold,
        'warning': None,
        'recommendation': None
    }

    # Graded warning system
    if cond_A > 1e12 or cond_B > 1e12:
        info['warning'] = f"Condition number severely large - A: {cond_A:.2e}, B: {cond_B:.2e}"
        info['recommendation'] = "Suggest reducing node count or using regularization"
        warnings.warn(info['warning'])
    elif cond_A > threshold or cond_B > threshold:
        info['warning'] = f"Large condition number - A: {cond_A:.2e}, B: {cond_B:.2e}"
        info['recommendation'] = "Suggest using preconditioning techniques"
        warnings.warn(info['warning'])
    
    return info

def get_chebyshev_weights(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chebyshev weight function (used for the weighted L2 norm)

    Parameters:
        x: node coordinates (normalized to [0,1])

    Returns:
        Chebyshev weights
    """
    # Map [0,1] to [-1,1]
    xi = 2*x - 1
    # Chebyshev weight function
    weights = torch.sqrt(1 - xi**2 + 1e-10)  # add a small value to avoid division by zero
    return weights

def precompute_dq_system(N: int, a: float = 0.0, b: float = 1.0,
                        device: Optional[torch.device] = None,
                        check_stability: bool = True,
                        dq_method: str = 'original',
                        # Taylor/Fornberg local-method parameters
                        x_nodes: str = 'cheb',
                        fd_stencil_size: int = 5,
                        fd_build_orders: tuple = (1, 2),
                        fd_build_C_D: bool = False,
                        fd_B_from_A: bool = False,
                        # GPU acceleration parameters
                        enable_gpu_acceleration: bool = True) -> dict:
    """
    Precompute all necessary matrices and information for the DQ system

    Parameters:
        N: number of nodes
        a, b: physical domain interval
        device: compute device
        check_stability: whether to check numerical stability
        dq_method: weighting computation method
                  ('original', 'negative_sum', 'fornberg_local')

        # Taylor/Fornberg local-method-specific parameters:
        x_nodes: node type ('cheb' or 'uniform')
        fd_stencil_size: local stencil size (odd, 5 or 7 recommended)
        fd_build_orders: tuple of derivative orders to construct
        fd_build_C_D: whether to construct third- and fourth-order matrices
        fd_B_from_A: whether to construct via B=A@A (not recommended)

        # GPU acceleration parameters:
        enable_gpu_acceleration: whether to enable GPU acceleration (applies to traditional DQ methods)

    Returns:
        Dictionary containing all DQ system information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select node generation and matrix computation approach based on dq_method
    if dq_method == 'fornberg_local':
        # === Taylor/Fornberg local-method branch ===

        # 1. Node generation
        if x_nodes == 'uniform':
            x = uniform_nodes(N, a, b, device)
        else:  # default to 'cheb'
            x = cheb_lobatto_nodes(N, a, b, device)

        # 2. Determine the derivative orders to construct
        build_orders = list(fd_build_orders)
        if fd_build_C_D:
            if 3 not in build_orders:
                build_orders.append(3)
            if 4 not in build_orders:
                build_orders.append(4)

        # 3. Construct matrices using the optimized Taylor/Fornberg core
        try:
            from .taylor_core import create_taylor_fornberg_system

            print(f"Using optimized Taylor/Fornberg local method:")
            print(f"  - Node type: {x_nodes}")
            print(f"  - Stencil size: {fd_stencil_size}")
            print(f"  - Construction orders: {build_orders}")
            print(f"  - Storage format: dense (ODIL-compatible)")

            # Create the optimized system
            taylor_system = create_taylor_fornberg_system(
                x=x,
                stencil_size=fd_stencil_size,
                orders=tuple(build_orders),
                sparse_format='dense',  # ODIL compatibility, use dense format
                device=device
            )

            matrices = taylor_system['matrices']
            A = matrices.get('A')
            B = matrices.get('B')
            C = matrices.get('C')
            D = matrices.get('D')

            # Print sparsity analysis
            sparsity_info = taylor_system.get('sparsity_analysis', {})
            if sparsity_info:
                print(f"  - Matrix sparsity analysis:")
                for name, analysis in sparsity_info.items():
                    if analysis:
                        print(f"    {name}: sparsity={analysis['sparsity_ratio']:.1%}, nonzero elements={analysis['nonzero_elements']}")

        except ImportError:
            # Fall back to the original implementation
            warnings.warn("taylor_core module unavailable, falling back to original Fornberg implementation")

            mode = 'from_A' if fd_B_from_A else 'direct'
            matrices = build_local_fd_matrices(
                x,
                stencil_size=fd_stencil_size,
                orders=tuple(build_orders),
                mode=mode,
                device=device
            )

            A = matrices['A']
            B = matrices['B']
            C = matrices['C']
            D = matrices['D']

            print(f"Using Taylor/Fornberg local method (fallback version):")
            print(f"  - Node type: {x_nodes}")
            print(f"  - Stencil size: {fd_stencil_size}")
            print(f"  - Construction orders: {build_orders}")

    else:
        # === Traditional DQ method branch ===

        # Generate Chebyshev nodes
        x = cheb_lobatto_nodes(N, a, b, device)

        # Select the derivative matrix computation based on the method
        if enable_gpu_acceleration and device.type == 'cuda' and dq_method in ['original', 'negative_sum']:
            # GPU-accelerated DQ weighting computation
            try:
                from ..utils.gpu_acceleration import GPUAccelerator
                print(f"Using GPU-accelerated DQ computation (method: {dq_method})")

                accelerator = GPUAccelerator(device)
                A, B = accelerator.optimize_dq_computation(x)

                # Compute higher-order derivative matrices using GPU acceleration
                C = torch.mm(A, B)  # third-order derivative
                D = torch.mm(B, B)  # fourth-order derivative

                print(f"GPU-accelerated DQ computation completed - N={N}")

            except ImportError:
                warnings.warn("GPU acceleration module unavailable, falling back to CPU computation")
                if dq_method == 'negative_sum':
                    A, B, C, D = weighting_coefficients_negsum(x)
                else:
                    A, B, C, D = weighting_coefficients(x)
        else:
            # Standard CPU computation
            if dq_method == 'negative_sum':
                A, B, C, D = weighting_coefficients_negsum(x)
            else:  # default to 'original'
                A, B, C, D = weighting_coefficients(x)
    
    # Check the condition number (optional)
    cond_info = None
    if check_stability and A is not None and B is not None:
        cond_info = check_matrix_condition(A, B)

        # Give different stability recommendations for different methods
        if dq_method == 'fornberg_local':
            # The Fornberg method is usually more stable and can handle larger N
            if N > 51:
                warnings.warn(f"Node count N={N} is large, suggest monitoring Fornberg method numerical accuracy")
        else:
            # Stability recommendation for the traditional DQ method
            if N > 21:
                warnings.warn(f"Node count N={N} too large, may cause numerical instability. Suggest N≤21 or switch to fornberg_local method")

    # Compute the weights (for Chebyshev nodes)
    x_normalized = (x - a) / (b - a)
    if x_nodes == 'cheb' or dq_method != 'fornberg_local':
        cheb_weights = get_chebyshev_weights(x_normalized)
    else:
        # For uniform nodes, use unit weights
        cheb_weights = torch.ones_like(x)

    return {
        'x': x,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'N': N,
        'domain': (a, b),
        'cheb_weights': cheb_weights,
        'condition_info': cond_info,
        'device': device,
        'dq_method': dq_method,
        # Save additional information for the Fornberg method
        'fornberg_info': {
            'x_nodes': x_nodes,
            'stencil_size': fd_stencil_size,
            'build_orders': fd_build_orders,
            'build_C_D': fd_build_C_D,
            'B_from_A': fd_B_from_A
        } if dq_method == 'fornberg_local' else None
    }

# Caching mechanism (optional)
_dq_cache = {}

def get_cached_dq_system(N: int, a: float = 0.0, b: float = 1.0,
                        device: Optional[torch.device] = None,
                        dq_method: str = 'original',
                        # Taylor/Fornberg parameters
                        x_nodes: str = 'cheb',
                        fd_stencil_size: int = 5,
                        fd_build_orders: tuple = (1, 2),
                        fd_build_C_D: bool = False,
                        fd_B_from_A: bool = False,
                        # GPU acceleration parameters
                        enable_gpu_acceleration: bool = True) -> dict:
    """
    Get the cached DQ system (avoids redundant computation)

    Parameters:
        N: number of nodes
        a, b: physical domain interval
        device: compute device
        dq_method: weighting computation method ('original', 'negative_sum', 'fornberg_local')

        # Taylor/Fornberg local-method parameters:
        x_nodes: node type ('cheb' or 'uniform')
        fd_stencil_size: local stencil size
        fd_build_orders: derivative orders to construct
        fd_build_C_D: whether to construct higher-order matrices
        fd_B_from_A: whether to construct B from A

        # GPU acceleration parameters:
        enable_gpu_acceleration: whether to enable GPU acceleration

    Returns:
        DQ system dictionary
    """
    # Build the cache key, including all relevant parameters
    cache_key = (
        N, a, b, str(device), dq_method,
        x_nodes, fd_stencil_size, fd_build_orders,
        fd_build_C_D, fd_B_from_A, enable_gpu_acceleration
    )

    if cache_key not in _dq_cache:
        _dq_cache[cache_key] = precompute_dq_system(
            N, a, b, device,
            check_stability=True,
            dq_method=dq_method,
            x_nodes=x_nodes,
            fd_stencil_size=fd_stencil_size,
            fd_build_orders=fd_build_orders,
            fd_build_C_D=fd_build_C_D,
            fd_B_from_A=fd_B_from_A,
            enable_gpu_acceleration=enable_gpu_acceleration
        )

    return _dq_cache[cache_key]

# ==============================================================================
# Taylor/Fornberg local finite-difference method implementation
# ==============================================================================

def fd_weights_fornberg_torch(x0: torch.Tensor, x: torch.Tensor, m: int) -> torch.Tensor:
    """
    PyTorch implementation of the Fornberg algorithm - compute local derivative weights

    Computes the weights for the m-th derivative at an arbitrary point x0
    Returns weights w such that sum_j w[j]*f(x[j]) ≈ f^{(m)}(x0)

    Parameters:
        x0: target point (scalar tensor or float)
        x: stencil node coordinates (1D tensor, length s)
        m: derivative order (0, 1, 2, ...)

    Returns:
        w: derivative weights (1D tensor, length s)

    Note:
        - x must be unique and contain x0 (or x0 very close to some point in x)
        - The algorithm is insensitive to node ordering, but requiring x to be increasing benefits numerical stability
    """
    # Ensure the input is a tensor and on the same device
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, device=x.device, dtype=x.dtype)

    n = len(x)
    device = x.device
    dtype = x.dtype

    # Initialize the weight table: w_table[i,k] denotes the weight for computing the k-th derivative using the first i+1 nodes
    w = torch.zeros((n, m+1), device=device, dtype=dtype)

    # Initial condition (starting point of the Fornberg recurrence)
    w[0, 0] = 1.0
    c1 = torch.tensor(1.0, device=device, dtype=dtype)
    c4 = x[0] - x0

    # Fornberg recurrence process
    for i in range(1, n):
        mn = min(i, m)  # the highest derivative order currently computable
        c2 = torch.tensor(1.0, device=device, dtype=dtype)
        c5 = c4
        c4 = x[i] - x0

        # Inner loop: update the weights of all previous nodes
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3

            # When j == i-1, update the weights of the newly added node
            if j == i - 1:
                for k in range(mn, 0, -1):  # update from high order to low order
                    # w[i, k] = c1 * (k * w[i-1, k-1] - c5 * w[i-1, k]) / c2
                    w[i, k] = c1 * (k * w[i-1, k-1] - c5 * w[i-1, k]) / c2
                # w[i, 0] = -c1 * c5 * w[i-1, 0] / c2
                w[i, 0] = -c1 * c5 * w[i-1, 0] / c2

            # Update the weights of the old nodes
            for k in range(mn, 0, -1):
                # w[j, k] = (c4 * w[j, k] - k * w[j, k-1]) / c3
                w[j, k] = (c4 * w[j, k] - k * w[j, k-1]) / c3
            # w[j, 0] = c4 * w[j, 0] / c3
            w[j, 0] = c4 * w[j, 0] / c3

        c1 = c2

    # Return the weight vector for the m-th derivative
    return w[:, m]

def select_stencil_indices(i: int, N: int, s: int) -> torch.Tensor:
    """
    Select the indices of the local stencil for the i-th node

    Parameters:
        i: current node index (0 <= i < N)
        N: total number of nodes
        s: stencil size (odd, 5 or 7 recommended)

    Returns:
        idx: stencil node indices (LongTensor, length s, ascending)

    Strategy:
        - Interior point: use a symmetric stencil [i-h, ..., i, ..., i+h], where h=(s-1)//2
        - Left boundary: use a one-sided stencil [0, 1, ..., s-1]
        - Right boundary: use a one-sided stencil [N-s, ..., N-2, N-1]
    """
    assert s % 2 == 1, f"Stencil size s={s} must be odd"
    assert s <= N, f"Stencil size s={s} cannot exceed total node count N={N}"

    h = (s - 1) // 2  # half-stencil width

    if i < h:
        # Left boundary: use a left one-sided stencil
        idx = torch.arange(0, s, dtype=torch.long)
    elif i > N - 1 - h:
        # Right boundary: use a right one-sided stencil
        idx = torch.arange(N - s, N, dtype=torch.long)
    else:
        # Interior point: use a symmetric stencil
        idx = torch.arange(i - h, i + h + 1, dtype=torch.long)

    return idx

def uniform_nodes(N: int, a: float = 0.0, b: float = 1.0,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate uniformly distributed nodes (as an alternative to Chebyshev nodes)

    Parameters:
        N: number of nodes
        a, b: interval endpoints
        device: compute device

    Returns:
        x: uniform node coordinates
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate uniform nodes
    x = torch.linspace(a, b, N, device=device, dtype=torch.float64)
    return x

def build_local_fd_matrices(x: torch.Tensor, stencil_size: int = 5,
                           orders: tuple = (1, 2), mode: str = 'direct',
                           device: Optional[torch.device] = None) -> dict:
    """
    Construct sparse derivative matrices based on local Fornberg weights

    Parameters:
        x: node coordinates (1D tensor)
        stencil_size: local stencil size (odd, 5 or 7 recommended)
        orders: derivative orders to construct (e.g. (1,2) means first and second order)
        mode: construction mode ('direct' or 'from_A')
        device: compute device

    Returns:
        Derivative matrix dictionary {'A': first-order, 'B': second-order, 'C': third-order, 'D': fourth-order}

    Core idea:
        - For each node i, select the local stencil indices idx
        - Compute the Fornberg weights on that stencil
        - Fill the weights into the i-th row of the global sparse matrix
    """
    N = len(x)
    if device is None:
        device = x.device
    dtype = x.dtype

    # Initialize the derivative matrix dictionary
    matrices = {}

    # Construct the corresponding matrices based on orders
    for order in orders:
        if order == 1:
            A = torch.zeros((N, N), device=device, dtype=dtype)

            # Construct local weights for each node
            for i in range(N):
                # Select the local stencil
                idx = select_stencil_indices(i, N, stencil_size)
                x_local = x[idx]
                x0 = x[i]

                # Compute the Fornberg weights
                w = fd_weights_fornberg_torch(x0, x_local, 1)

                # Fill into the global matrix
                A[i, idx] = w

            matrices['A'] = A

        elif order == 2:
            if mode == 'direct':
                # Directly compute the second-order derivative matrix
                B = torch.zeros((N, N), device=device, dtype=dtype)

                for i in range(N):
                    idx = select_stencil_indices(i, N, stencil_size)
                    x_local = x[idx]
                    x0 = x[i]

                    w = fd_weights_fornberg_torch(x0, x_local, 2)
                    B[i, idx] = w

                matrices['B'] = B

            elif mode == 'from_A' and 'A' in matrices:
                # Obtain via matrix multiplication of A (not recommended, expands the bandwidth)
                matrices['B'] = matrices['A'] @ matrices['A']

        elif order == 3:
            # Third-order derivative matrix
            C = torch.zeros((N, N), device=device, dtype=dtype)

            for i in range(N):
                idx = select_stencil_indices(i, N, stencil_size)
                x_local = x[idx]
                x0 = x[i]

                w = fd_weights_fornberg_torch(x0, x_local, 3)
                C[i, idx] = w

            matrices['C'] = C

        elif order == 4:
            # Fourth-order derivative matrix
            D = torch.zeros((N, N), device=device, dtype=dtype)

            for i in range(N):
                idx = select_stencil_indices(i, N, stencil_size)
                x_local = x[idx]
                x0 = x[i]

                w = fd_weights_fornberg_torch(x0, x_local, 4)
                D[i, idx] = w

            matrices['D'] = D

    # Fill in the missing matrices (maintain interface compatibility)
    if 'A' not in matrices:
        matrices['A'] = None
    if 'B' not in matrices:
        matrices['B'] = None
    if 'C' not in matrices:
        matrices['C'] = None
    if 'D' not in matrices:
        matrices['D'] = None

    return matrices