

import torch
import math
import time
from typing import Tuple, Optional


torch.set_default_dtype(torch.float64)

def cheb_lobatto_nodes(N: int, a: float = 0.0, b: float = 1.0,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if not isinstance(N, int):
        N = int(N.item() if torch.is_tensor(N) else N)

    
    
    
    i = torch.arange(N, dtype=torch.float64, device=device)
    x = 0.5 * (1.0 - torch.cos(i * math.pi / (N - 1)))

    
    x[0] = 0.0
    x[-1] = 1.0

    
    xx = a + (b - a) * x
    return xx

def _effective_cond(sv: torch.Tensor, rel_tol: float = 1e-8) -> float:
    
    if sv.numel() < 2:
        return 1.0
    sig = sv[sv > sv[0] * rel_tol]
    if sig.numel() < 2:
        return 1.0
    return (sig[0] / sig[-1]).item()

def weighting_coefficients(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
    
    N = len(X)
    device = X.device
    dtype = X.dtype
    
    
    
    
    Y = torch.ones(N, device=device, dtype=dtype)
    for I in range(N):
        for J in range(N):
            if I != J:
                Y[I] = Y[I] * (X[I] - X[J])

    
    A = torch.zeros((N, N), device=device, dtype=dtype)
    for I in range(N):
        sum_diag = 0.0
        for J in range(N):
            if I != J:
                
                
                A[I, J] = Y[I] / (X[I] - X[J]) / Y[J]
                
                sum_diag += 1.0 / (X[I] - X[J])
        
        A[I, I] = sum_diag

    
    
    B = A @ A  
    C = A @ B  
    D = B @ B  

    
    
    
    eff_cond_A = _effective_cond(torch.linalg.svdvals(A))
    if eff_cond_A > 1e10:
        print(f"[WARNING] First derivative matrix effective condition number too large: {eff_cond_A:.2e}")

    return A, B, C, D

def weighting_coefficients_negsum(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, torch.Tensor]:
    
    N = len(X)
    device = X.device
    dtype = X.dtype
    
    
    
    Y = torch.ones(N, device=device, dtype=dtype)
    for I in range(N):
        for J in range(N):
            if I != J:
                Y[I] = Y[I] * (X[I] - X[J])

    
    
    A = torch.zeros((N, N), device=device, dtype=dtype)
    for I in range(N):
        for J in range(N):
            if I != J:
                A[I, J] = Y[I] / (X[I] - X[J]) / Y[J]

    
    
    
    for I in range(N):
        A[I, I] = -torch.sum(A[I, :])

    
    B = A @ A  
    C = A @ B  
    D = B @ B  

    
    
    
    eff_cond_A = _effective_cond(torch.linalg.svdvals(A))
    if eff_cond_A > 1e10:
        print(f"[WARNING] First derivative matrix effective condition number too large: {eff_cond_A:.2e}")

    return A, B, C, D


def compute_derivatives(u: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                        C: Optional[torch.Tensor] = None,
                        D: Optional[torch.Tensor] = None) -> dict:
    
    derivatives = {
        'u': u,
        'ux': A @ u,    
        'uxx': B @ u,   
    }
    
    if C is not None:
        derivatives['uxxx'] = C @ u  
    
    if D is not None:
        derivatives['uxxxx'] = D @ u  
    
    return derivatives

def check_matrix_condition(A: torch.Tensor, B: torch.Tensor,
                          threshold: float = 1e10) -> dict:
    
    
    
    
    cond_A = _effective_cond(torch.linalg.svdvals(A))
    cond_B = _effective_cond(torch.linalg.svdvals(B))

    info = {
        'cond_A': cond_A,
        'cond_B': cond_B,
        'stable': cond_A < threshold and cond_B < threshold,
        'warning': None,
        'recommendation': None
    }
    
    
    if cond_A > 1e12 or cond_B > 1e12:
        info['warning'] = f"Effective condition number severely large - A: {cond_A:.2e}, B: {cond_B:.2e}"
        info['recommendation'] = "Suggest reducing node count or using regularization"
        print(f"[WARNING] {info['warning']}")
    elif cond_A > threshold or cond_B > threshold:
        info['warning'] = f"Large effective condition number - A: {cond_A:.2e}, B: {cond_B:.2e}"
        info['recommendation'] = "Suggest using preconditioning techniques"
        print(f"[WARNING] {info['warning']}")
    
    return info

def interpolate_to_fine_grid(u_coarse: torch.Tensor, N_coarse: int,
                            N_fine: int, device: Optional[torch.device] = None) -> torch.Tensor:
    
    if device is None:
        device = u_coarse.device
    
    
    x_coarse = cheb_lobatto_nodes(N_coarse, device=device)
    x_fine = cheb_lobatto_nodes(N_fine, device=device)
    
    
    
    from torch.nn.functional import interpolate
    
    
    u_coarse_reshaped = u_coarse.unsqueeze(0).unsqueeze(0)  
    
    
    u_fine = interpolate(u_coarse_reshaped, size=N_fine, mode='linear', 
                         align_corners=True)
    
    return u_fine.squeeze()

def get_chebyshev_weights(x: torch.Tensor) -> torch.Tensor:
    
    
    xi = 2*x - 1
    
    
    weights = torch.sqrt(1 - xi**2 + 1e-10)
    return weights

def precompute_dq_system(N: int, a: float = 0.0, b: float = 1.0,
                        device: Optional[torch.device] = None,
                        check_stability: bool = True,
                        dq_method: str = 'original',
                        
                        x_nodes: str = 'cheb',
                        fd_stencil_size: int = 5,
                        fd_build_orders: tuple = (1, 2),
                        fd_build_C_D: bool = False,
                        fd_B_from_A: bool = False,
                        
                        enable_gpu_acceleration: bool = True) -> dict:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if dq_method == 'fornberg_local':
        

        
        if x_nodes == 'uniform':
            x = uniform_nodes(N, a, b, device)
        else:  
            x = cheb_lobatto_nodes(N, a, b, device)

        
        build_orders = list(fd_build_orders)
        if fd_build_C_D:
            if 3 not in build_orders:
                build_orders.append(3)
            if 4 not in build_orders:
                build_orders.append(4)

        
        try:
            from .taylor_core import create_taylor_fornberg_system

            print(f"Using optimized Taylor/Fornberg local method:")
            print(f"  - Node type: {x_nodes}")
            print(f"  - Stencil size: {fd_stencil_size}")
            print(f"  - Build orders: {build_orders}")
            print(f"  - Storage format: dense (for DQ-PINNs)")

            
            taylor_system = create_taylor_fornberg_system(
                x=x,
                stencil_size=fd_stencil_size,
                orders=tuple(build_orders),
                sparse_format='dense',  
                device=device
            )

            matrices = taylor_system['matrices']
            A = matrices.get('A')
            B = matrices.get('B')
            C = matrices.get('C')
            D = matrices.get('D')

            
            sparsity_info = taylor_system.get('sparsity_analysis', {})
            if sparsity_info:
                print(f"  - Matrix sparsity analysis:")
                for name, analysis in sparsity_info.items():
                    if analysis:
                        print(f"    {name}: Sparsity={analysis['sparsity_ratio']:.1%}, Nonzero elements={analysis['nonzero_elements']}")

        except ImportError:
            
            print("[WARNING] taylor_core module unavailable, falling back to original Fornberg implementation")

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
            print(f"  - Build orders: {build_orders}")

    else:
        

        
        x = cheb_lobatto_nodes(N, a, b, device)

        
        if dq_method == 'negative_sum':
            A, B, C, D = weighting_coefficients_negsum(x)
        else:  
            A, B, C, D = weighting_coefficients(x)
    
    
    cond_info = None
    if check_stability and A is not None and B is not None:
        cond_info = check_matrix_condition(A, B)

        
        if dq_method == 'fornberg_local':
            
            if N > 99:
                print(f"[WARNING] Node count N={N} is large, suggest monitoring Fornberg method numerical accuracy")
        else:
            
            if N > 99:
                print(f"[WARNING] Node count N={N} too large, may cause numerical instability. Suggest N<=99 or switch to fornberg_local method")

    
    x_normalized = (x - a) / (b - a)
    if x_nodes == 'cheb' or dq_method != 'fornberg_local':
        cheb_weights = get_chebyshev_weights(x_normalized)
    else:
        
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
        
        'fornberg_info': {
            'x_nodes': x_nodes,
            'stencil_size': fd_stencil_size,
            'build_orders': fd_build_orders,
            'build_C_D': fd_build_C_D,
            'B_from_A': fd_B_from_A
        } if dq_method == 'fornberg_local' else None
    }


_dq_cache = {}

def get_cached_dq_system(N: int, a: float = 0.0, b: float = 1.0,
                        device: Optional[torch.device] = None,
                        dq_method: str = 'original',
                        
                        x_nodes: str = 'cheb',
                        fd_stencil_size: int = 5,
                        fd_build_orders: tuple = (1, 2),
                        fd_build_C_D: bool = False,
                        fd_B_from_A: bool = False,
                        
                        enable_gpu_acceleration: bool = True) -> dict:
    
    
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

def clear_dq_cache():
    
    global _dq_cache
    _dq_cache.clear()






def fd_weights_fornberg_torch(x0: torch.Tensor, x: torch.Tensor, m: int) -> torch.Tensor:
    
    
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, device=x.device, dtype=x.dtype)

    n = len(x)
    device = x.device
    dtype = x.dtype

    
    
    w = torch.zeros((n, m+1), device=device, dtype=dtype)

    
    w[0, 0] = 1.0
    
    c1 = torch.tensor(1.0, device=device, dtype=dtype)
    
    c4 = x[0] - x0

    
    for i in range(1, n):
        mn = min(i, m)  
        c2 = torch.tensor(1.0, device=device, dtype=dtype)  
        c5 = c4           
        c4 = x[i] - x0   

        
        for j in range(i):
            c3 = x[i] - x[j]  
            c2 = c2 * c3       

            
            if j == i - 1:
                
                for k in range(mn, 0, -1):
                    
                    w[i, k] = c1 * (k * w[i-1, k-1] - c5 * w[i-1, k]) / c2
                
                w[i, 0] = -c1 * c5 * w[i-1, 0] / c2

            
            for k in range(mn, 0, -1):
                
                w[j, k] = (c4 * w[j, k] - k * w[j, k-1]) / c3
            
            w[j, 0] = c4 * w[j, 0] / c3

        
        c1 = c2

    
    return w[:, m]

def select_stencil_indices(i: int, N: int, s: int) -> torch.Tensor:
    
    assert s % 2 == 1, f"Stencil size s={s} must be odd"
    assert s <= N, f"Stencil size s={s} cannot exceed total node count N={N}"

    h = (s - 1) // 2  

    if i < h:
        
        idx = torch.arange(0, s, dtype=torch.long)
    elif i > N - 1 - h:
        
        idx = torch.arange(N - s, N, dtype=torch.long)
    else:
        
        idx = torch.arange(i - h, i + h + 1, dtype=torch.long)

    return idx

def uniform_nodes(N: int, a: float = 0.0, b: float = 1.0,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    x = torch.linspace(a, b, N, device=device, dtype=torch.float64)
    return x


def create_nodes(N: int, node_type: str = 'cheb', a: float = 0.0, b: float = 1.0,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    
    if node_type == 'uniform':
        return uniform_nodes(N, a, b, device)
    else:
        return cheb_lobatto_nodes(N, a, b, device)


def build_local_fd_matrices(x: torch.Tensor, stencil_size: int = 5,
                           orders: tuple = (1, 2), mode: str = 'direct',
                           device: Optional[torch.device] = None) -> dict:
    
    N = len(x)
    if device is None:
        device = x.device
    dtype = x.dtype

    
    matrices = {}

    
    for order in orders:
        if order == 1:
            A = torch.zeros((N, N), device=device, dtype=dtype)

            
            for i in range(N):
                
                idx = select_stencil_indices(i, N, stencil_size)
                x_local = x[idx]
                x0 = x[i]

                
                w = fd_weights_fornberg_torch(x0, x_local, 1)

                
                A[i, idx] = w

            matrices['A'] = A

        elif order == 2:
            if mode == 'direct':
                
                B = torch.zeros((N, N), device=device, dtype=dtype)

                for i in range(N):
                    idx = select_stencil_indices(i, N, stencil_size)
                    x_local = x[idx]
                    x0 = x[i]

                    w = fd_weights_fornberg_torch(x0, x_local, 2)
                    B[i, idx] = w

                matrices['B'] = B

            elif mode == 'from_A' and 'A' in matrices:
                
                matrices['B'] = matrices['A'] @ matrices['A']

        elif order == 3:
            
            C = torch.zeros((N, N), device=device, dtype=dtype)

            for i in range(N):
                idx = select_stencil_indices(i, N, stencil_size)
                x_local = x[idx]
                x0 = x[i]

                w = fd_weights_fornberg_torch(x0, x_local, 3)
                C[i, idx] = w

            matrices['C'] = C

        elif order == 4:
            
            D = torch.zeros((N, N), device=device, dtype=dtype)

            for i in range(N):
                idx = select_stencil_indices(i, N, stencil_size)
                x_local = x[idx]
                x0 = x[i]

                w = fd_weights_fornberg_torch(x0, x_local, 4)
                D[i, idx] = w

            matrices['D'] = D

    
    if 'A' not in matrices:
        matrices['A'] = None
    if 'B' not in matrices:
        matrices['B'] = None
    if 'C' not in matrices:
        matrices['C'] = None
    if 'D' not in matrices:
        matrices['D'] = None

    return matrices
