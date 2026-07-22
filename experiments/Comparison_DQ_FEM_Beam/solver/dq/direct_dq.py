

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np











def _bending_moment_row_weak(
    N: int, b11: float, d11: float, A_np: np.ndarray,
    lambda_val: float, iNode: int,
    w_linearization: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    row = np.zeros(3 * N, dtype=np.float64)
    A_row = A_np[iNode, :]
    row[:N]           = b11 * A_row                             
    row[2*N:3*N]      = d11 * A_row                             
    if w_linearization is not None:
        w_prev = np.asarray(w_linearization, dtype=np.float64).reshape(-1)
        wx_prev = float(A_row @ w_prev)
        row[N:2*N] = row[N:2*N] + 0.5 * (b11 / lambda_val) * wx_prev * A_row
    return row


def _bending_axial_row_strong(
    N: int, a11: float, b11: float, A_np: np.ndarray,
    lambda_val: float, iNode: int,
    w_linearization: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    row = np.zeros(3 * N, dtype=np.float64)
    A_row = A_np[iNode, :]
    row[:N]           = a11 * A_row                              
    row[2*N:3*N]      = b11 * A_row                              
    if w_linearization is not None:
        w_prev = np.asarray(w_linearization, dtype=np.float64).reshape(-1)
        wx_prev = float(A_row @ w_prev)
        row[N:2*N] = row[N:2*N] + 0.5 * (a11 / lambda_val) * wx_prev * A_row
    return row


def _bending_shear_row(
    N: int, a55: float, k2: float, lambda_val: float,
    A_np: np.ndarray, iNode: int,
) -> np.ndarray:
    
    row = np.zeros(3 * N, dtype=np.float64)
    A_row = A_np[iNode, :]
    row[N:2*N] = (a55 + k2) * A_row
    row[2*N + iNode] = row[2*N + iNode] + a55 * lambda_val
    return row


def _compute_bending_residual_and_tangent(
    u, w, phi, A_np, B_np,
    a11, b11, d11, a55, lambda_val, n_xT, q_vec,
    k1, k2,
    compute_tangent=True,
):
    
    N = A_np.shape[0]
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    if np.isscalar(q_vec):
        q_vec = np.full(N, float(q_vec))
    else:
        q_vec = np.asarray(q_vec, dtype=np.float64).reshape(-1)

    has_found = (k1 != 0) or (k2 != 0)

    
    du_dx = A_np @ u
    d2u_dx2 = B_np @ u
    dw_dx = A_np @ w
    d2w_dx2 = B_np @ w
    dphi_dx = A_np @ phi
    d2phi_dx2 = B_np @ phi

    
    
    R1 = (a11 * d2u_dx2 + b11 * d2phi_dx2
          + (a11 / lambda_val) * (dw_dx * d2w_dx2))

    
    R2 = ((a11 / lambda_val) * (d2u_dx2 * dw_dx + du_dx * d2w_dx2)
          + (3.0 * a11 / (2.0 * lambda_val ** 2)) * d2w_dx2 * (dw_dx ** 2)
          + (b11 / lambda_val) * (d2phi_dx2 * dw_dx + dphi_dx * d2w_dx2)
          + n_xT * d2w_dx2
          + a55 * (d2w_dx2 + lambda_val * dphi_dx)
          + q_vec)
    if has_found:
        R2 = R2 - k1 * w + k2 * d2w_dx2

    
    R3 = (b11 * d2u_dx2 + d11 * d2phi_dx2
          + (b11 / lambda_val) * (dw_dx * d2w_dx2)
          - a55 * lambda_val * (dw_dx + lambda_val * phi))

    R = np.concatenate([R1, R2, R3])

    if not compute_tangent:
        return R, None

    
    I_N = np.eye(N)
    
    K11 = a11 * B_np
    K12 = (a11 / lambda_val) * (np.diag(dw_dx) @ B_np + np.diag(d2w_dx2) @ A_np)
    K13 = b11 * B_np

    
    K21 = (a11 / lambda_val) * (np.diag(d2w_dx2) @ A_np + np.diag(dw_dx) @ B_np)
    
    
    K22 = ((a11 / lambda_val) * (np.diag(d2u_dx2) @ A_np + np.diag(du_dx) @ B_np)
           + (3.0 * a11 / (2.0 * lambda_val ** 2)) * (
               np.diag(dw_dx ** 2) @ B_np + 2.0 * np.diag(d2w_dx2 * dw_dx) @ A_np)
           + (b11 / lambda_val) * (np.diag(d2phi_dx2) @ A_np + np.diag(dphi_dx) @ B_np)
           + n_xT * B_np
           + a55 * B_np)
    K23 = ((b11 / lambda_val) * (np.diag(dw_dx) @ B_np + np.diag(d2w_dx2) @ A_np)
           + a55 * lambda_val * A_np)

    
    K31 = b11 * B_np
    K32 = ((b11 / lambda_val) * (np.diag(dw_dx) @ B_np + np.diag(d2w_dx2) @ A_np)
           - a55 * lambda_val * A_np)
    K33 = d11 * B_np - a55 * (lambda_val ** 2) * I_N

    if has_found:
        K22 = K22 - k1 * I_N + k2 * B_np

    K_T = np.block([
        [K11, K12, K13],
        [K21, K22, K23],
        [K31, K32, K33],
    ])

    return R, K_T






def solve_bending_linear_direct(
    N: int,
    A_np: np.ndarray,
    B_np: np.ndarray,
    mat: Dict[str, float],
    q: float,
    *,
    bc_type: str = "C-C",
    foundation_params: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    from dq_stiffness import (
        assemble_linear_stiffness_matrix, get_boundary_dofs,
    )

    k1 = float(foundation_params.get("k1", 0.0)) if foundation_params else 0.0
    k2 = float(foundation_params.get("k2", 0.0)) if foundation_params else 0.0

    K_full = assemble_linear_stiffness_matrix(
        N, mat["a11"], mat["a55"], mat["b11"], mat["d11"],
        mat["n_xT"], mat["lambda_val"], A_np, B_np,
        k1=k1, k2=k2,
    )
    if not isinstance(K_full, np.ndarray):
        K_full = np.asarray(K_full)
    K_full = K_full.astype(np.float64).copy()

    F_full = np.zeros(3 * N, dtype=np.float64)
    F_full[N: 2 * N] = -float(q)

    bc_dofs = get_boundary_dofs(bc_type, N)
    for dof in bc_dofs:
        K_full[dof, :] = 0.0
        K_full[:, dof] = 0.0
        K_full[dof, dof] = 1.0
        F_full[dof] = 0.0

    
    
    if bc_type in ("H-H", "S-S", "C-H", "C-S"):
        if bc_type in ("H-H", "S-S"):
            h_end_nodes = [0, N - 1]
        else:  
            h_end_nodes = [N - 1]
        for iNode in h_end_nodes:
            row_M = _bending_moment_row_weak(
                N, mat["b11"], mat["d11"], A_np, mat["lambda_val"], iNode,
                w_linearization=None,  
            )
            K_full[2 * N + iNode, :] = row_M
            F_full[2 * N + iNode] = float(mat.get("m_xT", 0.0))

    
    
    if bc_type == "C-F":
        
        row_N = _bending_axial_row_strong(
            N, mat["a11"], mat["b11"], A_np, mat["lambda_val"], iNode=N - 1,
            w_linearization=None,
        )
        K_full[N - 1, :] = row_N
        F_full[N - 1] = float(mat.get("n_xT", 0.0))

        
        row_Q = _bending_shear_row(
            N, mat["a55"], k2, mat["lambda_val"], A_np, iNode=N - 1,
        )
        K_full[2 * N - 1, :] = row_Q
        F_full[2 * N - 1] = 0.0

        
        row_M = _bending_moment_row_weak(
            N, mat["b11"], mat["d11"], A_np, mat["lambda_val"], iNode=N - 1,
            w_linearization=None,
        )
        K_full[3 * N - 1, :] = row_M
        F_full[3 * N - 1] = float(mat.get("m_xT", 0.0))

    q_solution = np.linalg.solve(K_full, F_full)
    u = q_solution[0: N]
    w = q_solution[N: 2 * N]
    phi = q_solution[2 * N: 3 * N]
    return u, w, phi






def _apply_bc_newton(K_T, R, u, w, phi, bc_type, A_np, mat, k2, m_xT, n_xT_val):
    
    N = A_np.shape[0]
    b11 = mat['b11']; d11 = mat['d11']
    a11 = mat['a11']; a55 = mat['a55']
    lam = mat['lambda_val']

    def _apply_end_newton(iNode, end_type):
        ru = iNode
        rw = N + iNode
        rp = 2 * N + iNode
        A_row = A_np[iNode, :]
        
        K_T[[ru, rw, rp], :] = 0.0
        R[[ru, rw, rp]] = 0.0

        if end_type == 'C':
            
            K_T[ru, iNode] = 1.0
            R[ru] = u[iNode]
            K_T[rw, N + iNode] = 1.0
            R[rw] = w[iNode]
            K_T[rp, 2 * N + iNode] = 1.0
            R[rp] = phi[iNode]

        elif end_type == 'H':
            
            K_T[ru, iNode] = 1.0
            R[ru] = u[iNode]
            K_T[rw, N + iNode] = 1.0
            R[rw] = w[iNode]

            
            dw_dx = float(A_row @ w)
            gM = (b11 * float(A_row @ u)
                  + d11 * float(A_row @ phi)
                  + (b11 / (2 * lam)) * dw_dx ** 2)
            gM = gM - m_xT
            R[rp] = gM

            
            K_T[rp, 0:N] = b11 * A_row
            K_T[rp, 2 * N:3 * N] = d11 * A_row
            K_T[rp, N:2 * N] = (b11 / lam) * dw_dx * A_row

        elif end_type == 'F':
            
            dw_dx = float(A_row @ w)

            
            gN = (a11 * float(A_row @ u) + b11 * float(A_row @ phi)
                  + (a11 / (2 * lam)) * dw_dx ** 2)
            gN = gN - n_xT_val
            R[ru] = gN
            K_T[ru, 0:N] = a11 * A_row
            K_T[ru, 2 * N:3 * N] = b11 * A_row
            K_T[ru, N:2 * N] = (a11 / lam) * dw_dx * A_row

            
            gQ = (a55 + k2) * dw_dx + a55 * lam * phi[iNode]
            R[rw] = gQ
            K_T[rw, N:2 * N] = (a55 + k2) * A_row
            K_T[rw, 2 * N + iNode] = a55 * lam  

            
            gM = (b11 * float(A_row @ u)
                  + d11 * float(A_row @ phi)
                  + (b11 / (2 * lam)) * dw_dx ** 2)
            gM = gM - m_xT
            R[rp] = gM
            K_T[rp, 0:N] = b11 * A_row
            K_T[rp, 2 * N:3 * N] = d11 * A_row
            K_T[rp, N:2 * N] = (b11 / lam) * dw_dx * A_row
        else:
            raise ValueError(f"Unknown end type: {end_type}")

    
    bc_map = {
        'C-C': ('C', 'C'),
        'H-H': ('H', 'H'),
        'S-S': ('H', 'H'),
        'C-H': ('C', 'H'),
        'C-S': ('C', 'H'),
        'C-F': ('C', 'F'),
    }
    if bc_type not in bc_map:
        raise ValueError(f"Unknown bc_type for Newton: {bc_type}")
    left_type, right_type = bc_map[bc_type]
    _apply_end_newton(0, left_type)
    _apply_end_newton(N - 1, right_type)

    return K_T, R


def _make_initial_uwphi(
    N, init_guess, seed, scale,
    A_np, B_np, mat, q, bc_type, foundation_params,
):
    
    ig = str(init_guess).lower().strip()
    if ig == "linear":
        u0, w0, phi0 = solve_bending_linear_direct(
            N, A_np, B_np, mat, q=float(q), bc_type=bc_type,
            foundation_params=foundation_params,
        )
        return (u0.astype(np.float64).copy(),
                w0.astype(np.float64).copy(),
                phi0.astype(np.float64).copy())
    if ig == "zero":
        return (np.zeros(N, dtype=np.float64),
                np.zeros(N, dtype=np.float64),
                np.zeros(N, dtype=np.float64))
    if ig in ("random", "gauss"):
        rng = np.random.default_rng(int(seed))
        s = float(scale)
        if ig == "gauss":
            return (s * rng.standard_normal(N),
                    s * rng.standard_normal(N),
                    s * rng.standard_normal(N))
        
        return (rng.uniform(-s, s, N),
                rng.uniform(-s, s, N),
                rng.uniform(-s, s, N))
    raise ValueError(
        f"init_guess must be 'random' / 'gauss' / 'zero' / 'linear', got {init_guess!r}"
    )


def solve_bending_nonlinear_direct(
    N: int,
    A_np: np.ndarray,
    B_np: np.ndarray,
    mat: Dict[str, float],
    q: float,
    *,
    bc_type: str = "C-C",
    foundation_params: Optional[Dict[str, float]] = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    verbose: bool = False,
    iteration_method: str = "newton",
    init_guess: str = "random",
    init_seed: int = 0,
    init_scale: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    
    from dq_stiffness import (
        assemble_linear_stiffness_matrix,
        assemble_nonlinear_stiffness_matrix,
        get_boundary_dofs,
    )

    k1 = float(foundation_params.get("k1", 0.0)) if foundation_params else 0.0
    k2 = float(foundation_params.get("k2", 0.0)) if foundation_params else 0.0

    iteration_method = str(iteration_method).lower().strip()
    if iteration_method not in ('newton', 'picard'):
        raise ValueError(f"iteration_method must be 'newton' or 'picard', got {iteration_method!r}")

    
    
    t0 = time.time()

    
    if iteration_method == 'newton':
        
        u, w, phi = _make_initial_uwphi(
            N, init_guess, init_seed, init_scale,
            A_np, B_np, mat, q, bc_type, foundation_params,
        )

        m_xT_val = float(mat.get('m_xT', 0.0))
        n_xT_val = float(mat.get('n_xT', 0.0))
        q_vec = np.full(N, float(q), dtype=np.float64)

        converged = False
        last_res = float('inf')
        final_iter = 0
        res_hist = []
        for k in range(1, max_iter + 1):
            R, K_T = _compute_bending_residual_and_tangent(
                u, w, phi, A_np, B_np,
                mat['a11'], mat['b11'], mat['d11'], mat['a55'],
                mat['lambda_val'], n_xT_val, q_vec,
                k1, k2, compute_tangent=True,
            )
            
            
            
            R_int = np.concatenate([R[1:N - 1], R[N + 1:2 * N - 1], R[2 * N + 1:3 * N - 1]])
            
            K_T, R = _apply_bc_newton(
                K_T, R, u, w, phi, bc_type, A_np, mat,
                k2, m_xT_val, n_xT_val,
            )
            
            dX = np.linalg.solve(K_T, -R)
            u = u + dX[:N]
            w = w + dX[N:2 * N]
            phi = phi + dX[2 * N:3 * N]

            last_res = float(np.linalg.norm(R_int))   
            final_iter = k
            res_hist.append(last_res)
            if verbose:
                print(f"    [Newton bending direct_dq] iter {k}: ||R_PDE||_2(interior) = {last_res:.3e}")
            if last_res < tol:
                converged = True
                break

        elapsed = time.time() - t0
        info = {
            "iterations": final_iter,
            "final_rel_change": last_res,  
            "residual_history": res_hist,  
            "converged": bool(converged),
            "method": "newton_nonlinear_bending",
            "init_guess": str(init_guess).lower().strip(),
            "init_seed": int(init_seed),
            "elapsed_s": elapsed,
        }
        return u, w, phi, info

    
    K_L_full = assemble_linear_stiffness_matrix(
        N, mat["a11"], mat["a55"], mat["b11"], mat["d11"],
        mat["n_xT"], mat["lambda_val"], A_np, B_np, k1=k1, k2=k2,
    )
    if not isinstance(K_L_full, np.ndarray):
        K_L_full = np.asarray(K_L_full)
    K_L_full = K_L_full.astype(np.float64)

    F_full = np.zeros(3 * N, dtype=np.float64)
    F_full[N: 2 * N] = -float(q)

    bc_dofs = list(get_boundary_dofs(bc_type, N))

    def _apply_bc_and_solve(K: np.ndarray, F: np.ndarray,
                             w_lin: Optional[np.ndarray] = None) -> np.ndarray:
        
        K_bc = K.copy()
        F_bc = F.copy()
        for dof in bc_dofs:
            K_bc[dof, :] = 0.0
            K_bc[:, dof] = 0.0
            K_bc[dof, dof] = 1.0
            F_bc[dof] = 0.0

        
        
        if bc_type in ("H-H", "S-S", "C-H", "C-S"):
            if bc_type in ("H-H", "S-S"):
                h_end_nodes = [0, N - 1]
            else:  
                h_end_nodes = [N - 1]
            for iNode in h_end_nodes:
                row_M = _bending_moment_row_weak(
                    N, mat["b11"], mat["d11"], A_np, mat["lambda_val"], iNode,
                    w_linearization=w_lin,
                )
                K_bc[2 * N + iNode, :] = row_M
                F_bc[2 * N + iNode] = float(mat.get("m_xT", 0.0))

        
        
        if bc_type == "C-F":
            row_N = _bending_axial_row_strong(
                N, mat["a11"], mat["b11"], A_np, mat["lambda_val"], iNode=N - 1,
                w_linearization=w_lin,
            )
            K_bc[N - 1, :] = row_N
            F_bc[N - 1] = float(mat.get("n_xT", 0.0))

            row_Q = _bending_shear_row(
                N, mat["a55"], k2, mat["lambda_val"], A_np, iNode=N - 1,
            )
            K_bc[2 * N - 1, :] = row_Q
            F_bc[2 * N - 1] = 0.0

            row_M = _bending_moment_row_weak(
                N, mat["b11"], mat["d11"], A_np, mat["lambda_val"], iNode=N - 1,
                w_linearization=w_lin,
            )
            K_bc[3 * N - 1, :] = row_M
            F_bc[3 * N - 1] = float(mat.get("m_xT", 0.0))

        return np.linalg.solve(K_bc, F_bc)

    
    _u0, _w0, _phi0 = _make_initial_uwphi(
        N, init_guess, init_seed, init_scale,
        A_np, B_np, mat, q, bc_type, foundation_params,
    )
    d = np.concatenate([_u0, _w0, _phi0])

    converged = False
    last_rel_change = float("inf")
    final_iter = 0
    res_hist = []
    for k in range(max_iter):
        K_NL = assemble_nonlinear_stiffness_matrix(
            N, mat["a11"], mat["b11"], mat["lambda_val"],
            A_np, B_np, d,
        )
        if not isinstance(K_NL, np.ndarray):
            K_NL = np.asarray(K_NL)
        K_NL = K_NL.astype(np.float64)

        K_eff = K_L_full + K_NL
        
        w_prev = d[N: 2 * N] if bc_type == "C-F" else None
        d_new = _apply_bc_and_solve(K_eff, F_full, w_lin=w_prev)

        rel_change = float(
            np.linalg.norm(d_new - d) / max(np.linalg.norm(d_new), 1e-20)
        )
        last_rel_change = rel_change
        final_iter = k + 1
        res_hist.append(rel_change)
        if verbose:
            print(f"    [Picard bending direct_dq] iter {k + 1}: rel_change={rel_change:.3e}")
        d = d_new
        if rel_change < tol:
            converged = True
            break

    elapsed = time.time() - t0
    info = {
        "iterations": final_iter,
        "final_rel_change": last_rel_change,
        "residual_history": res_hist,  
        "converged": bool(converged),
        "method": "picard_nonlinear_bending",
        "init_guess": str(init_guess).lower().strip(),
        "init_seed": int(init_seed),
        "elapsed_s": elapsed,
    }
    return d[:N], d[N: 2 * N], d[2 * N:], info







def solve_direct_dq(problem_type: str, **kwargs) -> Any:
    
    pt = problem_type.lower().strip()
    if pt == "bending_linear":
        return solve_bending_linear_direct(**kwargs)
    if pt == "bending_nonlinear":
        return solve_bending_nonlinear_direct(**kwargs)
    raise ValueError(
        f"Unknown problem_type: {problem_type!r}. "
        "Expected one of: bending_linear / bending_nonlinear. "
        "(This Q2 baseline covers bending only.)"
    )


__all__ = [
    "solve_bending_linear_direct",
    "solve_bending_nonlinear_direct",
    "solve_direct_dq",
]
