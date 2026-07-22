

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


torch.set_default_dtype(torch.float64)







def _material_dir() -> str:
    
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "material"))


def load_material_params(
    h: float,
    L: float,
    num_layers: int,
    W_Gr: float,
    H_Gr: float,
    T: float,
    distribution: str,
    q: float,
) -> Dict[str, float]:
    
    matdir = _material_dir()
    if matdir not in sys.path:
        sys.path.insert(0, matdir)
    from material_properties import compute_material_params_for_solver  # type: ignore

    mp = compute_material_params_for_solver(
        h=h, L=L, num_layers=num_layers, W_Gr=W_Gr, H_Gr=H_Gr,
        T=T, distribution_type=distribution, q=q,
    )
    return dict(mp)






@dataclass
class FEMConfig:
    
    a11: float
    b11: float
    d11: float
    a55: float
    lambda_val: float          
    n_xT: float = 0.0
    m_xT: float = 0.0
    alpha_t: float = 0.0       
    DeltaT: float = 0.0        
    k1: float = 0.0            
    k2: float = 0.0            
    q: float = -0.08           
    bc_type: str = "C-C"       
    n_elem: int = 200          
    device: str = "cpu"        

    @property
    def n_node(self) -> int:
        return self.n_elem + 1

    @property
    def n_dof(self) -> int:
        return 3 * self.n_node






_G2_XI = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
_G2_W = (0.5, 0.5)  






def total_energy(
    d: torch.Tensor,
    cfg: FEMConfig,
    q_load: float,
    is_nonlinear: bool,
) -> torch.Tensor:
    
    n_node = cfg.n_node
    n_elem = cfg.n_elem
    he = 1.0 / n_elem  
    lam = cfg.lambda_val

    U = d[0:n_node]
    W = d[n_node:2 * n_node]
    P = d[2 * n_node:3 * n_node]

    
    u1, u2 = U[:-1], U[1:]
    w1, w2 = W[:-1], W[1:]
    p1, p2 = P[:-1], P[1:]

    
    ux = (u2 - u1) / he
    wx = (w2 - w1) / he
    px = (p2 - p1) / he
    phi_mid = 0.5 * (p1 + p2)  

    
    if is_nonlinear:
        g = (wx * wx) / (2.0 * lam)
    else:
        g = torch.zeros_like(wx)

    
    term1 = cfg.a11 * ux + cfg.b11 * px + cfg.a11 * g - lam * cfg.n_xT
    term2 = ux + g - lam * cfg.alpha_t * cfg.DeltaT
    term3 = cfg.b11 * ux + cfg.d11 * px + cfg.b11 * g - lam * cfg.m_xT

    
    strain_nonshear = term1 * term2 + term3 * px

    
    shear_mid = cfg.a55 * (wx + lam * phi_mid) ** 2

    Pi_str = 0.5 * he * torch.sum(strain_nonshear + shear_mid)

    
    Pi_w = 0.5 * cfg.k2 * he * torch.sum(wx * wx)
    if cfg.k1 != 0.0:
        acc_k1 = torch.zeros((), dtype=d.dtype, device=d.device)
        for xi, wt in zip(_G2_XI, _G2_W):
            wg = w1 * (1.0 - xi) + w2 * xi
            acc_k1 = acc_k1 + torch.sum(wg * wg) * wt
        Pi_w = Pi_w + 0.5 * cfg.k1 * he * acc_k1

    
    acc_e = torch.zeros((), dtype=d.dtype, device=d.device)
    for xi, wt in zip(_G2_XI, _G2_W):
        wg = w1 * (1.0 - xi) + w2 * xi
        acc_e = acc_e + torch.sum(wg) * wt
    Pi_e = q_load * he * acc_e

    return Pi_str + Pi_w - Pi_e






def free_dof_mask(cfg: FEMConfig) -> torch.Tensor:
    
    n_node = cfg.n_node
    nL, nR = 0, n_node - 1

    def idx(comp: int, node: int) -> int:
        
        return comp * n_node + node

    free = torch.ones(cfg.n_dof, dtype=torch.bool)

    bt = cfg.bc_type.upper().strip()
    if bt in ("C-C",):
        ends = {nL: "C", nR: "C"}
    elif bt in ("H-H", "S-S"):
        ends = {nL: "H", nR: "H"}
    elif bt in ("C-H", "C-S"):
        ends = {nL: "C", nR: "H"}
    elif bt in ("C-F",):
        ends = {nL: "C", nR: "F"}
    elif bt in ("H-C", "S-C"):
        ends = {nL: "H", nR: "C"}
    elif bt in ("F-C",):
        ends = {nL: "F", nR: "C"}
    else:
        raise ValueError(f"不支持的边界类型: {cfg.bc_type}")

    for node, kind in ends.items():
        if kind == "C":          
            for comp in (0, 1, 2):
                free[idx(comp, node)] = False
        elif kind in ("H", "S"):  
            free[idx(0, node)] = False
            free[idx(1, node)] = False
        elif kind == "F":         
            pass

    return free






def solve_linear(cfg: FEMConfig) -> Dict[str, torch.Tensor]:
    
    from torch.func import grad, hessian

    n_dof = cfg.n_dof
    d0 = torch.zeros(n_dof, requires_grad=False)

    def Pi(dvec: torch.Tensor) -> torch.Tensor:
        return total_energy(dvec, cfg, q_load=cfg.q, is_nonlinear=False)

    
    grad0 = grad(Pi)(d0)              
    K = hessian(Pi)(d0)              
    F = -grad0

    free = free_dof_mask(cfg)
    fidx = torch.where(free)[0]

    K_ff = K[fidx][:, fidx]
    F_f = F[fidx]

    d_free = torch.linalg.solve(K_ff, F_f)
    d = torch.zeros(n_dof)
    d[fidx] = d_free

    
    sym_err = float(torch.max(torch.abs(K_ff - K_ff.T)).item())

    return _split(d, cfg, extra={"K_sym_err": sym_err, "n_free": int(fidx.numel())})






def solve_nonlinear(
    cfg: FEMConfig,
    d_init: Optional[torch.Tensor] = None,
    n_load_steps: int = 1,
    tol: float = 1e-10,
    max_newton: int = 50,
    verbose: bool = False,
    init_guess: str = "random",
    init_seed: int = 0,
    init_scale: float = 0.01,
) -> Dict[str, torch.Tensor]:
    
    from torch.func import grad as fgrad, hessian as fhess

    n_dof = cfg.n_dof
    free = free_dof_mask(cfg)
    fidx = torch.where(free)[0]

    
    t0 = time.time()

    
    ig = str(init_guess).lower().strip()
    if d_init is not None:
        d = d_init.clone()
        ig = "provided"
    elif ig == "linear":
        d = solve_linear(cfg)["d"].clone()
    elif ig == "zero":
        d = torch.zeros(n_dof)
    elif ig in ("random", "gauss"):
        gen = torch.Generator()
        gen.manual_seed(int(init_seed))
        if ig == "gauss":
            d = init_scale * torch.randn(n_dof, generator=gen, dtype=torch.float64)
        else:  
            d = init_scale * (2.0 * torch.rand(n_dof, generator=gen, dtype=torch.float64) - 1.0)
    else:
        raise ValueError(
            f"init_guess must be 'random'/'gauss'/'zero'/'linear', got {init_guess!r}"
        )

    
    
    d[~free] = 0.0

    q_target = cfg.q
    history = []
    res_history = []   

    for step in range(1, n_load_steps + 1):
        q_cur = q_target * (step / n_load_steps)

        def Pi(dvec: torch.Tensor, _q=q_cur) -> torch.Tensor:
            return total_energy(dvec, cfg, q_load=_q, is_nonlinear=True)

        for it in range(1, max_newton + 1):
            R_full = fgrad(Pi)(d)                 
            R_f = R_full[fidx]
            rnorm = float(torch.max(torch.abs(R_f)).item())
            res_history.append(rnorm)
            if rnorm < tol:
                break

            K_full = fhess(Pi)(d)                 
            K_ff = K_full[fidx][:, fidx]
            delta_f = torch.linalg.solve(K_ff, -R_f)

            
            alpha = 1.0
            base = rnorm
            d_try = d.clone()
            for _ls in range(25):
                d_try = d.clone()
                d_try[fidx] = d[fidx] + alpha * delta_f
                r_try = float(torch.max(torch.abs(fgrad(Pi)(d_try)[fidx])).item())
                if r_try < base or alpha < 1e-6:
                    break
                alpha *= 0.5
            d = d_try

        history.append((step, q_cur, it, rnorm))
        if verbose:
            print(f"  [load step {step}/{n_load_steps}] q={q_cur:+.5f}  "
                  f"newton_iter={it}  |R|inf={rnorm:.3e}")

    elapsed = time.time() - t0
    out = _split(d, cfg, extra={"n_free": int(fidx.numel())})
    out["history"] = history
    out["residual_history"] = res_history   
    out["final_res"] = history[-1][3] if history else float("nan")
    out["init_guess"] = ig
    out["init_seed"] = int(init_seed)
    out["elapsed_s"] = elapsed
    return out






def _split(d: torch.Tensor, cfg: FEMConfig, extra: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    n_node = cfg.n_node
    x = torch.linspace(0.0, 1.0, n_node)
    res = {
        "x": x,
        "u": d[0:n_node].detach().clone(),
        "w": d[n_node:2 * n_node].detach().clone(),
        "phi": d[2 * n_node:3 * n_node].detach().clone(),
        "d": d.detach().clone(),
    }
    if extra:
        res.update(extra)
    return res






def deflection_at(x_query: float, res: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
    
    x = res["x"].numpy()
    u = res["u"].numpy()
    w = res["w"].numpy()
    p = res["phi"].numpy()
    return (
        float(np.interp(x_query, x, u)),
        float(np.interp(x_query, x, w)),
        float(np.interp(x_query, x, p)),
    )


if __name__ == "__main__":
    
    cfg = FEMConfig(a11=1.0, b11=0.0, d11=1.0, a55=1.0, lambda_val=20.0,
                    k1=0.0, k2=0.0, q=-0.08, bc_type="C-C", n_elem=100)
    rl = solve_linear(cfg)
    _, w_mid, _ = deflection_at(0.5, rl)
    print(f"[smoke] linear  C-C 中点 w = {w_mid:.6e}  (应为负)  K_sym_err={rl['K_sym_err']:.2e}")
    rn = solve_nonlinear(cfg, d_init=rl["d"], n_load_steps=6, verbose=True)
    _, w_mid_n, _ = deflection_at(0.5, rn)
    print(f"[smoke] nonlinear C-C 中点 w = {w_mid_n:.6e}  |R|inf={rn['final_res']:.2e}")
