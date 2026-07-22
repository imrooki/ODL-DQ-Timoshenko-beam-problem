

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   
sys.path.insert(0, os.path.join(HERE, "solver", "dq"))    
sys.path.insert(0, os.path.join(HERE, "material"))        

import params_q2 as P                                      # noqa: E402
import torch                                               # noqa: E402
import direct_dq                                           # noqa: E402
from dq_core import (                                      # noqa: E402
    cheb_lobatto_nodes,
    weighting_coefficients_negsum,
)
from material_properties import compute_material_params_for_solver  # noqa: E402


def _to_np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float64)


def main() -> None:
    torch.set_default_dtype(torch.float64)

    
    mat = dict(compute_material_params_for_solver(
        h=P.h, L=P.L, num_layers=P.num_layers, W_Gr=P.W_Gr, H_Gr=P.H_Gr,
        T=P.T, distribution_type=P.distribution, q=P.q,
    ))
    print("=" * 78)
    print("Q2 conventional baseline | Direct DQ (Newton-Raphson / Picard) | bending")
    print("=" * 78)
    print("Material coefficients (same material_properties.py as ODIL):")
    for k in ("a11", "b11", "d11", "a55", "lambda_val", "n_xT", "m_xT"):
        print(f"    {k:12s} = {mat[k]:.10e}")

    
    x = cheb_lobatto_nodes(P.N)
    A, B, _, _ = weighting_coefficients_negsum(x)
    A_np, B_np, x_np = _to_np(A), _to_np(B), _to_np(x)
    mid = int(np.argmin(np.abs(x_np - 0.5)))   
    print(f"\nDQ grid: N={P.N}, x[mid]={x_np[mid]:.6f} (should be 0.5), bc={P.bc_type}")
    print(f"Nonlinear init (default): init_guess={P.INIT_GUESS}, seed={P.INIT_SEED}, "
          f"scale={P.INIT_SCALE}  (* = default config)\n")

    rows = []
    hist_rows = []
    for sc in P.SCENARIOS:
        found = {"k1": sc["k1"], "k2": sc["k2"]}
        name = sc["name"]
        ref = P.REFERENCE_W_MID[name]

        
        _, w_l, _ = direct_dq.solve_bending_linear_direct(
            P.N, A_np, B_np, mat, q=P.q, bc_type=P.bc_type, foundation_params=found,
        )
        w_lin = float(w_l[mid])

        print(f"--- {name}  (k1={sc['k1']}, k2={sc['k2']}) ---")
        print(f"  linear  w(0.5) = {w_lin:+.6f}   [ref {ref['linear']:+.5f}]")
        print(f"  nonlinear init x method sweep "
              f"(ref nl w(0.5) = {ref['nonlinear']:+.5f}, scale={P.INIT_SCALE}, seed={P.INIT_SEED}):")
        print(f"    {'init':8s} | {'Newton: w(0.5)':>13s} {'it':>3s} {'conv':>5s} {'t(ms)':>8s} "
              f"| {'Picard: w(0.5)':>13s} {'it':>3s} {'conv':>5s} {'t(ms)':>8s}")

        for ig in P.INIT_SWEEP:
            
            _, wn, _, info_n = direct_dq.solve_bending_nonlinear_direct(
                P.N, A_np, B_np, mat, q=P.q, bc_type=P.bc_type, foundation_params=found,
                iteration_method="newton",
                init_guess=ig, init_seed=P.INIT_SEED, init_scale=P.INIT_SCALE,
            )
            t_n = info_n["elapsed_s"]
            wn_mid = float(wn[mid])

            
            _, wp, _, info_p = direct_dq.solve_bending_nonlinear_direct(
                P.N, A_np, B_np, mat, q=P.q, bc_type=P.bc_type, foundation_params=found,
                iteration_method="picard",
                init_guess=ig, init_seed=P.INIT_SEED, init_scale=P.INIT_SCALE,
            )
            t_p = info_p["elapsed_s"]
            wp_mid = float(wp[mid])

            mark = "*" if ig == P.INIT_GUESS else " "
            print(f"    {ig:7s}{mark}| {wn_mid:+13.6f} {info_n['iterations']:>3d} "
                  f"{str(info_n['converged']):>5s} {t_n*1e3:7.2f} | {wp_mid:+13.6f} {info_p['iterations']:>3d} "
                  f"{str(info_p['converged']):>5s} {t_p*1e3:7.2f}")

            rows.append([name, sc["k1"], sc["k2"], ig, P.INIT_SEED, P.INIT_SCALE,
                         "newton", wn_mid, info_n["iterations"], bool(info_n["converged"]),
                         ref["nonlinear"], t_n])
            rows.append([name, sc["k1"], sc["k2"], ig, P.INIT_SEED, P.INIT_SCALE,
                         "picard", wp_mid, info_p["iterations"], bool(info_p["converged"]),
                         ref["nonlinear"], t_p])

            for i, r in enumerate(info_n.get("residual_history", []), 1):
                hist_rows.append([name, ig, "newton", i, r])
            for i, r in enumerate(info_p.get("residual_history", []), 1):
                hist_rows.append([name, ig, "picard", i, r])
        print()

    
    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "dq_baseline.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scenario", "k1", "k2", "init_guess", "init_seed", "init_scale",
                       "method", "w_mid", "iters", "converged", "ref_nonlinear", "elapsed_s"])
        wcsv.writerows(rows)
    print(f"[saved] {csv_path}")

    hist_path = os.path.join(outdir, "dq_residual_history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scenario", "init_guess", "method", "iteration", "residual"])
        wcsv.writerows(hist_rows)
    print(f"[saved] {hist_path}")


if __name__ == "__main__":
    main()
