

from __future__ import annotations

import csv
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   
sys.path.insert(0, os.path.join(HERE, "solver", "fem"))   
sys.path.insert(0, os.path.join(HERE, "material"))        

import params_q2 as P                                      # noqa: E402
import torch                                               # noqa: E402
import fem_timoshenko as fem                               # noqa: E402
from material_properties import compute_material_params_for_solver  # noqa: E402


N_ELEM = 1500

N_LOAD_STEPS = 1


def main() -> None:
    torch.set_default_dtype(torch.float64)

    mat = dict(compute_material_params_for_solver(
        h=P.h, L=P.L, num_layers=P.num_layers, W_Gr=P.W_Gr, H_Gr=P.H_Gr,
        T=P.T, distribution_type=P.distribution, q=P.q,
    ))
    print("=" * 78)
    print("Q2 conventional baseline | FEM (standard Newton-Raphson, single load) | bending")
    print("=" * 78)
    print("Material coefficients (same material_properties.py as ODIL):")
    for k in ("a11", "b11", "d11", "a55", "lambda_val", "n_xT", "m_xT"):
        print(f"    {k:12s} = {mat[k]:.10e}")
    print(f"\nFEM: standard Newton-Raphson (single load, n_load_steps={N_LOAD_STEPS}), "
          f"n_elem={N_ELEM} (O(h^2)), bc={P.bc_type}")
    print(f"Nonlinear init (default): init_guess={P.INIT_GUESS}, seed={P.INIT_SEED}, "
          f"scale={P.INIT_SCALE}  (* = default config)\n")

    rows = []
    hist_rows = []
    for sc in P.SCENARIOS:
        name = sc["name"]
        ref = P.REFERENCE_W_MID[name]
        cfg = fem.FEMConfig(
            a11=mat["a11"], b11=mat["b11"], d11=mat["d11"], a55=mat["a55"],
            lambda_val=mat["lambda_val"], n_xT=mat["n_xT"], m_xT=mat["m_xT"],
            alpha_t=0.0, DeltaT=0.0,
            k1=sc["k1"], k2=sc["k2"], q=P.q, bc_type=P.bc_type, n_elem=N_ELEM,
        )

        
        rl = fem.solve_linear(cfg)
        _, w_lin, _ = fem.deflection_at(0.5, rl)

        print(f"--- {name}  (k1={sc['k1']}, k2={sc['k2']}) ---")
        print(f"  linear  w(0.5) = {w_lin:+.6f}   [ref {ref['linear']:+.5f}]   "
              f"K_sym_err={rl['K_sym_err']:.1e}")
        print(f"  nonlinear init sweep × standard Newton-Raphson "
              f"(ref nl w(0.5) = {ref['nonlinear']:+.5f}, scale={P.INIT_SCALE}, seed={P.INIT_SEED}):")
        print(f"    {'init':8s} | {'Newton: w(0.5)':>16s} {'|R|inf':>10s} {'t(ms)':>8s}")

        for ig in P.INIT_SWEEP:
            rn = fem.solve_nonlinear(
                cfg, n_load_steps=N_LOAD_STEPS,
                init_guess=ig, init_seed=P.INIT_SEED, init_scale=P.INIT_SCALE,
            )
            t_n = rn["elapsed_s"]   
            _, w_nl, _ = fem.deflection_at(0.5, rn)
            mark = "*" if ig == P.INIT_GUESS else " "
            print(f"    {ig:7s}{mark}| {w_nl:+16.6f} {rn['final_res']:10.1e} {t_n*1e3:8.1f}")
            rows.append([name, sc["k1"], sc["k2"], ig, P.INIT_SEED, P.INIT_SCALE,
                         w_lin, w_nl, ref["linear"], ref["nonlinear"],
                         float(rn["final_res"]), t_n, ig])

            for i, r in enumerate(rn.get("residual_history", []), 1):
                hist_rows.append([name, ig, i, r])
        print()

    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "fem_baseline.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scenario", "k1", "k2", "init_guess", "init_seed", "init_scale",
                       "w_linear", "w_nonlinear", "ref_linear", "ref_nonlinear",
                       "nl_final_res", "elapsed_s", "init_strategy"])
        wcsv.writerows(rows)
    print(f"[saved] {csv_path}")

    hist_path = os.path.join(outdir, "fem_residual_history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scenario", "init_guess", "iteration", "residual"])
        wcsv.writerows(hist_rows)
    print(f"[saved] {hist_path}")


if __name__ == "__main__":
    main()
