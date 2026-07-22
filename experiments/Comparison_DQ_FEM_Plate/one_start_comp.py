







import csv
import os
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "solver", "dq"))
sys.path.insert(0, os.path.join(HERE, "material"))
sys.path.insert(0, os.path.join(HERE, "solver", "odl"))
sys.path.insert(0, os.path.join(HERE, "solver", "fem"))

import params_q3 as P
from material_plate import build_goeam_material, compute_D0_reference
import direct_dq_plate as dq
import fem_plate_fsdt as fem
from modules.postprocess import extract_center_deflection
from modules.odl_solver import residual_solve
from run_odl_plate import (odl_strong_soft_nonlinear, DEV,
                           lbfgs_mg_kwargs, mg_prolongations_for_lbfgs,
                           ODL_SOFT_LINEAR_MAX_ITER_LBFGS)

torch.set_default_dtype(torch.float64)

OPTIMIZERS = ["lm", "gn", "lbfgs"]

ODL_LINEAR_MAX_ITER = 1000

LBFGS_MAX_ITER = None

FEM_NX = 96
FEM_SPARSE = False

HEADER = ["bc", "scenario", "method", "optimizer", "device", "discretization",
          "lin_w_center", "lin_w_over_h", "lin_iters", "lin_residual", "lin_residual_kind",
          "lin_solve_s", "lin_assemble_s", "lin_total_s",
          "nl_w_center", "nl_w_over_h", "nl_iters", "nl_residual", "nl_residual_kind", "nl_converged",
          "nl_solve_s", "nl_assemble_s", "nl_total_s"]


def _long_path(path):
    if os.name == "nt":
        ap = os.path.abspath(path)
        if not ap.startswith("\\\\?\\"):
            return "\\\\?\\" + ap
    return path


def _timed(fn):
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _save_loss(path, hist, header):
    np.savetxt(_long_path(path), np.asarray(hist, dtype=float).reshape(-1),
               delimiter=",", header=header, comments="")


def _save_field(fields_dir, name, lin_arr, nl_arr):
    try:
        np.savez(_long_path(os.path.join(fields_dir, name + ".npz")),
                 lin=np.asarray(lin_arr, dtype=float),
                 nl=np.asarray(nl_arr, dtype=float))
    except Exception as exc:
        print("[warn] field save failed for %s: %s" % (name, exc))


def main():
    mat = build_goeam_material(P.h, P.num_layers, P.MATERIAL["W_Gr"], P.MATERIAL["H_Gr"],
                               P.MATERIAL["T"], P.MATERIAL["distribution_type"], P.shear_corr)
    D0 = compute_D0_reference(P.h)

    out_dir = os.path.join(HERE, "compare_results")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(_long_path(log_dir), exist_ok=True)
    fields_dir = os.path.join(out_dir, "fields")
    os.makedirs(_long_path(fields_dir), exist_ok=True)

    spectral_disc = "CGL N=%d" % P.N
    fem_disc = "Q4 nx=%d%s" % (FEM_NX, " (sparse)" if FEM_SPARSE else "")

    if P.use_cuda and DEV != "cuda":
        print("[warn] use_cuda=True but CUDA unavailable -> ODL running on CPU (device column = actual)")
    print("=== Q3 focused comparison: ODL(soft, %s) x {LM,GN,LBFGS}  vs  DQ-Newton(cpu)  vs  FEM-Newton(cpu, %s) ===" % (DEV, fem_disc))
    print("GOEAM | a=b=%.2f h=%.3f a/h=%.0f | q=%.3e | spectral %s | nonlinear, init=linear warm-start" %
          (P.a, P.h, P.a / P.h, P.q, spectral_disc))
    print("held identical: problem/material/foundation, bc_weight_soft=%.0e, load_steps=%d, newton_max_iter=%d | LBFGS cap=%s"
          % (P.bc_weight_soft, P.load_steps, P.newton_max_iter, LBFGS_MAX_ITER))

    out = os.path.join(out_dir, "comparison.csv")
    fh = open(_long_path(out), "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(HEADER)
    fh.flush()
    n_rows = 0

    def emit(row):
        nonlocal n_rows
        writer.writerow(row)
        fh.flush()
        n_rows += 1

    try:
        for bc in P.BOUNDARY_CONDITIONS:
            bckw = dict(x_bc_type=bc["x_bc_type"], y_bc_type=bc["y_bc_type"],
                        x_bc_convention=bc["x_bc_convention"], y_bc_convention=bc["y_bc_convention"])
            for sc in P.SCENARIOS:
                cfg = P.make_config(sc, P.N, "nonlinear", D0, bc)
                k_w, k_s = cfg["k_w"], cfg["k_s"]
                t_asm0 = time.perf_counter()
                strong = dq.build_strong_form_plate_system(cfg, mat)
                t_asm_strong = time.perf_counter() - t_asm0
                print("\n-- %s | %s  (k_w=%.3e k_s=%.3e)" % (bc["name"], sc["name"], k_w, k_s))

                dq_l = dq.solve_bending_linear_dq(strong)
                w_dq_l = extract_center_deflection(dq_l["displacement_full"], strong)
                dq_n = dq.solve_bending_nonlinear_dq(strong, load_steps=P.load_steps, init=P.INIT_GUESS,
                                                     init_seed=P.INIT_SEED, init_scale=P.INIT_SCALE,
                                                     max_iter=P.newton_max_iter)
                w_dq = extract_center_deflection(dq_n["displacement_full"], strong)
                emit([bc["name"], sc["name"], "DQ-Newton", "newton", "cpu", spectral_disc,
                      w_dq_l, abs(w_dq_l) / P.h, 1, dq_l["residual_norm"], "L2_strong_abs",
                      dq_l["elapsed_s"], t_asm_strong, t_asm_strong + dq_l["elapsed_s"],
                      w_dq, abs(w_dq) / P.h, dq_n["iterations"], dq_n["residual_l2"], "L2_strong_abs", dq_n["converged"],
                      dq_n["elapsed_s"], t_asm_strong, t_asm_strong + dq_n["elapsed_s"]])
                print("   DQ-Newton     LIN w/h=%.5f t=%.3fs | NL w/h=%.5f it=%2d t=%.3fs"
                      % (abs(w_dq_l) / P.h, t_asm_strong + dq_l["elapsed_s"],
                         abs(w_dq) / P.h, dq_n["iterations"], t_asm_strong + dq_n["elapsed_s"]))
                _save_field(fields_dir, "%s_%s_DQ-Newton" % (bc["name"], sc["name"]),
                            dq_l["displacement_full"], dq_n["displacement_full"])

                fem_l = fem.solve_linear(FEM_NX, FEM_NX, mat, P.a, P.b, k_w, k_s, P.q,
                                         h=P.h, nondim=P.NONDIM, sparse=FEM_SPARSE, **bckw)
                fem_n = fem.solve_nonlinear(FEM_NX, FEM_NX, mat, P.a, P.b, k_w, k_s, P.q,
                                            load_steps=P.load_steps, max_iter=P.newton_max_iter,
                                            h=P.h, nondim=P.NONDIM, sparse=FEM_SPARSE, **bckw)
                emit([bc["name"], sc["name"], "FEM-Newton", "newton", "cpu", fem_disc,
                      fem_l["w_center"], abs(fem_l["w_center"]) / P.h, 1, fem_l["residual"], "rel_weak_Linf",
                      "", "", fem_l["elapsed_s"],
                      fem_n["w_center"], abs(fem_n["w_center"]) / P.h, fem_n["iterations"],
                      fem_n["residual_rel"], "rel_weak_Linf", fem_n["converged"],
                      "", "", fem_n["elapsed_s"]])
                print("   FEM-Newton    LIN w/h=%.5f t=%.3fs | NL w/h=%.5f it=%2d t=%.3fs  (%s)"
                      % (abs(fem_l["w_center"]) / P.h, fem_l["elapsed_s"],
                         abs(fem_n["w_center"]) / P.h, fem_n["iterations"], fem_n["elapsed_s"], fem_disc))
                _save_field(fields_dir, "%s_%s_FEM-Newton" % (bc["name"], sc["name"]),
                            fem_l["d"], fem_n["d"])

                resid_soft, d_lin_soft = dq.build_strong_vk_residual_soft(
                    strong, bc_weight=P.bc_weight_soft, device=DEV, is_nonlinear=True)
                resid_soft_lin, _ = dq.build_strong_vk_residual_soft(
                    strong, bc_weight=P.bc_weight_soft, device=DEV, is_nonlinear=False)
                x0_zero = torch.zeros(int(np.asarray(d_lin_soft).shape[0]), dtype=torch.float64, device=DEV)
                for opt in OPTIMIZERS:
                    mi_lin = (ODL_SOFT_LINEAR_MAX_ITER_LBFGS
                              if opt == "lbfgs" and mg_prolongations_for_lbfgs() is not None
                              else ODL_LINEAR_MAX_ITER)
                    ol, t_ol = _timed(lambda: residual_solve(
                        lambda x: resid_soft_lin(x), x0_zero.clone(),
                        optimizer=opt, max_iter=mi_lin, **lbfgs_mg_kwargs(opt)))
                    w_ol = extract_center_deflection(
                        dq.to_physical(ol["x"].cpu().numpy(), strong), strong)
                    mi = LBFGS_MAX_ITER if opt == "lbfgs" else None
                    s = odl_strong_soft_nonlinear(strong, resid_soft, ol["x"], optimizer=opt, max_iter=mi,
                                                  bc_name=bc["name"])
                    w = s["wN"]
                    emit([bc["name"], sc["name"], "ODL-soft", opt, DEV, spectral_disc,
                          w_ol, abs(w_ol) / P.h, ol.get("n_iter"), ol.get("residual_norm"), "soft_rowscaled",
                          t_ol, "", t_ol,
                          w, abs(w) / P.h, s["nl"]["n_iter"], s["nl"]["residual_norm"], "soft_rowscaled", s["converged"],
                          s["elapsed_s"], "", s["elapsed_s"]])
                    print("   ODL-soft %-5s LIN w/h=%.5f t=%.3fs | NL w/h=%.5f it=%6d t=%.3fs"
                          % (opt, abs(w_ol) / P.h, t_ol, abs(w) / P.h, s["nl"]["n_iter"], s["elapsed_s"]))
                    if "mg_restart_kicks" in s["nl"]:
                        print("                 restart: kicks=%d suppressed=%d (scale=%g, gate=%s, seed=%d)"
                              % (s["nl"]["mg_restart_kicks"], s["nl"]["mg_restart_suppressed"],
                                 s["nl"]["mg_restart_scale"], s["nl"]["mg_restart_gate"],
                                 s["nl"]["mg_restart_seed"]))
                    _save_loss(os.path.join(log_dir, "loss_ODL_soft_%s_%s_%s.csv" % (opt, bc["name"], sc["name"])),
                               s["nl"].get("loss_history", []), "residual_sq_loss")
                    _save_field(fields_dir, "%s_%s_ODL-soft_%s" % (bc["name"], sc["name"], opt),
                                dq.to_physical(ol["x"].cpu().numpy(), strong), s["dN"])
    finally:
        fh.close()

    print("\nwrote %s  (%d rows)" % (out, n_rows))
    print("loss histories in %s" % log_dir)


if __name__ == "__main__":
    main()
