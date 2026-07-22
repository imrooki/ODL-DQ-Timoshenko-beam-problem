



import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, CB)
sys.path.insert(0, os.path.join(CB, "solver", "dq"))
sys.path.insert(0, os.path.join(CB, "material"))

from modules.odl_config import DEFAULT_DEVICE, DEFAULT_OPTIMIZER
from modules.material_plate import build_goeam_material, compute_D0_reference
from modules.plate_weak import build_weak_form_plate_system
from modules.plate_nonlinear import build_weak_vk_residual
from modules.odl_solver import odl_energy_solve, residual_solve
from modules.postprocess import extract_center_deflection
import direct_dq_plate as dq
import params_q3 as P

torch.set_default_dtype(torch.float64)
DEV = P.resolve_device()
GN_VARIANT = getattr(P, "GN_VARIANT", "original")
LBFGS_PRECONDITION = getattr(P, "LBFGS_PRECONDITION", True)
ODL_SOFT_LINEAR_MAX_ITER = 1000

LBFGS_MULTIGRID = bool(getattr(P, "LBFGS_MULTIGRID", False))
LBFGS_MG_LEVELS = tuple(getattr(P, "LBFGS_MG_LEVELS", (2, 3, 5, 8, 13)))
LBFGS_MG_STAGE_FRACTIONS = tuple(getattr(P, "LBFGS_MG_STAGE_FRACTIONS", (0.15, 0.25, 0.60)))
LBFGS_MG_TOL = float(getattr(P, "LBFGS_MG_TOL", 1.0e-5))
ODL_SOFT_LINEAR_MAX_ITER_LBFGS = 20000

LBFGS_MG_RESTART = bool(getattr(P, "LBFGS_MG_RESTART", False))
LBFGS_MG_RESTART_SCALE = float(getattr(P, "LBFGS_MG_RESTART_SCALE", 1.0e-5))
LBFGS_MG_RESTART_GATE = getattr(P, "LBFGS_MG_RESTART_GATE", 1.0e-2)
LBFGS_MG_RESTART_PATIENCE = int(getattr(P, "LBFGS_MG_RESTART_PATIENCE", 600))
LBFGS_MG_RESTART_THRESHOLD = float(getattr(P, "LBFGS_MG_RESTART_THRESHOLD", 5.0e-4))
LBFGS_MG_RESTART_SEEDS = dict(getattr(P, "LBFGS_MG_RESTART_SEEDS", {}))

_MG_PROLONGATIONS = None


def mg_prolongations_for_lbfgs():
    global _MG_PROLONGATIONS
    if not LBFGS_MULTIGRID:
        return None
    if _MG_PROLONGATIONS is None:
        from modules.odl_solver import build_mg_prolongations
        _MG_PROLONGATIONS = build_mg_prolongations(P.N, LBFGS_MG_LEVELS)
    return _MG_PROLONGATIONS


def lbfgs_mg_kwargs(optimizer):
    if str(optimizer).lower() not in ("lbfgs", "l-bfgs"):
        return {}
    pro = mg_prolongations_for_lbfgs()
    if pro is None:
        return {}
    return {"lbfgs_mg_prolongations": pro,
            "lbfgs_mg_stage_fractions": LBFGS_MG_STAGE_FRACTIONS,
            "lbfgs_mg_tol": LBFGS_MG_TOL}


def lbfgs_mg_restart_kwargs(optimizer, bc_name):
    if str(optimizer).lower() not in ("lbfgs", "l-bfgs"):
        return {}
    if bc_name is None or not LBFGS_MG_RESTART or mg_prolongations_for_lbfgs() is None:
        return {}
    seed = LBFGS_MG_RESTART_SEEDS.get(str(bc_name))
    if seed is None:
        raise ValueError("params_q3.LBFGS_MG_RESTART_SEEDS 缺少边界条件 %r 的种子" % (bc_name,))
    return {"lbfgs_mg_restart_scale": LBFGS_MG_RESTART_SCALE,
            "lbfgs_mg_restart_gate": LBFGS_MG_RESTART_GATE,
            "lbfgs_mg_restart_patience": LBFGS_MG_RESTART_PATIENCE,
            "lbfgs_mg_restart_threshold": LBFGS_MG_RESTART_THRESHOLD,
            "lbfgs_mg_restart_seed": int(seed)}


def _timed(fn):
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _long_path(path):
    if os.name == "nt":
        ap = os.path.abspath(path)
        if not ap.startswith("\\\\?\\"):
            return "\\\\?\\" + ap
    return path


def _save_loss(path, hist, header):
    np.savetxt(_long_path(path), np.asarray(hist, dtype=float).reshape(-1),
               delimiter=",", header=header, comments="")


def build_material():
    m = P.MATERIAL
    if m["kind"] == "goeam":
        return build_goeam_material(P.h, P.num_layers, m["W_Gr"], m["H_Gr"], m["T"],
                                    m["distribution_type"], P.shear_corr)
    raise ValueError(f"unsupported material kind: {m['kind']}")


def _random_like(ref_vec_np, seed, scale):
    rng = np.random.default_rng(int(seed))
    ref = float(np.max(np.abs(ref_vec_np))) or 1.0
    return scale * ref * (2.0 * rng.random(len(ref_vec_np)) - 1.0)


def odl_energy_linear(weak):
    rl = odl_energy_solve(weak, verbose=False, device=DEV)
    wL = extract_center_deflection(dq.to_physical(rl["displacement_full"], weak), weak)
    return {"lin": rl, "wL": wL}


def odl_energy_nonlinear(weak, x0):
    resid = build_weak_vk_residual(weak, device=DEV)
    on = residual_solve(lambda x: resid(x, 1.0), x0.clone(), optimizer=DEFAULT_OPTIMIZER, max_iter=300)
    dN = dq.to_physical(weak["transform"] @ on["x"].cpu().numpy(), weak)
    wN = extract_center_deflection(dN, weak)
    return {"nl": on, "wN": wN, "dN": dN}


def odl_strong_linear(strong):
    n = strong["transform"].shape[1]
    KL_t = torch.as_tensor(strong["KL_red"], device=DEV)
    f_t = torch.as_tensor(strong["f_red"], device=DEV)
    ol = residual_solve(lambda y: KL_t @ y - f_t, torch.zeros(n, dtype=torch.float64, device=DEV),
                        optimizer="gn", max_iter=100, gn_variant=GN_VARIANT)
    dL = dq.to_physical(strong["transform"] @ ol["x"].cpu().numpy(), strong)
    wL = extract_center_deflection(dL, strong)
    return {"lin": ol, "wL": wL, "dL": dL}


def odl_strong_nonlinear(strong, x0):
    resid = dq.build_strong_vk_residual(strong, device=DEV)
    on = residual_solve(lambda x: resid(x, 1.0), x0.clone(), optimizer="gn", max_iter=200, gn_variant=GN_VARIANT)
    dN = dq.to_physical(strong["transform"] @ on["x"].cpu().numpy(), strong)
    wN = extract_center_deflection(dN, strong)
    return {"nl": on, "wN": wN, "dN": dN}


def odl_strong_soft_nonlinear(strong, resid_soft, x0, optimizer="gn", max_iter=None, bc_name=None):
    mi = max_iter if max_iter is not None else ({"lbfgs": 50000, "lm": 300, "gn": 200}.get(optimizer, 200))
    t0 = time.time()
    on = residual_solve(lambda x: resid_soft(x), x0.clone(), optimizer=optimizer, max_iter=mi,
                        gn_variant=GN_VARIANT, lbfgs_precondition=LBFGS_PRECONDITION,
                        **lbfgs_mg_kwargs(optimizer),
                        **lbfgs_mg_restart_kwargs(optimizer, bc_name))
    el = time.time() - t0
    dN = dq.to_physical(on["x"].cpu().numpy(), strong)
    converged = bool(on["residual_norm"] < 1.0e-3)
    return {"wN": extract_center_deflection(dN, strong), "nl": on, "dN": dN,
            "elapsed_s": el, "optimizer": optimizer, "converged": converged}


def main():
    mat = build_material()
    D0 = compute_D0_reference(P.h)
    res_root = os.path.join(CB, "results", "odl")
    os.makedirs(_long_path(res_root), exist_ok=True)
    print("=== Q3 ODL (proposed method) plate baseline -- STRONG form + ENERGY form ===")
    print("material: GOEAM W_Gr=%.3f H_Gr=%.2f T=%.0f dist=%s | N=%d | q=%.3e | nonlinear opt: strong=GN, energy=%s | device=%s"
          % (P.MATERIAL["W_Gr"], P.MATERIAL["H_Gr"], P.MATERIAL["T"], P.MATERIAL["distribution_type"],
             P.N, P.q, DEFAULT_OPTIMIZER, DEV))
    print("nonlinear init strategies (per form): %s  (random reuses INIT_SEED=%s, INIT_SCALE=%s)"
          % (P.INIT_STRATEGIES, P.INIT_SEED, P.INIT_SCALE))

    rows = []
    for bc in P.BOUNDARY_CONDITIONS:
        for sc in P.SCENARIOS:
            cfg = P.make_config(sc, P.N, "nonlinear", D0, bc)
            k_w, k_s = cfg["k_w"], cfg["k_s"]
            sc_dir = os.path.join(res_root, bc["name"], sc["name"])
            os.makedirs(_long_path(sc_dir), exist_ok=True)
            print("\n-- %s | %s (k_w=%.3e k_s=%.3e)" % (bc["name"], sc["name"], k_w, k_s))

            weak = build_weak_form_plate_system(cfg, mat)
            strong = dq.build_strong_form_plate_system(cfg, mat)
            st_lin, t_st_lin = _timed(lambda: odl_strong_linear(strong))
            en_lin, t_en_lin = _timed(lambda: odl_energy_linear(weak))
            resid_soft, d_lin_soft = dq.build_strong_vk_residual_soft(
                strong, bc_weight=P.bc_weight_soft, device=DEV)
            resid_soft_lin, _ = dq.build_strong_vk_residual_soft(
                strong, bc_weight=P.bc_weight_soft, device=DEV, is_nonlinear=False)
            ndof_soft = int(np.asarray(d_lin_soft).shape[0])
            print("   LINEAR  STRONG w/h=%.5f (%d it)  ENERGY w/h=%.5f (%d it)"
                  % (abs(st_lin["wL"]) / P.h, st_lin["lin"]["n_iter"],
                     abs(en_lin["wL"]) / P.h, en_lin["lin"]["n_iter"]))

            rows.append([bc["name"], sc["name"], k_w, k_s, "strong", "gn", "linear", st_lin["wL"], abs(st_lin["wL"]) / P.h, st_lin["lin"]["n_iter"], st_lin["lin"]["residual_norm"], True,
                         t_st_lin, "-"])
            rows.append([bc["name"], sc["name"], k_w, k_s, "energy", "lbfgs", "linear", en_lin["wL"], abs(en_lin["wL"]) / P.h, en_lin["lin"]["n_iter"], en_lin["lin"]["rel_residual"], True,
                         t_en_lin, "-"])
            _save_loss(os.path.join(sc_dir, "loss_strong_linear.csv"),
                       st_lin["lin"].get("loss_history", []), "residual_sq_loss")
            _save_loss(os.path.join(sc_dir, "loss_energy_linear.csv"),
                       en_lin["lin"].get("loss_history", []), "energy_loss")

            model_payload = {"bc": bc["name"], "scenario": sc["name"],
                             "strong_linear": st_lin["dL"],
                             "w_center": {"strong_lin": st_lin["wL"], "energy_lin": en_lin["wL"]}}

            for strat in P.INIT_STRATEGIES:
                if strat == "linear":
                    x0_strong = st_lin["lin"]["x"].clone()
                    x0_energy = torch.as_tensor(en_lin["lin"]["displacement_red"], dtype=torch.float64, device=DEV)
                    x0_soft_by_opt = {
                        opt: residual_solve(lambda x: resid_soft_lin(x),
                                            torch.zeros(ndof_soft, dtype=torch.float64, device=DEV),
                                            optimizer=opt,
                                            max_iter=(ODL_SOFT_LINEAR_MAX_ITER_LBFGS
                                                      if opt == "lbfgs" and mg_prolongations_for_lbfgs() is not None
                                                      else ODL_SOFT_LINEAR_MAX_ITER),
                                            gn_variant=GN_VARIANT, lbfgs_precondition=LBFGS_PRECONDITION,
                                            **lbfgs_mg_kwargs(opt))["x"]
                        for opt in ("lm", "gn", "lbfgs")}
                else:
                    x0_strong = torch.as_tensor(_random_like(st_lin["lin"]["x"].cpu().numpy(), P.INIT_SEED, P.INIT_SCALE), dtype=torch.float64, device=DEV)
                    x0_energy = torch.as_tensor(_random_like(np.asarray(en_lin["lin"]["displacement_red"]), P.INIT_SEED, P.INIT_SCALE), dtype=torch.float64, device=DEV)
                    _x0_soft_rand = torch.as_tensor(_random_like(np.asarray(d_lin_soft), P.INIT_SEED, P.INIT_SCALE), dtype=torch.float64, device=DEV)
                    x0_soft_by_opt = {opt: _x0_soft_rand for opt in ("lm", "gn", "lbfgs")}

                st_nl, t_st_nl = _timed(lambda: odl_strong_nonlinear(strong, x0_strong))
                soft = {opt: odl_strong_soft_nonlinear(strong, resid_soft, x0_soft_by_opt[opt], optimizer=opt,
                                                       bc_name=bc["name"])
                        for opt in ("lm", "gn", "lbfgs")}
                en_nl, t_en_nl = _timed(lambda: odl_energy_nonlinear(weak, x0_energy))

                print("   [init=%-6s] STRONG NL w/h=%.5f (%d it, |R|=%.1e) | ENERGY NL w/h=%.5f (%d it)"
                      % (strat, abs(st_nl["wN"]) / P.h, st_nl["nl"]["n_iter"], st_nl["nl"]["residual_norm"],
                         abs(en_nl["wN"]) / P.h, en_nl["nl"]["n_iter"]))
                for opt in ("lm", "gn", "lbfgs"):
                    s = soft[opt]
                    print("   [init=%-6s] SOFT %-5s NL w/h=%.5f (%6d it, |R|=%.1e, t=%.1fs, conv=%s)"
                          % (strat, opt, abs(s["wN"]) / P.h, s["nl"]["n_iter"], s["nl"]["residual_norm"],
                             s["elapsed_s"], s["converged"]))

                rows.append([bc["name"], sc["name"], k_w, k_s, "strong", "gn", "nonlinear", st_nl["wN"], abs(st_nl["wN"]) / P.h, st_nl["nl"]["n_iter"], st_nl["nl"]["residual_norm"], True,
                             t_st_nl, strat])
                for opt in ("lm", "gn", "lbfgs"):
                    s = soft[opt]
                    rows.append([bc["name"], sc["name"], k_w, k_s, "strong_soft", opt, "nonlinear", s["wN"], abs(s["wN"]) / P.h, s["nl"]["n_iter"], s["nl"]["residual_norm"], s["converged"],
                                 s["elapsed_s"], strat])
                rows.append([bc["name"], sc["name"], k_w, k_s, "energy", DEFAULT_OPTIMIZER, "nonlinear", en_nl["wN"], abs(en_nl["wN"]) / P.h, en_nl["nl"]["n_iter"], en_nl["nl"]["residual_norm"], True,
                             t_en_nl, strat])

                _save_loss(os.path.join(sc_dir, "loss_strong_nonlinear_%s.csv" % strat),
                           st_nl["nl"].get("loss_history", []), "residual_sq_loss")
                for opt in ("lm", "gn", "lbfgs"):
                    _save_loss(os.path.join(sc_dir, "loss_soft_%s_nonlinear_%s.csv" % (opt, strat)),
                               soft[opt]["nl"].get("loss_history", []), "residual_sq_loss")
                _save_loss(os.path.join(sc_dir, "loss_energy_nonlinear_%s.csv" % strat),
                           en_nl["nl"].get("loss_history", []), "residual_sq_loss")

                model_payload["strong_nonlinear_%s" % strat] = st_nl["dN"]
                model_payload["strong_soft_nonlinear_gn_%s" % strat] = soft["gn"]["dN"]
                model_payload["energy_nonlinear_%s" % strat] = en_nl["dN"]
                model_payload["w_center"]["strong_nl_%s" % strat] = st_nl["wN"]
                model_payload["w_center"]["energy_nl_%s" % strat] = en_nl["wN"]

            torch.save(model_payload, _long_path(os.path.join(sc_dir, "model_odl.pth")))

    out = os.path.join(res_root, "odl_plate_baseline.csv")
    header = ["bc", "scenario", "k_w", "k_s", "form", "optimizer", "problem",
              "w_center", "w_over_h", "iters", "residual", "converged",
              "solve_s", "init_strategy"]
    with open(_long_path(out), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print("\nwrote %s  (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
