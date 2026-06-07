"""
ODIL sweep runner.

Runs ODIL-DQ on the two foundation scenarios, with two optimizer
variants per scenario:

    - 'levenberg-marquardt' (paper default)
    - 'lbfgs' (gradient-based reference)

For each (scenario, optimizer) it solves both the linear and nonlinear
problems sequentially (linear warmup + nonlinear via the project's
strategy), records max|w|, final PDE residual norm, iteration count,
and wall-clock time, and writes outputs to

    experiments/Comparison_of_computational_time/results/odil/<scenario>/<optimizer>/

Outputs per run:
    - displacement CSV (linear + nonlinear, x, u, w, phi)
    - loss history CSV
    - JSON summary with all metrics

Usage:
    conda activate claude_test
    cd experiments/Comparison_of_computational_time/odil
    python run_odil.py

This script does NOT import the project root params.py.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

# ---------------------------------------------------------------------------
# Path bootstrap: this script lives at <project>/experiments/Comparison_of_computational_time/odil/
# Add the FSDT_bending project root so `modules.*` and `utils.*` resolve.
# Do NOT import the project root `params.py`.
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import params_odil as P  # noqa: E402

from modules.material_properties import compute_material_params_for_solver  # noqa: E402
from modules.residuals import TimoshenkoBeamResiduals  # noqa: E402
from modules.solver_odil import ODILSolver  # noqa: E402
from utils.boundary_conditions import get_boundary_conditions  # noqa: E402
from utils.common import initialize_computing_environment  # noqa: E402
from utils.tensor_ops import create_boundary_mask  # noqa: E402
# Nonlinear solve uses the project's multi-attempt strategy.
from utils.nonlinear_solving import solve_nonlinear_with_strategy  # noqa: E402
# Per-attempt validation of the nonlinear solution.
from utils.solution_validation import validate_nonlinear_solution  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _long_path(path: str) -> str:
    """Return a Windows long-path-safe form of `path`.

    On Windows (os.name == 'nt') the absolute path is returned with the
    extended-length prefix so that open()/os.makedirs() can exceed the
    260-character MAX_PATH limit (needed for the deep results tree). The
    prefix is an OS-level hint only: the file is created at the normal
    location and the file CONTENTS are unaffected. On other platforms the
    path is returned unchanged.
    """
    if os.name == "nt":
        ap = os.path.abspath(path)
        prefix = "\\\\?\\"
        if not ap.startswith(prefix):
            return prefix + ap
        return ap
    return path


def reseed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_config(base_module: Any, scen: Dict[str, Any],
                    opt: Dict[str, Any]) -> SimpleNamespace:
    """Build a per-(scenario, optimizer) config namespace from base module."""
    cfg = SimpleNamespace()
    for name in dir(base_module):
        if name.startswith("_"):
            continue
        try:
            setattr(cfg, name, getattr(base_module, name))
        except (AttributeError, TypeError):
            pass

    # Override foundation
    cfg.k1 = float(scen["k1"])
    cfg.k2 = float(scen["k2"])
    cfg.foundation_params = {"k1": cfg.k1, "k2": cfg.k2}

    # Override optimizer-specific knobs
    cfg.optim_name = opt["name"]
    cfg.max_iter_linear = int(opt["max_iter_linear"])
    cfg.max_iter_nonlinear = int(opt["max_iter_nonlinear"])
    cfg.lr = float(opt["lr"])

    return cfg


def compute_pde_residual_norm(result: Dict[str, Any],
                              material_params: Dict[str, float],
                              q: float, k1: float, k2: float,
                              is_nonlinear: bool) -> float:
    """Euclidean norm of the three PDE residuals on interior nodes only."""
    x, u, w, phi = result["x"], result["u"], result["w"], result["phi"]
    A, B = result["A"], result["B"]

    residual_calc = TimoshenkoBeamResiduals(material_params, q, k1, k2)
    R1, R2, R3 = residual_calc.compute(
        u, w, phi, A, B, is_nonlinear
    )
    N_pts = u.shape[0]
    mask = create_boundary_mask(N_pts, x.device, x.dtype)
    sq = (R1 * mask).pow(2).sum() + (R2 * mask).pow(2).sum() + (R3 * mask).pow(2).sum()
    return float(torch.sqrt(sq).item())


def _to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else t


def _build_metrics(result: Dict[str, Any], cfg: SimpleNamespace,
                   material_params: Dict[str, float],
                   is_nonlinear: bool, elapsed: float) -> Dict[str, Any]:
    """Convert a solver/strategy tensor result into the run-record dict.

    Used for both linear (direct solver.solve) and nonlinear
    (solve_nonlinear_with_strategy) paths so the downstream CSV/JSON
    writers see the same flat schema.
    """
    w = result["w"]
    if torch.isfinite(w).all():
        w_max = float(torch.max(torch.abs(w)).item())
    else:
        w_max = float("nan")

    try:
        r_pde = compute_pde_residual_norm(
            result, material_params, cfg.q, cfg.k1, cfg.k2,
            is_nonlinear=is_nonlinear,
        )
    except Exception:
        r_pde = float("nan")

    iterations = len(result.get("loss_history") or [])

    return {
        "x":             _to_np(result["x"]),
        "u":             _to_np(result["u"]),
        "w":             _to_np(result["w"]),
        "phi":           _to_np(result["phi"]),
        "w_max":         w_max,
        "R_PDE_norm":    r_pde,
        "iterations":    iterations,
        "final_loss":    result.get("final_loss"),
        "loss_history":  list(result.get("loss_history") or []),
        "elapsed_s":     elapsed,
    }


def _solve_linear(cfg: SimpleNamespace, solver: ODILSolver,
                  bcs, material_params: Dict[str, float]) -> Dict[str, Any]:
    """Linear pass: direct call to solver.solve."""
    reseed(cfg.seed)
    t0 = time.time()
    result = solver.solve(
        bcs=bcs,
        q=cfg.q,
        is_nonlinear=False,
        optim_name=cfg.optim_name,
        max_iter=cfg.max_iter_linear,
        lr=cfg.lr,
        loss_weights=cfg.pde_weights,
        reg=cfg.reg_weight,
        verbose=False,
        bc_weight=cfg.bc_weight,
        print_every=cfg.print_every,
        gpu_monitor_interval=cfg.gpu_monitor_interval,
    )
    elapsed = time.time() - t0
    metrics = _build_metrics(result, cfg, material_params,
                             is_nonlinear=False, elapsed=elapsed)
    metrics["_tensor_result"] = result  # kept for the strategic nonlinear pass
    return metrics


def _solve_nonlinear_strategic(cfg: SimpleNamespace, solver: ODILSolver,
                               bcs, material_params: Dict[str, float],
                               linear_tensor_result: Dict[str, Any]) -> Dict[str, Any]:
    """Nonlinear pass: invoke main project's solve_nonlinear_with_strategy.

    This is what main.py uses when mode='both'. It honours
    `cfg.use_linear_as_initial`, `cfg.num_solution_attempts`,
    `cfg.initial_value_scale`, `cfg.initial_value_mix_ratio`,
    `cfg.validate_physical_solution`.
    """
    reseed(cfg.seed)

    # Per-attempt probe: record each attempt's metrics.
    # Wrap solver.solve to record EACH of the `num_solution_attempts` attempts
    # (iterations, wall time, final loss, max|w|, physical validity). The probe
    # only times the call and reads the returned dict; it changes no argument,
    # RNG state, or tensor, so the probe does not change the selected solution.
    attempts_log: List[Dict[str, Any]] = []
    _orig_solve = solver.solve

    def _probe_solve(*args, **kwargs):
        a0 = time.time()
        res = _orig_solve(*args, **kwargs)
        a_elapsed = time.time() - a0
        if kwargs.get("is_nonlinear", False):
            lh = res.get("loss_history") or []
            w = res.get("w")
            try:
                w_max = (float(torch.max(torch.abs(w)).item())
                         if torch.isfinite(w).all() else float("nan"))
            except Exception:
                w_max = float("nan")
            try:
                is_valid, _ = validate_nonlinear_solution(linear_tensor_result, res)
                is_valid = bool(is_valid)
            except Exception:
                is_valid = None
            rec = {
                "attempt":        len(attempts_log) + 1,
                "iterations":     len(lh),
                "elapsed_s":      a_elapsed,
                "final_loss":     res.get("final_loss"),
                "w_max":          w_max,
                "physical_valid": is_valid,
                "_obj_id":        id(res),
            }
            attempts_log.append(rec)
            fl = rec["final_loss"]
            fl_str = f"{fl:.3e}" if isinstance(fl, (int, float)) else "n/a"
            print(f"    [attempt {rec['attempt']}/{cfg.num_solution_attempts}] "
                  f"iters={rec['iterations']:>6}  "
                  f"time={rec['elapsed_s']:9.2f} s  "
                  f"final_loss={fl_str}  "
                  f"w_max={rec['w_max']:.6f}  "
                  f"valid={rec['physical_valid']}")
        return res

    solver.solve = _probe_solve
    try:
        t0 = time.time()
        result = solve_nonlinear_with_strategy(
            solver=solver,
            bcs=bcs,
            q=cfg.q,
            linear_result=linear_tensor_result,
            params=cfg,
            verbose=False,
            verbose_attempts=False,
        )
        elapsed = time.time() - t0
    finally:
        solver.solve = _orig_solve  # always restore the original bound method

    # Tag which recorded attempt is the one select_best_solution returned.
    for rec in attempts_log:
        rec["selected"] = (rec.pop("_obj_id") == id(result))

    metrics = _build_metrics(result, cfg, material_params,
                             is_nonlinear=True, elapsed=elapsed)
    metrics["_attempts_log"] = attempts_log
    return metrics


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_displacement_csv(linear: Dict, nonlinear: Dict, path: str) -> None:
    with open(_long_path(path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "linear_u", "linear_w", "linear_phi",
                    "nonlinear_u", "nonlinear_w", "nonlinear_phi"])
        N = len(linear["x"])
        for i in range(N):
            w.writerow([
                f"{linear['x'][i]:.18e}",
                f"{linear['u'][i]:.18e}",
                f"{linear['w'][i]:.18e}",
                f"{linear['phi'][i]:.18e}",
                f"{nonlinear['u'][i]:.18e}",
                f"{nonlinear['w'][i]:.18e}",
                f"{nonlinear['phi'][i]:.18e}",
            ])


def _write_loss_csv(linear_hist: List[float], nonlinear_hist: List[float],
                    path: str) -> None:
    n = max(len(linear_hist), len(nonlinear_hist))
    with open(_long_path(path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "linear_loss", "nonlinear_loss"])
        for i in range(n):
            ln = linear_hist[i] if i < len(linear_hist) else ""
            nl = nonlinear_hist[i] if i < len(nonlinear_hist) else ""
            w.writerow([i, ln, nl])


def _write_attempts_csv(attempts: List[Dict[str, Any]], path: str) -> None:
    """test_LBFGS: persist per-attempt detail of the strategic nonlinear solve."""
    with open(_long_path(path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attempt", "iterations", "elapsed_s",
                    "final_loss", "w_max", "physical_valid", "selected"])
        for a in attempts:
            w.writerow([
                a.get("attempt"),
                a.get("iterations"),
                f"{a.get('elapsed_s', float('nan')):.6f}",
                a.get("final_loss"),
                f"{a.get('w_max', float('nan')):.18e}",
                a.get("physical_valid"),
                a.get("selected"),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one_combination(scen: Dict[str, Any], opt: Dict[str, Any],
                        device, material_params,
                        results_root: str) -> Dict[str, Any]:
    cfg = make_run_config(P, scen, opt)
    bcs = get_boundary_conditions(bc_type=cfg.bc_type, device=device)

    out_dir = os.path.join(results_root, scen["name"], opt["name"])
    log_dir = os.path.join(out_dir, "logs")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(_long_path(log_dir), exist_ok=True)
    os.makedirs(_long_path(data_dir), exist_ok=True)

    print()
    print("#" * 72)
    print(f"# ODIL  scenario={scen['name']}  optimizer={opt['name']}")
    print(f"#       k1={scen['k1']}, k2={scen['k2']}")
    print(f"#       max_iter_linear={cfg.max_iter_linear}, "
          f"max_iter_nonlinear={cfg.max_iter_nonlinear}")
    print(f"#       use_linear_as_initial={cfg.use_linear_as_initial}, "
          f"num_solution_attempts={cfg.num_solution_attempts}")
    print("#" * 72)

    # Build ONE solver instance and reuse for both linear and nonlinear, so
    # the nonlinear call can leverage the cached DQ matrices, the same logger
    # and the same condition-number checks.
    solver = ODILSolver(
        N=cfg.N,
        material_params=material_params,
        device=device,
        dq_method=cfg.dq_method,
        bc_type=cfg.bc_type,
        log_dir=log_dir,
        config=cfg,
    )

    # ----- Linear -----
    print("[linear] solving...")
    linear = _solve_linear(cfg, solver, bcs, material_params)
    print(f"[linear] w_max = {linear['w_max']:.6e}, "
          f"||R_PDE|| = {linear['R_PDE_norm']:.3e}, "
          f"iters = {linear['iterations']}, "
          f"time = {linear['elapsed_s']:.2f} s")

    # ----- Nonlinear (multi-attempt strategy from main project) -----
    # This invokes utils.nonlinear_solving.solve_nonlinear_with_strategy,
    # the same routine that main.py uses when mode='both'. It performs up to
    # `cfg.num_solution_attempts` attempts with linear-warm-start +
    # randomised perturbation, and selects the best-physical solution
    # automatically. Replaces the previous single hand-written warm-start.
    print("[nonlinear] solving (strategic multi-attempt)...")
    nonlinear = _solve_nonlinear_strategic(
        cfg, solver, bcs, material_params,
        linear_tensor_result=linear["_tensor_result"],
    )
    # `_tensor_result` was internal — drop it before downstream JSON serialise.
    linear.pop("_tensor_result", None)
    print(f"[nonlinear] w_max = {nonlinear['w_max']:.6e}, "
          f"||R_PDE|| = {nonlinear['R_PDE_norm']:.3e}, "
          f"iters = {nonlinear['iterations']}, "
          f"time = {nonlinear['elapsed_s']:.2f} s")

    # test_LBFGS: detach the per-attempt probe log from the nonlinear metrics.
    attempts_log = nonlinear.pop("_attempts_log", [])

    # Persist artifacts
    disp_csv = os.path.join(data_dir, "displacement.csv")
    loss_csv = os.path.join(data_dir, "loss_history.csv")
    attempts_csv = os.path.join(data_dir, "nonlinear_attempts.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    _write_displacement_csv(linear, nonlinear, disp_csv)
    _write_loss_csv(linear["loss_history"], nonlinear["loss_history"], loss_csv)
    if attempts_log:
        _write_attempts_csv(attempts_log, attempts_csv)

    summary = {
        "scenario":    scen["name"],
        "k1":          scen["k1"],
        "k2":          scen["k2"],
        "optimizer":   opt["name"],
        "lr":          opt["lr"],
        "max_iter_linear":    opt["max_iter_linear"],
        "max_iter_nonlinear": opt["max_iter_nonlinear"],
        "linear":     {k: v for k, v in linear.items()
                       if k not in ("x", "u", "w", "phi", "loss_history")},
        "nonlinear":  {k: v for k, v in nonlinear.items()
                       if k not in ("x", "u", "w", "phi", "loss_history")},
        "nonlinear_attempts": attempts_log,
        "displacement_csv": disp_csv,
        "loss_history_csv": loss_csv,
        "nonlinear_attempts_csv": attempts_csv if attempts_log else None,
    }
    with open(_long_path(summary_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _parse_cli():
    """Optional CLI: --only-optimizer NAME runs just that optimizer and
    merges its results into an existing sweep_index.json (other optimizer
    runs are preserved). NAME must match an entry in P.optimizers.
    """
    import argparse
    valid_names = [o["name"] for o in P.optimizers]
    p = argparse.ArgumentParser(description="ODIL sweep")
    p.add_argument(
        "--only-optimizer", default=None,
        choices=valid_names,
        help="If set, only run this optimizer (preserve previous results "
             "for the others by merging the sweep index).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_cli()

    device = initialize_computing_environment(
        seed=P.seed,
        use_cuda=P.use_cuda,
        dtype=torch.float64 if P.dtype_str == "float64" else torch.float32,
        verbose=False,
    )

    material_params = compute_material_params_for_solver(
        h=P.h, L=P.L, num_layers=P.num_layers,
        W_Gr=P.W_Gr, H_Gr=P.H_Gr, T=P.T,
        distribution_type=P.distr_type, q=P.q,
    )
    material_params.pop("q", None)

    results_root = os.path.abspath(os.path.join(THIS_DIR, "..", "results", "odil"))
    os.makedirs(_long_path(results_root), exist_ok=True)

    # Apply --only-optimizer filter
    optimizers_to_run = P.optimizers
    if args.only_optimizer is not None:
        optimizers_to_run = [o for o in P.optimizers
                             if o["name"] == args.only_optimizer]

    print("=" * 72)
    print("ODIL sweep (DQ N=13, C-C, X-distribution, q=-0.08)")
    if args.only_optimizer is not None:
        print(f"FILTER: --only-optimizer={args.only_optimizer}")
    print("=" * 72)
    print(f"  scenarios = {[s['name'] for s in P.scenarios]}")
    print(f"  optimizers = {[o['name'] for o in optimizers_to_run]}")
    print(f"  results -> {results_root}")
    print(f"  material: a11={material_params['a11']:.4f}, "
          f"b11={material_params['b11']:.4e}, "
          f"d11={material_params['d11']:.4f}, "
          f"a55={material_params['a55']:.4f}, "
          f"lambda={material_params['lambda_val']:.1f}")
    print("=" * 72)

    all_summaries = []
    total_t0 = time.time()
    for scen in P.scenarios:
        for opt in optimizers_to_run:
            try:
                s = run_one_combination(scen, opt, device,
                                        material_params, results_root)
                all_summaries.append(s)
            except Exception as exc:
                print(f"[ERROR] scenario={scen['name']} optimizer={opt['name']}: {exc!r}")

    total_elapsed = time.time() - total_t0

    sweep_index = os.path.join(results_root, "sweep_index.json")

    # Merge with existing sweep_index when filter is active (preserve runs
    # of the other optimizers that were not re-executed).
    if args.only_optimizer is not None and os.path.exists(sweep_index):
        try:
            with open(_long_path(sweep_index), "r", encoding="utf-8") as f:
                existing = json.load(f)
            old_runs = existing.get("runs", []) or []
            kept = [r for r in old_runs
                    if r.get("optimizer") != args.only_optimizer]
            payload = {
                "total_elapsed_s": (existing.get("total_elapsed_s", 0.0) or 0.0)
                                   + total_elapsed,
                "runs": kept + all_summaries,
            }
            print(f"[INDEX] Merged: kept {len(kept)} old run(s) for the other "
                  f"optimizer(s) + {len(all_summaries)} new run(s).")
        except Exception as exc:
            print(f"[WARN] Could not merge sweep_index.json ({exc!r}); writing fresh.")
            payload = {"total_elapsed_s": total_elapsed, "runs": all_summaries}
    else:
        payload = {"total_elapsed_s": total_elapsed, "runs": all_summaries}

    with open(_long_path(sweep_index), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f"ODIL sweep finished in {total_elapsed:.1f} s")
    print(f"Index: {sweep_index}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
