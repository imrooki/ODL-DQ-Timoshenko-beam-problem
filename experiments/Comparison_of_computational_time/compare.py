"""
comparison: PINN vs ODIL on the same Timoshenko beam problem.

Reads outputs from
    experiments/Comparison_of_computational_time/results/pinn/...     (energy-form Pure PINN)
    experiments/Comparison_of_computational_time/results/odil/...     (DQ + LM / LBFGS / Gauss-Newton)

and produces a single comparison table (CSV + Markdown) plus four
overlay plots, one per (scenario, mode) combination.

Run order:
    1) python pinn/run_pinn.py
    2) python odil/run_odil.py
    3) python compare.py

This script does not require any project-root params.py and does not
modify any code or data outside `experiments/Comparison_of_computational_time/`.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
PINN_ROOT = os.path.join(RESULTS_DIR, "pinn")
ODIL_ROOT = os.path.join(RESULTS_DIR, "odil")
OUT_DIR = RESULTS_DIR
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def _long_path(path: str) -> str:
    r"""Return a Windows long-path-safe (\\?\) form of `path`.

    The absolute path is returned with the extended-length prefix so that
    open()/glob/os.path.exists can exceed the 260-character MAX_PATH limit
    (the results tree is deeply nested). The prefix is an OS-level hint only:
    it does not change the file location or contents. On non-Windows platforms
    the path is returned unchanged.
    """
    if os.name != "nt":
        return path
    prefix = "\\\\?\\"
    if path.startswith(prefix):
        return path
    return prefix + os.path.abspath(path)


SCENARIOS = [
    {"name": "with_foundation", "k1": 0.01, "k2": 0.001},
    {"name": "no_foundation",   "k1": 0.0,  "k2": 0.0},
]

# ODIL optimizer variants under results/odil/<scenario>/<optimizer>/.
ODIL_OPTIMIZERS = ["levenberg-marquardt", "lbfgs", "gauss-newton"]

# (script_name under results/pinn/, label, optimizer key)
PINN_VARIANTS = [
    {"script_name": "pure_pinn_adam",  "label": "PINN-Adam (final)",
     "optimizer_key": "adam"},
]


# ---------------------------------------------------------------------------
# Curve identity & styling
# ---------------------------------------------------------------------------
# Per-method plot styling. CURVE_ORDER fixes the draw/legend order; `group`
# selects the loss-trajectory panel row (pinn = top, odil = bottom).
CURVE_ORDER = ["PINN-Adam", "ODIL-LM", "ODIL-LBFGS", "ODIL-GN"]
CURVE_STYLE: Dict[str, Dict[str, Any]] = {
    "PINN-Adam": {
        "group": "pinn", "color": "tab:blue",
        "w_fmt": "-",   "w_kw": {"lw": 2.0},
        "loss_fmt": "-",
        "w_label": "PINN-Adam (energy form, final)",
        "loss_label": "PINN-Adam",
    },
    "ODIL-LM": {
        "group": "odil", "color": "tab:red",
        "w_fmt": "o-",  "w_kw": {"lw": 1.2, "ms": 6, "mfc": "white"},
        "loss_fmt": "-",
        "w_label": "ODIL-DQ + LM (N=13)",
        "loss_label": "ODIL-LM",
    },
    "ODIL-LBFGS": {
        "group": "odil", "color": "tab:green",
        "w_fmt": "s--", "w_kw": {"lw": 1.0, "ms": 5, "mfc": "white"},
        "loss_fmt": "--",
        "w_label": "ODIL-DQ + LBFGS (N=13)",
        "loss_label": "ODIL-LBFGS",
    },
    "ODIL-GN": {
        "group": "odil", "color": "tab:purple",
        "w_fmt": "^-.", "w_kw": {"lw": 1.2, "ms": 5, "mfc": "white"},
        "loss_fmt": "-.",
        "w_label": "ODIL-DQ + Gauss-Newton (N=13)",
        "loss_label": "ODIL-GN",
    },
}

# Pretty short labels for ODIL optimizers (used in the wall-time summary).
ODIL_LABELS = {
    "levenberg-marquardt": "ODIL-LM",
    "lbfgs":               "ODIL-LBFGS",
    "gauss-newton":        "ODIL-GN",
}


def _classify_row(row: Dict[str, Any]) -> Optional[str]:
    """Map a metrics row to a CURVE_STYLE key, or None if unrecognised."""
    method = row.get("method", "") or ""
    opt = (row.get("optimizer") or "").lower()
    if method.startswith("PINN"):
        return "PINN-Adam"
    if not method.startswith("ODIL"):
        return None
    if "levenberg-marquardt" in opt:
        return "ODIL-LM"
    if "gauss-newton" in opt:
        return "ODIL-GN"
    if "lbfgs" in opt:
        return "ODIL-LBFGS"
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_pinn_run(scen: Dict[str, Any],
                      script_name: str) -> Optional[Dict[str, str]]:
    """Locate PINN's result directory and key files for one (scenario, optimizer)."""
    # The upstream OutputManager nests as
    #   pinn/<script_name>/C-C/X/<param-folder>/{data,logs,plots,models}
    pattern = os.path.join(PINN_ROOT, script_name, "C-C", "X", "*")
    candidates = [c for c in glob.glob(pattern) if os.path.isdir(_long_path(c))]
    for cand in candidates:
        index_path = os.path.join(cand, "logs", "index.json")
        if not os.path.exists(_long_path(index_path)):
            continue
        try:
            with open(_long_path(index_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            params = data.get("run", {}).get("params", {})
            if (float(params.get("k1", -1)) == float(scen["k1"])
                    and float(params.get("k2", -1)) == float(scen["k2"])):
                data_dir = os.path.join(cand, "data")
                disp_csvs = glob.glob(_long_path(os.path.join(data_dir, "w_*.csv")))
                loss_csvs = glob.glob(_long_path(os.path.join(cand, "logs", "loss_*.csv")))
                return {
                    "dir": cand,
                    "index_json": index_path,
                    "disp_csv": disp_csvs[0] if disp_csvs else "",
                    "loss_csv": loss_csvs[0] if loss_csvs else "",
                }
        except Exception:
            continue
    return None


def discover_odil_run(scen_name: str, optimizer: str) -> Optional[Dict[str, str]]:
    """Locate ODIL's result directory for one (scenario, optimizer)."""
    base = os.path.join(ODIL_ROOT, scen_name, optimizer)
    summary = os.path.join(base, "summary.json")
    disp_csv = os.path.join(base, "data", "displacement.csv")
    loss_csv = os.path.join(base, "data", "loss_history.csv")
    if os.path.exists(_long_path(summary)) and os.path.exists(_long_path(disp_csv)):
        return {
            "dir": base,
            "summary_json": summary,
            "disp_csv": disp_csv,
            "loss_csv": loss_csv,
        }
    return None


# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------
def load_pinn_metrics(scen: Dict[str, Any],
                      variant: Dict[str, str]) -> Dict[str, Any]:
    found = discover_pinn_run(scen, variant["script_name"])
    if found is None:
        return {"available": False, "scenario": scen["name"],
                "optimizer": variant["optimizer_key"]}

    with open(_long_path(found["index_json"]), "r", encoding="utf-8") as f:
        data = json.load(f)
    run = data.get("run", {})
    linear = run.get("linear", {}) or {}
    nonlinear = run.get("nonlinear", {}) or {}
    params = run.get("params", {}) or {}
    full = params.get("full_params", {}) or {}

    # For LBFGS: report best loss (model deliverable is best). For Adam: report final.
    use_best = (variant["optimizer_key"] == "lbfgs")
    if use_best:
        loss_lin = linear.get("best_loss", linear.get("final_loss"))
        loss_nl = nonlinear.get("best_loss", nonlinear.get("final_loss"))
        iters_lin = linear.get("best_epoch", full.get("epochs"))
        iters_nl = nonlinear.get("best_epoch", full.get("epochs"))
    else:
        loss_lin = linear.get("final_loss")
        loss_nl = nonlinear.get("final_loss")
        iters_lin = full.get("epochs")
        iters_nl = full.get("epochs")

    return {
        "available":    True,
        "scenario":     scen["name"],
        "method":       variant["label"],
        "optimizer":    f"{full.get('optimizer_type', '?')}, "
                        f"lr={full.get('lr', '?')}, "
                        f"epochs={full.get('epochs', '?')}, "
                        f"patience={full.get('patience', 'None')}, "
                        f"restore_best={full.get('restore_best', True)}",
        "k1":           scen["k1"],
        "k2":           scen["k2"],
        "max_w_linear":     linear.get("max_w"),
        "max_w_nonlinear":  nonlinear.get("max_w"),
        "final_loss_linear":     loss_lin,
        "final_loss_nonlinear":  loss_nl,
        "iterations_linear":    iters_lin,
        "iterations_nonlinear": iters_nl,
        "wall_time_s":  _read_pinn_wall_time(scen["name"], variant["optimizer_key"]),
        "disp_csv":     found["disp_csv"],
        "loss_csv":     found["loss_csv"],
    }


def _read_pinn_wall_time(scen_name: str, optimizer: str) -> Optional[float]:
    p = os.path.join(PINN_ROOT, "sweep_index.json")
    if not os.path.exists(_long_path(p)):
        return None
    try:
        with open(_long_path(p), "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("runs", []):
            if r.get("scenario") == scen_name and r.get("optimizer") == optimizer:
                return r.get("elapsed_s")
    except Exception:
        pass
    return None


def load_odil_metrics(scen_name: str, optimizer: str,
                      k1: float, k2: float) -> Dict[str, Any]:
    found = discover_odil_run(scen_name, optimizer)
    if found is None:
        return {"available": False,
                "scenario": scen_name, "optimizer": optimizer}

    with open(_long_path(found["summary_json"]), "r", encoding="utf-8") as f:
        summary = json.load(f)
    lin = summary.get("linear", {}) or {}
    nl = summary.get("nonlinear", {}) or {}
    return {
        "available":    True,
        "scenario":     scen_name,
        "method":       "ODIL-DQ (residual form)",
        "optimizer":    f"{optimizer}, max_iter_nl={summary.get('max_iter_nonlinear')}",
        "k1":           k1,
        "k2":           k2,
        "max_w_linear":     lin.get("w_max"),
        "max_w_nonlinear":  nl.get("w_max"),
        "final_loss_linear":     lin.get("R_PDE_norm"),  # this is the residual norm
        "final_loss_nonlinear":  nl.get("R_PDE_norm"),
        "iterations_linear":    lin.get("iterations"),
        "iterations_nonlinear": nl.get("iterations"),
        "wall_time_s":  (lin.get("elapsed_s") or 0.0) + (nl.get("elapsed_s") or 0.0),
        "disp_csv":     found["disp_csv"],
        "loss_csv":     found["loss_csv"],
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _fmt_num(value, fmt: str = ".6e") -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if x != x:  # NaN
        return "NaN"
    return f"{x:{fmt}}"


def write_csv_table(rows: List[Dict[str, Any]], path: str) -> None:
    fields = [
        "scenario", "k1", "k2", "method", "optimizer",
        "max_w_linear", "max_w_nonlinear",
        "iterations_linear", "iterations_nonlinear",
        "final_loss_linear", "final_loss_nonlinear",
        "wall_time_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md_table(rows: List[Dict[str, Any]], path: str) -> None:
    # Group by scenario, write one sub-table per scenario
    by_scen: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_scen.setdefault(r["scenario"], []).append(r)

    lines: List[str] = []
    lines.append("# Comparison Table — PINN vs ODIL (Timoshenko Beam)\n")
    lines.append("**Common configuration:** C-C, X-distribution, "
                 "h=0.1, L/h=20, W_Gr=0.025, H_Gr=0.8, T=300 K, "
                 "q=-0.08, num_layers=10, seed=42.\n")
    lines.append("**Notes on metrics:**")
    lines.append("- `max|w|` is the maximum absolute transverse deflection.")
    lines.append("- For PINN, `final_loss` is the converged total potential "
                 "energy `Π` (negative under load).")
    lines.append("- For ODIL, `final_loss` is the converged PDE-residual "
                 "L2-norm `||R_PDE||` on interior nodes.")
    lines.append("- The two are NOT directly comparable scalars; treat them "
                 "as method-specific convergence indicators.")
    lines.append("- `wall_time_s` is end-to-end training/solve time including "
                 "both linear and nonlinear sub-problems.\n")

    # Row order within a scenario: PINN first, then ODIL in LM / LBFGS / GN order.
    odil_suborder = {"levenberg-marquardt": 0, "lbfgs": 1, "gauss-newton": 2}

    def _row_sort_key(r):
        m = r.get("method", "")
        o = (r.get("optimizer") or "").lower()
        if m.startswith("PINN"):
            return (0, 0)
        if m.startswith("ODIL"):
            sub = next((v for k, v in odil_suborder.items() if k in o), 9)
            return (1, sub)
        return (99, 99)

    for scen_name in ["with_foundation", "no_foundation"]:
        scen_rows = sorted(by_scen.get(scen_name, []), key=_row_sort_key)
        if not scen_rows:
            continue
        k1 = scen_rows[0].get("k1", "?")
        k2 = scen_rows[0].get("k2", "?")
        lines.append(f"## Scenario `{scen_name}` — k1={k1}, k2={k2}\n")
        lines.append("| Method | Optimizer | max\\|w\\| linear | max\\|w\\| nonlinear "
                     "| iters lin / nl | final loss lin / nl | wall (s) |")
        lines.append("|--------|-----------|------------------|---------------------"
                     "|---------------|--------------------|----------|")

        for r in scen_rows:
            lines.append(
                "| {m} | {o} | {wl} | {wn} | {il} / {inl} | {ll} / {ln_loss} | {t} |".format(
                    m=r["method"],
                    o=r.get("optimizer", "—"),
                    wl=_fmt_num(r.get("max_w_linear")),
                    wn=_fmt_num(r.get("max_w_nonlinear")),
                    il=_fmt_num(r.get("iterations_linear"), ".0f"),
                    inl=_fmt_num(r.get("iterations_nonlinear"), ".0f"),
                    ll=_fmt_num(r.get("final_loss_linear"), ".3e"),
                    ln_loss=_fmt_num(r.get("final_loss_nonlinear"), ".3e"),
                    t=_fmt_num(r.get("wall_time_s"), ".1f"),
                )
            )
        lines.append("")

    # ---------- Total wall-time summary ("computational cost") --------
    lines.append("## Total wall-time summary\n")
    lines.append("PINN's total time is the sum of the linear and nonlinear "
                 "model trainings (the upstream Pure-PINN trains the two "
                 "models independently, with no warm-start, no "
                 "pseudo-supervision, and no transfer learning); each "
                 "training is `epochs` long.\n")
    lines.append("ODIL's total time is the sum of the linear solve and the "
                 "warm-started nonlinear solve.\n")

    odil_cols = [ODIL_LABELS.get(o, f"ODIL-{o}") for o in ODIL_OPTIMIZERS]
    header_cells = (["Scenario", "PINN-Adam (s)"]
                    + [f"{c} (s)" for c in odil_cols]
                    + [f"PINN / {c}" for c in odil_cols])
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["----"] * len(header_cells)) + "|")

    def _ratio(num, den):
        try:
            if num is None or den is None or float(den) == 0.0:
                return "n/a"
            return f"{float(num) / float(den):.1f}x"
        except Exception:
            return "n/a"

    for scen_name in ["with_foundation", "no_foundation"]:
        scen_rows = by_scen.get(scen_name, [])
        if not scen_rows:
            continue
        t_pinn = None
        t_odil: Dict[str, Optional[float]] = {o: None for o in ODIL_OPTIMIZERS}
        for r in scen_rows:
            method = r.get("method", "")
            opt_str = (r.get("optimizer") or "").lower()
            if method.startswith("PINN"):
                t_pinn = r.get("wall_time_s")
            elif method.startswith("ODIL"):
                for o in ODIL_OPTIMIZERS:
                    if o in opt_str:
                        t_odil[o] = r.get("wall_time_s")
                        break

        cells = [scen_name, _fmt_num(t_pinn, ".1f")]
        cells += [_fmt_num(t_odil[o], ".1f") for o in ODIL_OPTIMIZERS]
        cells += [_ratio(t_pinn, t_odil[o]) for o in ODIL_OPTIMIZERS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("---")
    lines.append("Generated by `Response2reviewers/Comparison_of_computational_time/compare.py`.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _load_disp_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(_long_path(path)):
        return None
    try:
        with open(_long_path(path), "r", encoding="utf-8", newline="") as fh:
            return pd.read_csv(fh)
    except Exception:
        return None


def plot_w_overlay(scen_name: str, mode: str,
                   curves: List[Dict[str, str]],
                   out_path: str) -> None:
    """Overlay w(x) for every method present in `curves`.

    `curves` is an ordered list of {"key", "disp_csv", "loss_csv"} dicts; one
    line is drawn per curve whose CSV has an "x" and a f"{mode}_w" column.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    col = f"{mode}_w"

    any_plotted = False
    for c in curves:
        df = _load_disp_csv(c.get("disp_csv", ""))
        if df is None or col not in df.columns or "x" not in df.columns:
            continue
        st = CURVE_STYLE[c["key"]]
        ax.plot(df["x"].to_numpy(), df[col].to_numpy(),
                st["w_fmt"], color=st["color"], label=st["w_label"], **st["w_kw"])
        any_plotted = True

    ax.set_xlabel("x")
    ax.set_ylabel("w(x)")
    ax.set_title(f"Deflection overlay — {mode}, {scen_name}")
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _safe_log_y_series(arr: np.ndarray) -> np.ndarray:
    """Return |arr| with zeros replaced by a tiny positive number (for log y)."""
    arr = np.asarray(arr, dtype=float)
    a = np.abs(arr)
    tiny = max(np.nanmin(a[a > 0]) * 1e-3, 1e-300) if np.any(a > 0) else 1e-300
    return np.where(a > 0, a, tiny)


def plot_loss_trajectory(scen_name: str,
                         curves: List[Dict[str, str]],
                         out_path: str) -> None:
    """Generate a 2x2 panel figure showing convergence trajectories.

        ┌────────────────────────────┬────────────────────────────┐
        │ PINN linear loss vs epoch  │ PINN nonlinear loss vs ep. │
        │   all PINN variants        │   all PINN variants        │
        ├────────────────────────────┼────────────────────────────┤
        │ ODIL linear loss vs iter   │ ODIL nonlinear loss vs it. │
        │   all ODIL optimizers      │   all ODIL optimizers      │
        └────────────────────────────┴────────────────────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Loss trajectories — scenario: {scen_name}", fontsize=12)

    def _load(p):
        if p and os.path.exists(_long_path(p)):
            try:
                with open(_long_path(p), "r", encoding="utf-8", newline="") as fh:
                    return pd.read_csv(fh)
            except Exception:
                return None
        return None

    # Preload each curve's loss CSV once (reused across the two mode columns).
    loss_dfs = {c["key"]: _load(c.get("loss_csv", "")) for c in curves}
    pinn_curves = [c for c in curves if CURVE_STYLE[c["key"]]["group"] == "pinn"]
    odil_curves = [c for c in curves if CURVE_STYLE[c["key"]]["group"] == "odil"]

    # ---- Top row: PINN energy loss vs epoch ----
    for j, mode in enumerate(("linear", "nonlinear")):
        ax = axes[0, j]
        col = f"{mode}_total"
        plotted = False
        for c in pinn_curves:
            df = loss_dfs.get(c["key"])
            if df is None or col not in df.columns:
                continue
            ep = df["epoch"].to_numpy() if "epoch" in df else np.arange(len(df))
            st = CURVE_STYLE[c["key"]]
            ax.plot(ep, df[col].to_numpy(), st["loss_fmt"],
                    color=st["color"], lw=1.3, label=st["loss_label"])
            plotted = True
        if plotted:
            ax.set_xscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("PINN loss (energy)")
            ax.set_title(f"PINN — {mode}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.set_title("PINN loss CSV not found")
            ax.axis("off")

    # ---- Bottom row: ODIL |residual| vs iteration (log-log) ----
    for j, mode in enumerate(("linear", "nonlinear")):
        ax = axes[1, j]
        col = f"{mode}_loss"
        plotted = False
        for c in odil_curves:
            df = loss_dfs.get(c["key"])
            if df is None or col not in df.columns:
                continue
            it = df["iter"].to_numpy() if "iter" in df else np.arange(len(df))
            y = df[col].dropna().to_numpy()
            x = it[:len(y)]
            st = CURVE_STYLE[c["key"]]
            ax.plot(x, _safe_log_y_series(y), st["loss_fmt"],
                    color=st["color"], lw=1.4, label=st["loss_label"])
            plotted = True
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("ODIL loss (|value|, log)")
        ax.set_title(f"ODIL — {mode}")
        ax.grid(True, alpha=0.3, which="both")
        if plotted:
            ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_walltime_bars(rows: List[Dict[str, Any]], out_path: str) -> None:
    """Stacked bar chart of wall time per (method+optimizer, scenario)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # Build label per row
    df["label"] = df.apply(
        lambda r: ("PINN-Adam" if r["method"].startswith("PINN")
                   else f"ODIL-{r['optimizer'].split(',')[0]}"),
        axis=1
    )
    # Pivot: index=label, columns=scenario, values=wall_time_s
    pivot = df.pivot_table(index="label", columns="scenario",
                           values="wall_time_s", aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("wall time (s)")
    ax.set_title("End-to-end wall time per method × scenario")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("comparison: PINN vs ODIL")
    print("=" * 72)

    rows: List[Dict[str, Any]] = []

    for scen in SCENARIOS:
        # PINN rows: one per variant
        for variant in PINN_VARIANTS:
            pinn = load_pinn_metrics(scen, variant)
            if pinn.get("available"):
                rows.append(pinn)
            else:
                print(f"[warn] PINN result not found: scenario='{scen['name']}', "
                      f"variant='{variant['optimizer_key']}'.")

        # ODIL rows: LM, LBFGS, Gauss-Newton
        for opt in ODIL_OPTIMIZERS:
            odil = load_odil_metrics(scen["name"], opt,
                                     k1=scen["k1"], k2=scen["k2"])
            if odil.get("available"):
                rows.append(odil)
            else:
                print(f"[warn] ODIL result not found: scenario='{scen['name']}', "
                      f"optimizer='{opt}'.")

    if not rows:
        print("[error] No comparable runs found. Did you run "
              "`pinn/run_pinn.py` and `odil/run_odil.py` first?")
        return 1

    csv_path = os.path.join(OUT_DIR, "comparison_table.csv")
    md_path = os.path.join(OUT_DIR, "comparison_table.md")
    write_csv_table(rows, csv_path)
    write_md_table(rows, md_path)
    print(f"[ok] Comparison CSV : {csv_path}")
    print(f"[ok] Comparison MD  : {md_path}")

    # Plots — build, per scenario, the ordered list of present method curves.
    for scen in SCENARIOS:
        scen_name = scen["name"]
        by_key: Dict[str, Dict[str, str]] = {}
        for r in rows:
            if r["scenario"] != scen_name:
                continue
            key = _classify_row(r)
            if key is None:
                continue
            by_key[key] = {
                "key": key,
                "disp_csv": r.get("disp_csv", "") or "",
                "loss_csv": r.get("loss_csv", "") or "",
            }
        curves = [by_key[k] for k in CURVE_ORDER if k in by_key]

        # w(x) overlays — every present method in one plot, per mode
        for mode in ("linear", "nonlinear"):
            out_p = os.path.join(PLOTS_DIR,
                                 f"w_overlay_{mode}_{scen_name}.png")
            plot_w_overlay(scen_name, mode, curves, out_p)
            print(f"[ok] Plot saved: {out_p}")

        # Loss trajectories — PINN (top) and ODIL (bottom) panels
        out_loss = os.path.join(PLOTS_DIR,
                                f"loss_trajectory_{scen_name}.png")
        plot_loss_trajectory(scen_name, curves, out_loss)
        print(f"[ok] Plot saved: {out_loss}")

    plot_walltime_bars(rows, os.path.join(PLOTS_DIR, "wall_time_bar.png"))
    print(f"[ok] Wall-time bar  : {os.path.join(PLOTS_DIR, 'wall_time_bar.png')}")

    # Echo total wall-time summary to stdout for quick monitoring
    print("\n" + "-" * 80)
    print("Total wall-time summary:")
    print("-" * 80)
    for scen in SCENARIOS:
        t_pinn = None
        t_odil: Dict[str, Optional[float]] = {o: None for o in ODIL_OPTIMIZERS}
        for r in rows:
            if r["scenario"] != scen["name"]:
                continue
            method = r.get("method", "")
            opt_str = (r.get("optimizer") or "").lower()
            if method.startswith("PINN"):
                t_pinn = r.get("wall_time_s")
            elif method.startswith("ODIL"):
                for o in ODIL_OPTIMIZERS:
                    if o in opt_str:
                        t_odil[o] = r.get("wall_time_s")
                        break
        parts = [f"PINN-Adam={_fmt_num(t_pinn, '.1f')}s"]
        for o in ODIL_OPTIMIZERS:
            parts.append(f"{ODIL_LABELS.get(o, o)}={_fmt_num(t_odil[o], '.1f')}s")
        print(f"  {scen['name']:18s}  " + "  ".join(parts))
    print("-" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
