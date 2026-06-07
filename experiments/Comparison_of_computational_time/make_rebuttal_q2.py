from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

import compare
from odil import params_odil as odil_params


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "rebuttal")
os.makedirs(OUT_DIR, exist_ok=True)


def lp(path: str) -> str:
    """Extended-length path helper for Windows."""
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    abs_path = os.path.abspath(path)
    return "\\\\?\\" + abs_path


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(lp(path))


def count_pinn_params(model_path: str) -> int:
    obj = torch.load(lp(model_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected model object: {type(obj)}")
    return int(sum(v.numel() for v in obj.values()))


@dataclass
class RebuttalRow:
    scenario: str
    method: str
    n_var: str
    iters_nl: int
    wall_s: float
    max_w_nl: float
    rel_err_inf_pct: float
    eps_w_l2_pct: float


def scenario_label(name: str) -> str:
    return "With foundation" if name == "with_foundation" else "No foundation"


def load_comparison_rows() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    return pd.read_csv(path)


def discover_artifact_paths() -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for scen in compare.SCENARIOS:
        scen_name = scen["name"]

        pinn = compare.load_pinn_metrics(scen, compare.PINN_VARIANTS[0])
        if pinn.get("available"):
            out[(scen_name, "PINN-Adam")] = {
                "disp_csv": pinn["disp_csv"],
                "loss_csv": pinn["loss_csv"],
            }

        pinn_bad = compare.load_pinn_metrics(scen, compare.PINN_VARIANTS[1])
        if pinn_bad.get("available"):
            out[(scen_name, "PINN-LBFGS")] = {
                "disp_csv": pinn_bad["disp_csv"],
                "loss_csv": pinn_bad["loss_csv"],
            }

        odil_lm = compare.load_odil_metrics(scen_name, "levenberg-marquardt", scen["k1"], scen["k2"])
        if odil_lm.get("available"):
            out[(scen_name, "ODIL-LM")] = {
                "disp_csv": odil_lm["disp_csv"],
                "loss_csv": odil_lm["loss_csv"],
            }

        odil_lbfgs = compare.load_odil_metrics(scen_name, "lbfgs", scen["k1"], scen["k2"])
        if odil_lbfgs.get("available"):
            out[(scen_name, "ODIL-LBFGS")] = {
                "disp_csv": odil_lbfgs["disp_csv"],
                "loss_csv": odil_lbfgs["loss_csv"],
            }
    return out


def compute_profile_errors(paths: Dict[Tuple[str, str], Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    for scen_name in ["with_foundation", "no_foundation"]:
        ref_df = read_csv(paths[(scen_name, "ODIL-LM")]["disp_csv"])
        ref_x = ref_df["x"].to_numpy(dtype=float)
        ref_w = ref_df["nonlinear_w"].to_numpy(dtype=float)
        ref_inf = np.max(np.abs(ref_w))
        ref_l2 = np.linalg.norm(ref_w)

        for method in ["ODIL-LM", "ODIL-LBFGS", "PINN-Adam", "PINN-LBFGS"]:
            df = read_csv(paths[(scen_name, method)]["disp_csv"])
            x = df["x"].to_numpy(dtype=float)
            w = df["nonlinear_w"].to_numpy(dtype=float)
            if len(x) != len(ref_x) or not np.allclose(x, ref_x):
                w = np.interp(ref_x, x, w)
            err = w - ref_w
            metrics[(scen_name, method)] = {
                "rel_err_inf_pct": 100.0 * np.max(np.abs(err)) / ref_inf,
                "eps_w_l2_pct": 100.0 * np.linalg.norm(err) / ref_l2,
            }
    return metrics


def build_table_rows(comp_df: pd.DataFrame,
                     profile_metrics: Dict[Tuple[str, str], Dict[str, float]],
                     pinn_param_count: int) -> List[RebuttalRow]:
    rows: List[RebuttalRow] = []
    method_map = [
        ("ODIL-LM", ("ODIL-DQ (residual form)", "levenberg-marquardt")),
        ("ODIL-LBFGS", ("ODIL-DQ (residual form)", "lbfgs")),
        ("PINN-Adam", ("PINN-Adam (final)", "Adam")),
    ]

    for scen_name in ["with_foundation", "no_foundation"]:
        for short_method, (method_name, opt_contains) in method_map:
            mask = (comp_df["scenario"] == scen_name) & (comp_df["method"] == method_name)
            mask &= comp_df["optimizer"].str.contains(opt_contains, case=False, regex=False)
            row = comp_df.loc[mask]
            if row.empty:
                raise ValueError(f"Missing row for {scen_name} / {short_method}")
            r = row.iloc[0]
            if short_method.startswith("ODIL"):
                n_var = str(int(odil_params.N * 3))
            else:
                n_var = f"{pinn_param_count:,}"
            rows.append(
                RebuttalRow(
                    scenario=scenario_label(scen_name),
                    method=short_method,
                    n_var=n_var,
                    iters_nl=int(r["iterations_nonlinear"]),
                    wall_s=float(r["wall_time_s"]),
                    max_w_nl=float(r["max_w_nonlinear"]),
                    rel_err_inf_pct=float(profile_metrics[(scen_name, short_method)]["rel_err_inf_pct"]),
                    eps_w_l2_pct=float(profile_metrics[(scen_name, short_method)]["eps_w_l2_pct"]),
                )
            )
    return rows


def write_table_outputs(rows: List[RebuttalRow], profile_metrics: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    csv_path = os.path.join(OUT_DIR, "table_q2.csv")
    md_path = os.path.join(OUT_DIR, "table_q2.md")
    tex_path = os.path.join(OUT_DIR, "table_q2.tex")

    df = pd.DataFrame([{
        "scenario": r.scenario,
        "method": r.method,
        "n_var": r.n_var,
        "iters_nl": r.iters_nl,
        "wall_s": r.wall_s,
        "max_w_nl": r.max_w_nl,
        "rel_err_inf_pct": r.rel_err_inf_pct,
        "eps_w_l2_pct": r.eps_w_l2_pct,
    } for r in rows])
    df.to_csv(csv_path, index=False)

    md_lines = [
        "# Q2 rebuttal table",
        "",
        "Reference in each scenario: ODIL-LM. Error metrics use the nonlinear deflection profile on the 13 stored ODIL nodes.",
        "",
        "| Scenario | Method | n_var† | iters_nl | wall (s) | max|w|_nl | rel.err_∞ (%) | ε_w^L2 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(rows):
        if idx == 3:
            md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        md_lines.append(
            f"| {r.scenario} | {r.method} | {r.n_var} | {r.iters_nl} | {r.wall_s:.1f} | "
            f"{r.max_w_nl:.6f} | {r.rel_err_inf_pct:.6g} | {r.eps_w_l2_pct:.6g} |"
        )
    md_lines += [
        "",
        "† `n_var` is a scale indicator, not a strict apples-to-apples DOF count: ODIL uses 39 discrete field unknowns (N=13 × {u,w,φ}) per solve; PINN uses 41,603 trainable parameters per network, with separate linear and nonlinear networks.",
        "",
        "PINN-LBFGS is intentionally excluded from the rebuttal table because it converged to nonphysical states with max|w| = 10.521280 (with foundation) and 9.569670 (no foundation), while the ODIL-LM physical references are 0.428953 and 0.465201, respectively.",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Scenario & Method & $n_{\\mathrm{var}}^{\\dagger}$ & $\\mathrm{iters}_{nl}$ & wall (s) & $\\max|w|_{nl}$ & rel.err$_\\infty$ (\\%) & $\\varepsilon_w^{L2}$ (\\%)\\\\",
        "\\midrule",
    ]
    for idx, r in enumerate(rows):
        if idx == 3:
            tex_lines.append("\\midrule")
        tex_lines.append(
            f"{r.scenario} & {r.method} & {r.n_var} & {r.iters_nl} & {r.wall_s:.1f} & "
            f"{r.max_w_nl:.6f} & {r.rel_err_inf_pct:.6g} & {r.eps_w_l2_pct:.6g}\\\\"
        )
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")


def make_figure(comp_df: pd.DataFrame,
                paths: Dict[Tuple[str, str], Dict[str, str]],
                profile_metrics: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    png_path = os.path.join(OUT_DIR, "figure_q2.png")
    pdf_path = os.path.join(OUT_DIR, "figure_q2.pdf")

    colors = {
        "PINN-Adam": "#1f77b4",
        "ODIL-LM": "#d62728",
        "ODIL-LBFGS": "#2ca02c",
    }
    markers = {
        "with_foundation": "o",
        "no_foundation": "s",
    }

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.7), dpi=180)

    # Panel A: Pareto
    method_map = [
        ("PINN-Adam", "PINN-Adam (final)", "Adam"),
        ("ODIL-LM", "ODIL-DQ (residual form)", "levenberg-marquardt"),
        ("ODIL-LBFGS", "ODIL-DQ (residual form)", "lbfgs"),
    ]
    floor_pct = 1e-6
    for short_method, method_name, opt_contains in method_map:
        for scen_name in ["with_foundation", "no_foundation"]:
            mask = (comp_df["scenario"] == scen_name) & (comp_df["method"] == method_name)
            mask &= comp_df["optimizer"].str.contains(opt_contains, case=False, regex=False)
            r = comp_df.loc[mask].iloc[0]
            y = profile_metrics[(scen_name, short_method)]["eps_w_l2_pct"]
            y_plot = max(y, floor_pct)
            axA.scatter(
                float(r["wall_time_s"]),
                y_plot,
                s=78,
                marker=markers[scen_name],
                color=colors[short_method],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(5e-1, 1e4)
    axA.set_ylim(1e-6, 1e0)
    axA.set_xlabel("Total wall time for linear+nonlinear solve, $t_{wall}$ (s)")
    axA.set_ylabel(r"Nonlinear deflection error, $\varepsilon_w^{L2}$ (\%)")
    axA.set_title("A. Cost–accuracy Pareto view")
    axA.grid(True, which="both", alpha=0.25)

    legend_method = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m],
               markeredgecolor="black", markeredgewidth=0.6, markersize=8, label=m)
        for m in ["PINN-Adam", "ODIL-LM", "ODIL-LBFGS"]
    ]
    legend_scenario = [
        Line2D([0], [0], marker=markers["with_foundation"], color="black", linestyle="None",
               markerfacecolor="white", markersize=7, label="WF"),
        Line2D([0], [0], marker=markers["no_foundation"], color="black", linestyle="None",
               markerfacecolor="white", markersize=7, label="NF"),
    ]
    leg1 = axA.legend(handles=legend_method, loc="upper left", fontsize=8, frameon=True)
    axA.add_artist(leg1)
    axA.legend(handles=legend_scenario, loc="lower right", fontsize=8, frameon=True, title="Scenario")
    axA.text(0.03, 0.04, r"ODIL-LM points shown at $10^{-6}\%$ floor; true error = 0.",
             transform=axA.transAxes, fontsize=7.6)

    # Panel B: representative convergence trajectories
    wf_pinn_df = read_csv(paths[("with_foundation", "PINN-Adam")]["loss_csv"])
    wf_odil_df = read_csv(paths[("with_foundation", "ODIL-LM")]["loss_csv"])

    def plot_norm(ax, x, y, color, label, linestyle, marker=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        y = np.abs(y)
        y = y / y[0]
        x = x.copy()
        x[x <= 0] = 1
        ax.plot(
            x,
            y,
            color=color,
            lw=2.0 if marker is None else 1.6,
            linestyle=linestyle,
            marker=marker,
            ms=5 if marker else 0,
            mfc="white" if marker else None,
            label=label,
        )

    plot_norm(axB,
              wf_pinn_df["epoch"].to_numpy(),
              wf_pinn_df["linear_total"].to_numpy(),
              colors["PINN-Adam"], "PINN-Adam, linear", "--")
    plot_norm(axB,
              wf_pinn_df["epoch"].to_numpy(),
              wf_pinn_df["nonlinear_total"].to_numpy(),
              colors["PINN-Adam"], "PINN-Adam, nonlinear", "-")
    plot_norm(axB,
              wf_odil_df["iter"].to_numpy() + 1.0,
              wf_odil_df["linear_loss"].to_numpy(),
              colors["ODIL-LM"], "ODIL-LM, linear", "--", marker="o")
    plot_norm(axB,
              wf_odil_df["iter"].to_numpy() + 1.0,
              wf_odil_df["nonlinear_loss"].to_numpy(),
              colors["ODIL-LM"], "ODIL-LM, nonlinear", "-", marker="o")

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(1e0, 1e5)
    axB.set_ylim(1e-24, 2e0)
    axB.set_xlabel("Optimization step $k$ (epoch for PINN, iteration for ODIL)")
    axB.set_ylabel(r"Normalized method-specific objective, $|J_k|/|J_0|$")
    axB.set_title("B. Representative convergence (with foundation)")
    axB.grid(True, which="both", alpha=0.25)
    axB.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comp_df = load_comparison_rows()
    paths = discover_artifact_paths()
    profile_metrics = compute_profile_errors(paths)

    pinn_model = os.path.join(
        RESULTS_DIR, "pinn", "pure_pinn_adam", "C-C", "X",
        "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh", "models",
        "Linearw_W_0.025_T_300.0_H_0.8_qn0.08_Tanh.pth"
    )
    pinn_param_count = count_pinn_params(pinn_model)

    rows = build_table_rows(comp_df, profile_metrics, pinn_param_count)
    write_table_outputs(rows, profile_metrics)
    make_figure(comp_df, paths, profile_metrics)

    print(f"[ok] wrote {os.path.join(OUT_DIR, 'table_q2.csv')}")
    print(f"[ok] wrote {os.path.join(OUT_DIR, 'table_q2.md')}")
    print(f"[ok] wrote {os.path.join(OUT_DIR, 'table_q2.tex')}")
    print(f"[ok] wrote {os.path.join(OUT_DIR, 'figure_q2.png')}")
    print(f"[ok] wrote {os.path.join(OUT_DIR, 'figure_q2.pdf')}")


if __name__ == "__main__":
    main()
