"""Extract numeric data for the rebuttal Table X.

Pulls max|w|, R² (full-field, vs ODIL-LM reference), GPU wall time, and
converged status for the four methods x two scenarios. Outputs both the
raw values and the formatted markdown table.

R² is computed against ODIL-LM nonlinear w(x) (residual ~1e-11 ~ discrete
ground truth) on a common 200-point grid built by interpolating each
method's stored nonlinear_w(x) onto ODIL-LM's reference grid.

Run:  python extract_table_data.py
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS_DIR, "results")

# -------- file locations ----------------------------------------------
ODIL_DISP = lambda scen, opt: os.path.join(
    RES, "odil", scen, opt, "data", "displacement.csv")

PINN_DISP_WF = os.path.join(
    RES, "pinn", "pure_pinn_adam", "C-C", "X",
    "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh-k0.01_0.001",
    "data", "w_W_0.025_T_300.0_H_0.8_qn0.08_Tanh_k1_0.01_k2_0.001.csv",
)
PINN_DISP_NF = os.path.join(
    RES, "pinn", "pure_pinn_adam", "C-C", "X",
    "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh",
    "data", "w_W_0.025_T_300.0_H_0.8_qn0.08_Tanh.csv",
)

ODIL_INDEX = os.path.join(RES, "odil", "sweep_index.json")
PINN_INDEX = os.path.join(RES, "pinn", "sweep_index.json")

# Reference scalar values given by the user
REF_W = {"no_foundation": 0.46520, "with_foundation": 0.42895}

# External reference solutions for R² (full-field nonlinear w(x))
REF_PROFILE = {
    "no_foundation": os.path.join(
        RES, "nonlinear_C-C_N13_A00.00_k10.000_k20.000.csv"),
    "with_foundation": os.path.join(
        RES, "nonlinear_C-C_N13_A00.00_k10.010_k20.001.csv"),
}


def lp(path: str) -> str:
    """Windows long-path helper (>260 chars)."""
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    return "\\\\?\\" + os.path.abspath(path)


def load_w(csv_path):
    df = pd.read_csv(lp(csv_path))
    return df["x"].to_numpy(float), df["nonlinear_w"].to_numpy(float)


def load_ref_w(csv_path):
    """External reference CSV uses columns x, u, w, phi (no 'nonlinear_' prefix)."""
    df = pd.read_csv(lp(csv_path))
    return df["x"].to_numpy(float), df["w"].to_numpy(float)


def r2_full_field(x_method, w_method, x_ref, w_ref):
    """R² of method vs reference on the reference grid (interpolating method)."""
    if len(x_method) != len(x_ref) or not np.allclose(x_method, x_ref):
        w_method_on_ref = np.interp(x_ref, x_method, w_method)
    else:
        w_method_on_ref = w_method
    ss_res = float(np.sum((w_method_on_ref - w_ref) ** 2))
    ss_tot = float(np.sum((w_ref - np.mean(w_ref)) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    # ------- load ODIL sweep summary --------
    with open(ODIL_INDEX) as f:
        odil_runs = json.load(f)["runs"]

    odil_time = {}
    odil_iters = {}
    odil_residual = {}
    for r in odil_runs:
        key = (r["scenario"], r["optimizer"])
        odil_time[key] = r["linear"]["elapsed_s"] + r["nonlinear"]["elapsed_s"]
        odil_iters[key] = (r["linear"]["iterations"]
                           + r["nonlinear"]["iterations"])
        odil_residual[key] = r["nonlinear"]["R_PDE_norm"]

    # ------- load PINN sweep summary --------
    with open(PINN_INDEX) as f:
        pinn_runs = json.load(f)["runs"]

    pinn_time = {}
    for r in pinn_runs:
        if r.get("optimizer") == "adam":
            pinn_time[r["scenario"]] = r["elapsed_s"]

    # ------- load displacement profiles ----
    profiles = {}
    for scen in ("no_foundation", "with_foundation"):
        profiles[(scen, "ODIL-LM")] = load_w(
            ODIL_DISP(scen, "levenberg-marquardt"))
        profiles[(scen, "ODIL-LBFGS")] = load_w(ODIL_DISP(scen, "lbfgs"))
        profiles[(scen, "ODIL-GN")] = load_w(ODIL_DISP(scen, "gauss-newton"))
        profiles[(scen, "PINN-Adam")] = load_w(
            PINN_DISP_NF if scen == "no_foundation" else PINN_DISP_WF)

    # ------- load external reference profiles --------
    ref_profiles = {}
    for scen in ("no_foundation", "with_foundation"):
        ref_profiles[scen] = load_ref_w(REF_PROFILE[scen])
        x_ref, w_ref = ref_profiles[scen]
        print(f"[ref] {scen}: {REF_PROFILE[scen]} "
              f"(N={len(x_ref)}, max|w|_ref={np.max(np.abs(w_ref)):.6f})")

    # ------- compute metrics ---------------
    rows = {}
    for scen in ("no_foundation", "with_foundation"):
        x_ref, w_ref = ref_profiles[scen]
        for method in ("ODIL-LBFGS", "ODIL-GN", "ODIL-LM", "PINN-Adam"):
            x_m, w_m = profiles[(scen, method)]
            wmax = float(np.max(np.abs(w_m)))
            r2 = r2_full_field(x_m, w_m, x_ref, w_ref)
            rel_diff_pct = 100.0 * abs(wmax - REF_W[scen]) / abs(REF_W[scen])

            # GPU time + converged status
            if method == "ODIL-LM":
                t = odil_time[(scen, "levenberg-marquardt")]
                conv = odil_residual[(scen, "levenberg-marquardt")] < 1e-6
            elif method == "ODIL-LBFGS":
                t = odil_time[(scen, "lbfgs")]
                conv = odil_residual[(scen, "lbfgs")] < 1e-6
            elif method == "ODIL-GN":
                t = odil_time[(scen, "gauss-newton")]
                conv = odil_residual[(scen, "gauss-newton")] < 1e-6
            else:  # PINN-Adam
                t = pinn_time[scen]
                # PINN converged if max|w| within physical range vs reference
                conv = abs(wmax - REF_W[scen]) / abs(REF_W[scen]) < 0.05

            rows[(scen, method)] = {
                "wmax": wmax,
                "rel_diff_pct": rel_diff_pct,
                "r2": r2,
                "time_s": t,
                "converged": "Yes" if conv else "No",
            }

    # ------- format table ------------------
    methods = ("ODIL-LBFGS", "ODIL-GN", "ODIL-LM", "PINN-Adam")
    method_labels = ("ODL-DQ (L-BFGS)", "ODL-DQ (GN)", "ODL-DQ (LM)",
                     "Energy-based PINN (Adam)")

    def fmt_wmax(v): return f"{v:.5f}"
    def fmt_rel(v):
        if v < 1e-2:
            return f"{v:.3g}"
        return f"{v:.3f}"
    def fmt_r2(v):
        if v >= 0:
            return f"{v:.6f}"
        return f"{v:.6f}"
    def fmt_t(v): return f"{v:.2f}"

    print("\n=== RAW METRICS ===")
    for scen in ("no_foundation", "with_foundation"):
        print(f"\n--- {scen} (ref = {REF_W[scen]}) ---")
        for m in methods:
            r = rows[(scen, m)]
            print(f"  {m:14s}: w_max={r['wmax']:.6f}, "
                  f"rel_diff={r['rel_diff_pct']:.4g}%, "
                  f"R2={r['r2']:.8f}, t={r['time_s']:.2f}s, conv={r['converged']}")

    # ------- markdown table ----------------
    md = []
    md.append("Table X. Comparison between ODL-DQ methods and energy-based PINN "
              "for the nonlinear FG-GOEAM beam benchmark.")
    md.append("")
    md.append("| Method | " + " | ".join(method_labels) + " |")
    md.append("|---|" + "---:|" * len(method_labels))

    def row(label, key_fn):
        cells = [key_fn(rows[("no_foundation", m)] if "without" in label
                        else rows[("with_foundation", m)]) for m in methods]
        return f"| {label} | " + " | ".join(cells) + " |"

    md.append(row("w_max, without foundation", lambda r: fmt_wmax(r["wmax"])))
    md.append(row("w_max, with foundation",    lambda r: fmt_wmax(r["wmax"])))
    md.append(row("Relative difference, without foundation",
                  lambda r: f"{fmt_rel(r['rel_diff_pct'])}%"))
    md.append(row("Relative difference, with foundation",
                  lambda r: f"{fmt_rel(r['rel_diff_pct'])}%"))
    md.append(row("R^2, without foundation", lambda r: fmt_r2(r["r2"])))
    md.append(row("R^2, with foundation",    lambda r: fmt_r2(r["r2"])))
    md.append(row("Computational time (GPU), without foundation",
                  lambda r: f"{fmt_t(r['time_s'])} s"))
    md.append(row("Computational time (GPU), with foundation",
                  lambda r: f"{fmt_t(r['time_s'])} s"))
    md.append(row("Converged, without foundation", lambda r: r["converged"]))
    md.append(row("Converged, with foundation",    lambda r: r["converged"]))
    md.append("")
    md.append("Relative difference is calculated with respect to the reference "
              "DQ solutions, 0.46520 for the case without foundation and 0.42895 "
              "for the case with foundation. The R^2 value is calculated using "
              "the full transverse deflection field w(x).")

    out_md = os.path.join(RES, "rebuttal", "table_x_paper.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n[ok] Wrote {out_md}")

    # ------- CSV output (same row/column schema, no markdown formatting) -------
    csv_rows = [["Method"] + list(method_labels)]

    def csv_row(label, key_fn):
        scen = "no_foundation" if "without" in label else "with_foundation"
        return [label] + [key_fn(rows[(scen, m)]) for m in methods]

    csv_rows.append(csv_row("w_max, without foundation",
                            lambda r: fmt_wmax(r["wmax"])))
    csv_rows.append(csv_row("w_max, with foundation",
                            lambda r: fmt_wmax(r["wmax"])))
    csv_rows.append(csv_row("Relative difference, without foundation",
                            lambda r: f"{fmt_rel(r['rel_diff_pct'])}%"))
    csv_rows.append(csv_row("Relative difference, with foundation",
                            lambda r: f"{fmt_rel(r['rel_diff_pct'])}%"))
    csv_rows.append(csv_row("R^2, without foundation",
                            lambda r: fmt_r2(r["r2"])))
    csv_rows.append(csv_row("R^2, with foundation",
                            lambda r: fmt_r2(r["r2"])))
    csv_rows.append(csv_row("Computational time (GPU), without foundation",
                            lambda r: f"{fmt_t(r['time_s'])} s"))
    csv_rows.append(csv_row("Computational time (GPU), with foundation",
                            lambda r: f"{fmt_t(r['time_s'])} s"))
    csv_rows.append(csv_row("Converged, without foundation",
                            lambda r: r["converged"]))
    csv_rows.append(csv_row("Converged, with foundation",
                            lambda r: r["converged"]))

    out_csv = os.path.join(RES, "rebuttal", "table_x_paper.csv")
    import csv as _csv
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = _csv.writer(f)
        wr.writerows(csv_rows)
        wr.writerow([])  # blank line
        wr.writerow(["Note: Relative difference is calculated with respect to "
                     "the reference DQ solutions, 0.46520 for the case without "
                     "foundation and 0.42895 for the case with foundation. The "
                     "R^2 value is calculated using the full transverse "
                     "deflection field w(x)."])
    print(f"[ok] Wrote {out_csv}")

    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
