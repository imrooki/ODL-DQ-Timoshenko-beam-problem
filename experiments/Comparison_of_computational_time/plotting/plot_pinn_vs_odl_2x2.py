"""
2x2 PINN vs ODL training-history figure
==================================================

Layout (figsize=(15, 9), each subplot 7.5x4.5 = 15:9):
    (a) PINN linear      Pi_beam   (epoch, symlog y)        2 lines  With/Without foundation
    (b) ODL  linear      Loss      (iter, log y)            6 lines  {With,Without} x {LM,GN,LBFGS}
    (c) PINN nonlinear   Pi_beam   (epoch, symlog y)        2 lines
    (d) ODL  nonlinear   Loss      (iter, log y)            6 lines

Color = scenario  (blue = With foundation, orange = Without foundation)
Line  = (a)(c) scenario  /  (b)(d) optimizer (solid LM, dashed GN, dash-dot LBFGS)

Inputs:
    PINN-Adam logs  Comparison_of_computational_time/results/pinn/pure_pinn_adam/C-C/X/.../logs/loss_*.csv
    ODL  loss_history  Comparison_of_computational_time/results/odil/{with,no}_foundation/{LM,GN,LBFGS}/data/loss_history.csv

Output:
    Comparison_of_computational_time/results/plots/pinn_vs_odl_2x2.{png,pdf,svg,eps}
"""

from __future__ import annotations

import os
from pathlib import Path
from string import ascii_lowercase

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")

# ============================================================================
# Style (consistent with the style of plot_depth_analysis_impl.py)
# ============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

FONTSIZE_TITLE = 17
FONTSIZE_LABEL = 16
FONTSIZE_TICK = 13
FONTSIZE_LEGEND = 11
FONTSIZE_PANEL = 16

COLOR_WITH = "#1f77b4"
COLOR_WITHOUT = "#ff7f0e"
LINEWIDTH = 2.0

# (a)(c) line style: With=dashed, Without=solid
LS_WITH_PINN = "--"
LS_WITHOUT_PINN = "-"

# (b)(d) line style: optimizer
LS_LM = "-"
LS_GN = "--"
LS_LBFGS = "-."

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.labelsize"] = FONTSIZE_TICK
plt.rcParams["ytick.labelsize"] = FONTSIZE_TICK


# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent

PINN_BASE = EXP_ROOT / "results" / "pinn" / "pure_pinn_adam" / "C-C" / "X"
PINN_WITH_DIR = PINN_BASE / "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh-k0.01_0.001"
PINN_WITHOUT_DIR = PINN_BASE / "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh"

ODL_BASE = EXP_ROOT / "results" / "odil"
ODL_WITH_DIR = ODL_BASE / "with_foundation"
ODL_WITHOUT_DIR = ODL_BASE / "no_foundation"

OPTIMIZERS = [
    ("levenberg-marquardt", "LM",     LS_LM),
    ("gauss-newton",        "GN",     LS_GN),
    ("lbfgs",               "L-BFGS", LS_LBFGS),
]

# The underscore segment in the filename corresponds to each optimizer's stem (hyphens removed)
OPT_FILE_STEM = {
    "levenberg-marquardt": "levenbergmarquardt",
    "gauss-newton":        "gaussnewton",
    "lbfgs":               "lbfgs",
}

OUTPUT_DIR = EXP_ROOT / "results" / "plots"
OUTPUT_STEM = "pinn_vs_odl_2x2"


# ============================================================================
# Helpers
# ============================================================================
def _long_path(p: Path) -> str:
    s = str(p.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def load_pinn_loss(folder: Path) -> pd.DataFrame:
    csv_files = sorted((folder / "logs").glob("loss_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No PINN loss_*.csv under: {folder/'logs'}")
    df = pd.read_csv(_long_path(csv_files[0]))
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)
    return df


def load_odl_dq_loss(scenario_dir: Path, opt_name: str, mode: str) -> pd.DataFrame:
    """
    Load the new-format ODL DQ loss CSV:
        <scenario_dir>/<opt_name>/DQ_<opt_stem>_<mode>_loss.csv
    Columns: epoch, loss, pde_loss, bc_loss, time_elapsed, gradient_norm, learning_rate
    """
    opt_stem = OPT_FILE_STEM[opt_name]
    p = scenario_dir / opt_name / f"DQ_{opt_stem}_{mode}_loss.csv"
    if not p.exists():
        raise FileNotFoundError(f"No DQ loss CSV: {p}")
    return pd.read_csv(_long_path(p))


def safe_log_x(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x[x <= 0] = 1.0
    return x


def smooth_loss_curve(x: np.ndarray, y: np.ndarray, n: int = 300) -> tuple:
    """
    PCHIP monotonic shape-preserving interpolation in log-y space, designed
    specifically for the monotonically decreasing ODIL loss curves.
    Returns (x_smooth, y_smooth). Falls back to the original discrete points when
    there are fewer than 3 data points or any non-positive values are present.
    """
    if len(x) < 3 or not np.all(y > 0):
        return x, y
    sort_idx = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[sort_idx]
    log_y_sorted = np.log10(np.asarray(y, dtype=float)[sort_idx])
    pchip = PchipInterpolator(x_sorted, log_y_sorted, extrapolate=False)
    x_smooth = np.linspace(x_sorted[0], x_sorted[-1], n)
    y_smooth = 10 ** pchip(x_smooth)
    return x_smooth, y_smooth


def apply_pi_beam_y_ticks(ax) -> None:
    custom_ticks = [-1e-2, -1e-4, -1e-6, 1e-6, 1e-4, 1e-2]
    y_lower, y_upper = ax.get_ylim()
    visible_ticks = [t for t in custom_ticks if y_lower <= t <= y_upper]
    if not visible_ticks:
        return

    def _fmt(value: float, _pos=None) -> str:
        for tick in visible_ticks:
            if np.isclose(value, tick, rtol=0.0, atol=abs(tick) * 1e-12 + 1e-15):
                exp = int(round(np.log10(abs(tick))))
                sign = "-" if tick < 0 else ""
                return rf"${sign}10^{{{exp}}}$"
        return ""

    ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="minor", left=False)


# ============================================================================
# Sub-plots
# ============================================================================
def plot_pinn_panel(ax, with_df: pd.DataFrame, without_df: pd.DataFrame, mode: str) -> tuple:
    """(a)/(c) PINN energy panel: epoch vs symlog Pi_all."""
    col = f"{mode}_Pi_all"
    e_with = safe_log_x(with_df["epoch"].to_numpy())
    v_with = with_df[col].to_numpy()
    e_without = safe_log_x(without_df["epoch"].to_numpy())
    v_without = without_df[col].to_numpy()

    # For 100,000 rows of data, rasterized=True embeds the curve as a bitmap, greatly reducing the SVG size;
    # the axes / legend / text remain vector graphics.
    line_without, = ax.plot(
        e_without, v_without,
        linestyle=LS_WITHOUT_PINN, color=COLOR_WITHOUT,
        linewidth=LINEWIDTH, label="Without foundation",
        zorder=2, rasterized=True,
    )
    line_with, = ax.plot(
        e_with, v_with,
        linestyle=LS_WITH_PINN, color=COLOR_WITH,
        linewidth=LINEWIDTH, label="With foundation",
        zorder=3, rasterized=True,
    )

    finite = np.concatenate([v_with[np.isfinite(v_with)], v_without[np.isfinite(v_without)]])
    if finite.size > 0:
        y_min, y_max = float(np.min(finite)), float(np.max(finite))
        pad = max((y_max - y_min) * 0.08, 1e-10)
        y_lower, y_upper = y_min - pad, y_max + pad
        if y_lower >= y_upper:
            y_lower = y_upper - max(abs(y_upper) * 0.1, 1e-6)
        ax.set_ylim(y_lower, y_upper)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    apply_pi_beam_y_ticks(ax)

    return line_with, line_without


def plot_odl_panel(ax, odl_with: dict, odl_without: dict, mode: str) -> tuple:
    """
    (b)/(d) ODL DQ loss panel with dual linear X axes:
        bottom X (ax)      -> LM, GN     (short epoch range)
        top X    (ax_top)  -> L-BFGS     (long epoch range)
    Y axis = log loss, shared between ax and ax_top via twiny.

    Data comes from DQ_<opt>_<mode>_loss.csv, columns: epoch, loss, pde_loss, bc_loss, ...
    odl_with[opt_name][mode] -> DataFrame
    """
    ax_top = ax.twiny()

    # ---- LM, GN on bottom X (ax) ----
    bottom_max = 0
    for opt_name, opt_ls, opt_marker in [
        ("levenberg-marquardt", LS_LM, "o"),
        ("gauss-newton",        LS_GN, "s"),
    ]:
        for scenario, color, df_map in [
            ("with",    COLOR_WITH,    odl_with),
            ("without", COLOR_WITHOUT, odl_without),
        ]:
            df = df_map.get(opt_name, {}).get(mode)
            if df is None or "epoch" not in df.columns or "loss" not in df.columns:
                continue
            x = df["epoch"].to_numpy().astype(float)
            y_loss = df["loss"].to_numpy()
            mask = np.isfinite(y_loss) & (y_loss > 0)
            if not np.any(mask):
                continue
            # ||R_PDE||_2 = sqrt(loss) (C-C hard constraint -> loss = loss_pde + 0 + reg ~ 0)
            y = np.sqrt(y_loss)
            x_clean, y_clean = x[mask], y[mask]
            # Smoothed curve (log-y PCHIP shape-preserving interpolation)
            x_s, y_s = smooth_loss_curve(x_clean, y_clean, n=300)
            ax.plot(
                x_s, y_s,
                linestyle=opt_ls, color=color,
                linewidth=LINEWIDTH,
                zorder=3 if scenario == "with" else 2,
            )
            # Place a marker only at the position of the last epoch (marking the convergence point)
            ax.plot(
                [x_clean[-1]], [y_clean[-1]],
                linestyle="none", color=color,
                marker=opt_marker, markersize=9,
                markeredgewidth=0.6, markerfacecolor=color,
                zorder=4 if scenario == "with" else 3,
            )
            bottom_max = max(bottom_max, float(np.max(x_clean)))

    # ---- L-BFGS on top X (ax_top) ----
    top_max = 0
    for scenario, color, df_map in [
        ("with",    COLOR_WITH,    odl_with),
        ("without", COLOR_WITHOUT, odl_without),
    ]:
        df = df_map.get("lbfgs", {}).get(mode)
        if df is None or "epoch" not in df.columns or "loss" not in df.columns:
            continue
        x = df["epoch"].to_numpy().astype(float)
        y_loss = df["loss"].to_numpy()
        mask = np.isfinite(y_loss) & (y_loss > 0)
        if not np.any(mask):
            continue
        # ||R_PDE||_2 = sqrt(loss)
        y = np.sqrt(y_loss)
        x_clean, y_clean = x[mask], y[mask]
        x_s, y_s = smooth_loss_curve(x_clean, y_clean, n=300)
        ax_top.plot(
            x_s, y_s,
            linestyle=LS_LBFGS, color=color,
            linewidth=LINEWIDTH,
            zorder=3 if scenario == "with" else 2,
        )
        # Place a marker only at the position of the last epoch (marking the convergence point)
        ax_top.plot(
            [x_clean[-1]], [y_clean[-1]],
            linestyle="none", color=color,
            marker="^", markersize=9,
            markeredgewidth=0.6, markerfacecolor=color,
            zorder=4 if scenario == "with" else 3,
        )
        top_max = max(top_max, float(np.max(x_clean)))

    # Y axis: log
    ax.set_yscale("log")

    # X axes: linear (default), leave a small margin
    if bottom_max > 0:
        ax.set_xlim(0, bottom_max * 1.05)
    if top_max > 0:
        ax_top.set_xlim(0, top_max * 1.05)

    # Force integer ticks (iter is a discrete positive integer; avoids non-integer ticks like 0.25/0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Custom legend handles:
    #   (1) optimizer line style + marker (black, 3 items)
    #   (2) foundation color block (blue/orange, 2 items)
    optimizer_handles = [
        Line2D([0], [0], color="black", linestyle=LS_LM, linewidth=LINEWIDTH,
               marker="o", markersize=9, label="LM"),
        Line2D([0], [0], color="black", linestyle=LS_GN, linewidth=LINEWIDTH,
               marker="s", markersize=9, label="GN"),
        Line2D([0], [0], color="black", linestyle=LS_LBFGS, linewidth=LINEWIDTH,
               marker="^", markersize=9, label="L-BFGS"),
    ]
    foundation_handles = [
        Patch(facecolor=COLOR_WITH,    edgecolor="none", label="With foundation"),
        Patch(facecolor=COLOR_WITHOUT, edgecolor="none", label="Without foundation"),
    ]

    return ax_top, optimizer_handles, foundation_handles


# ============================================================================
# Main figure
# ============================================================================
def make_figure() -> None:
    pinn_with = load_pinn_loss(PINN_WITH_DIR)
    pinn_without = load_pinn_loss(PINN_WITHOUT_DIR)

    # Nested dict: odl_with[opt_name][mode] -> DataFrame
    odl_with = {
        opt: {
            "linear":    load_odl_dq_loss(ODL_WITH_DIR, opt, "linear"),
            "nonlinear": load_odl_dq_loss(ODL_WITH_DIR, opt, "nonlinear"),
        }
        for opt, _, _ in OPTIMIZERS
    }
    odl_without = {
        opt: {
            "linear":    load_odl_dq_loss(ODL_WITHOUT_DIR, opt, "linear"),
            "nonlinear": load_odl_dq_loss(ODL_WITHOUT_DIR, opt, "nonlinear"),
        }
        for opt, _, _ in OPTIMIZERS
    }

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0))

    title_size = FONTSIZE_TITLE + 2
    label_size = FONTSIZE_LABEL + 2
    tick_size = FONTSIZE_TICK + 2
    legend_size = FONTSIZE_LEGEND + 1
    panel_size = FONTSIZE_PANEL + 2

    panel_idx = 0
    for row, mode in enumerate(["linear", "nonlinear"]):
        # For PINN (a)(c), move the title down to y=1.1 (the left column has no ax_top conflict, so it can be aligned uniformly)
        # ---- (a)/(c) PINN ----
        ax = axes[row, 0]
        line_with, line_without = plot_pinn_panel(ax, pinn_with, pinn_without, mode)
        ax.set_title(r"PINN: $\Pi_{beam}$", fontsize=title_size, y=1.1)
        ax.set_ylabel("Linear" if row == 0 else "Nonlinear", fontsize=label_size)
        if row == 1:
            ax.set_xlabel("Epoch", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.legend(
            handles=[line_with, line_without],
            fontsize=legend_size + 3, frameon=False,
            loc="lower left", bbox_to_anchor=(0.005, 0.005), borderaxespad=0,
        )
        ax.text(
            0.03, 0.95, f"({ascii_lowercase[panel_idx]})",
            transform=ax.transAxes,
            fontsize=panel_size, fontweight="bold",
            va="top", ha="left",
        )
        panel_idx += 1

        # ---- (b)/(d) ODL ----
        ax = axes[row, 1]
        ax_top, optimizer_handles, foundation_handles = plot_odl_panel(
            ax, odl_with, odl_without, mode
        )

        # The ODL subplot has an ax_top X-axis label at the top, so the title must stay at y=1.18 to avoid collision
        ax.set_title("ODL: PDE Residual", fontsize=title_size, y=1.18)

        # Top X axis (L-BFGS): draw the xlabel on every row
        ax_top.set_xlabel("Epoch (L-BFGS)", fontsize=label_size, labelpad=4)
        ax_top.tick_params(axis="x", labelsize=tick_size)

        # Bottom X axis (LM, GN): draw the xlabel only on the last row
        if row == 1:
            ax.set_xlabel("Epoch (LM, GN)", fontsize=label_size)
        ax.tick_params(axis="x", labelsize=tick_size)
        ax.tick_params(axis="y", labelsize=tick_size)

        # First legend: optimizers (line style + marker), upper right ncol=3
        legend_opt = ax.legend(
            handles=optimizer_handles,
            fontsize=legend_size, frameon=False,
            loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0.3,
            ncol=3, columnspacing=1.2, handlelength=2.4,
        )
        ax.add_artist(legend_opt)

        # Second legend: foundation colors (patches), shifted right to avoid the early descending segment of LM/GN;
        # font size increased by 3 units, no background. (b) at 0.10, (d) at 0.15.
        foundation_x = 0.10 if row == 0 else 0.15
        ax.legend(
            handles=foundation_handles,
            fontsize=legend_size + 3, frameon=False,
            loc="lower left", bbox_to_anchor=(foundation_x, 0.005), borderaxespad=0,
            ncol=1, handlelength=1.2, handleheight=1.0,
        )
        ax.text(
            0.03, 0.05, f"({ascii_lowercase[panel_idx]})",
            transform=ax.transAxes,
            fontsize=panel_size, fontweight="bold",
            va="bottom", ha="left",
        )
        panel_idx += 1

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # bbox_inches='tight' automatically trims the whitespace outside the figure, pad_inches=0.05 keeps a tiny margin
    save_kw = dict(bbox_inches="tight", pad_inches=0.05)
    fig.savefig(_long_path(OUTPUT_DIR / f"{OUTPUT_STEM}.png"), dpi=900, **save_kw)
    # rasterized portions are embedded as bitmaps: SVG uses dpi=200 to keep the size < 500 KB;
    # PDF uses dpi=300 (the PDF backend compresses well); EPS uses dpi=150 (the EPS backend compresses poorly, to avoid huge files)
    fig.savefig(_long_path(OUTPUT_DIR / f"{OUTPUT_STEM}.svg"), format="svg", dpi=200, **save_kw)
    fig.savefig(_long_path(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"), format="pdf", dpi=300, **save_kw)
    fig.savefig(_long_path(OUTPUT_DIR / f"{OUTPUT_STEM}.eps"), format="eps", dpi=150, **save_kw)
    plt.close(fig)

    print(f"[OK] saved -> {OUTPUT_DIR / OUTPUT_STEM}.{{png,pdf,svg,eps}}")


def main() -> None:
    print(f"[INFO] PINN with    : {PINN_WITH_DIR}")
    print(f"[INFO] PINN without : {PINN_WITHOUT_DIR}")
    print(f"[INFO] ODL  with    : {ODL_WITH_DIR}")
    print(f"[INFO] ODL  without : {ODL_WITHOUT_DIR}")
    make_figure()


if __name__ == "__main__":
    main()
