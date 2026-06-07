"""
PINN-Adam training Pi_beam (= Pi_all) comparison figure
========================================

Reads two PINN-Adam training loss logs under Comparison_of_computational_time
(with foundation vs without foundation) and generates a 1x2 subplot:
    Left (a) Linear Pi_beam
    Right (b) Nonlinear Pi_beam
Each subplot contains two curves:
    With foundation     -> blue dashed line (k1=0.01, k2=0.001)
    Without foundation  -> orange solid line (k1=k2=0)

The visual details fully follow the plot_components_raw style of
plot_depth_analysis_impl.py: Times New Roman, symlog (linthresh=1e-6), log x,
custom tick map, (a)(b) subplot labels, frameon=False legend, blue dashed /
orange solid lines, dpi=900, output png/pdf/svg/eps to
Comparison_of_computational_time/results/plots/.

Run directly (no command-line arguments):
    conda activate claude_test
    python plot_adam_pi_beam.py
"""

from __future__ import annotations

import os
from pathlib import Path
from string import ascii_lowercase

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

matplotlib.use("Agg")

# ============================================================================
# Global style configuration (consistent with plot_depth_analysis_impl.py)
# ============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

FONTSIZE_TITLE = 17
FONTSIZE_LABEL = 16
FONTSIZE_TICK = 13
FONTSIZE_LEGEND = 12
FONTSIZE_PANEL = 16

COLOR_WITH = "#1f77b4"
COLOR_WITHOUT = "#ff7f0e"
LINESTYLE_WITH = "--"
LINESTYLE_WITHOUT = "-"
LINEWIDTH = 2.0

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.labelsize"] = FONTSIZE_TICK
plt.rcParams["ytick.labelsize"] = FONTSIZE_TICK


# ============================================================================
# Path configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent

ADAM_BASE = EXP_ROOT / "results" / "pinn" / "pure_pinn_adam" / "C-C" / "X"

WITH_FOUNDATION_DIR = ADAM_BASE / "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh-k0.01_0.001"
WITHOUT_FOUNDATION_DIR = ADAM_BASE / "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh"

OUTPUT_DIR = EXP_ROOT / "results" / "plots"
OUTPUT_STEM = "pinn_adam_pi_beam"


# ============================================================================
# Utility functions (consistent with plot_depth_analysis_impl.py)
# ============================================================================
def safe_epoch(epoch: np.ndarray) -> np.ndarray:
    epoch = epoch.astype(float)
    epoch[epoch <= 0] = 1.0
    return epoch


def _long_path(p: Path) -> str:
    """Windows MAX_PATH compatibility: prepend the \\\\?\\ prefix to absolute paths longer than 260 characters."""
    s = str(p.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def load_loss_csv(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("loss_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No loss_*.csv under: {folder}")
    return pd.read_csv(_long_path(csv_files[0]))


def ensure_epoch(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df.columns:
        df = df.copy()
        df["epoch"] = np.arange(1, len(df) + 1)
    return df


def apply_pi_beam_y_ticks(ax) -> None:
    """Reuse the Pi_beam tick set from (row=0/1, col_index=0) in plot_components_raw."""
    custom_ticks = [-1e-2, -1e-4, -1e-6, 1e-6, 1e-4, 1e-2]
    y_lower, y_upper = ax.get_ylim()
    visible_ticks = [tick for tick in custom_ticks if y_lower <= tick <= y_upper]
    if not visible_ticks:
        return

    def _tick_formatter(value: float, _pos=None) -> str:
        for tick in visible_ticks:
            if np.isclose(value, tick, rtol=0.0, atol=abs(tick) * 1e-12 + 1e-15):
                exponent = int(round(np.log10(abs(tick))))
                sign = "-" if tick < 0 else ""
                return rf"${sign}10^{{{exponent}}}$"
        return ""

    ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_tick_formatter))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="minor", left=False)


# ============================================================================
# Main plotting function
# ============================================================================
def plot_pi_beam_adam(
    with_df: pd.DataFrame,
    without_df: pd.DataFrame,
    output_dir: Path,
    output_stem: str,
) -> None:
    """1x2 layout, plotting only Pi_all (Pi_beam): left=Linear, right=Nonlinear."""
    component_title_size = FONTSIZE_TITLE + 2
    component_label_size = FONTSIZE_LABEL + 2
    component_tick_size = FONTSIZE_TICK + 2
    component_legend_size = FONTSIZE_LEGEND + 2
    component_panel_size = FONTSIZE_PANEL + 2

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.5))
    title_text = r"$\Pi_{beam}$"

    for idx, (ax, mode) in enumerate(zip(axes, ["linear", "nonlinear"])):
        col = f"{mode}_Pi_all"
        if col not in with_df.columns or col not in without_df.columns:
            ax.set_visible(False)
            continue

        epoch_with = safe_epoch(with_df["epoch"].to_numpy())
        values_with = with_df[col].to_numpy()
        epoch_without = safe_epoch(without_df["epoch"].to_numpy())
        values_without = without_df[col].to_numpy()

        # Draw Without foundation first (orange solid line, bottom layer), then With foundation (blue dashed line, top layer),
        # to prevent the blue dashed line from being covered by the orange solid line. The two nearly overlap, so the dashed line must be on top to be visible.
        line_without, = ax.plot(
            epoch_without, values_without,
            linestyle=LINESTYLE_WITHOUT, color=COLOR_WITHOUT,
            linewidth=LINEWIDTH, label="Without foundation",
            zorder=2,
        )
        line_with, = ax.plot(
            epoch_with, values_with,
            linestyle=LINESTYLE_WITH, color=COLOR_WITH,
            linewidth=LINEWIDTH, label="With foundation",
            zorder=3,
        )

        y_min = float(min(np.min(values_with), np.min(values_without)))
        y_max = float(max(np.max(values_with), np.max(values_without)))
        y_pad = max((y_max - y_min) * 0.08, 1e-10)
        y_lower = y_min - y_pad
        y_upper = y_max + y_pad
        if y_lower >= y_upper:
            y_lower = y_upper - max(abs(y_upper) * 0.1, 1e-6)
        ax.set_ylim(y_lower, y_upper)

        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        apply_pi_beam_y_ticks(ax)
        ax.tick_params(axis="both", labelsize=component_tick_size)

        ax.set_xlabel("Epoch", fontsize=component_label_size)
        mode_label = "Linear Energy" if mode == "linear" else "Nonlinear Energy"
        ax.set_ylabel(mode_label, fontsize=component_label_size)
        ax.set_title(title_text, fontsize=component_title_size)

        ax.legend(
            handles=[line_with, line_without],
            fontsize=component_legend_size, frameon=False,
            loc="lower left", bbox_to_anchor=(0, 0), borderaxespad=0,
        )

        ax.text(
            0.03, 0.95,
            f"({ascii_lowercase[idx]})",
            transform=ax.transAxes,
            fontsize=component_panel_size, fontweight="bold",
            va="top", ha="left",
        )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(_long_path(output_dir / f"{output_stem}.png"), dpi=900)
    fig.savefig(_long_path(output_dir / f"{output_stem}.svg"), format="svg")
    fig.savefig(_long_path(output_dir / f"{output_stem}.pdf"), format="pdf")
    fig.savefig(_long_path(output_dir / f"{output_stem}.eps"), format="eps")
    plt.close(fig)


def main() -> None:
    print(f"[INFO] with foundation logs:    {WITH_FOUNDATION_DIR}")
    print(f"[INFO] without foundation logs: {WITHOUT_FOUNDATION_DIR}")

    with_df = ensure_epoch(load_loss_csv(WITH_FOUNDATION_DIR / "logs"))
    without_df = ensure_epoch(load_loss_csv(WITHOUT_FOUNDATION_DIR / "logs"))

    print(f"[INFO] with    rows = {len(with_df)}, cols = {list(with_df.columns)}")
    print(f"[INFO] without rows = {len(without_df)}")

    plot_pi_beam_adam(with_df, without_df, OUTPUT_DIR, OUTPUT_STEM)

    print(f"[OK] saved -> {OUTPUT_DIR / OUTPUT_STEM}.{{png,pdf,svg,eps}}")


if __name__ == "__main__":
    main()
