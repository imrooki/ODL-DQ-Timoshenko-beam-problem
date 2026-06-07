from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")

# ============================================================================
# Global style configuration - following the style of image 1.png
# ============================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # STIX font, for nicer-looking math symbols

# Font size configuration (all increased by 3 units)
FONTSIZE_TITLE = 17       # Subplot title font size (increased by 2 more)
FONTSIZE_LABEL = 16       # Axis label font size (increased by 2 more)
FONTSIZE_TICK = 13        # Tick font size
FONTSIZE_LEGEND = 12      # Legend font size
FONTSIZE_PANEL = 16       # Subplot label (a)(b), etc. (increased by 2 more)

# Color configuration - matplotlib default blue/orange color scheme
COLOR_PINN = "#1f77b4"    # Blue
COLOR_PS_PINN = "#ff7f0e" # Orange

# Line style configuration
LINESTYLE_PINN = "--"     # PINN dashed line
LINESTYLE_PS_PINN = "-"   # PS-PINN solid line
LINEWIDTH = 2.0           # Line width (bold)

# Tick configuration
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.labelsize"] = FONTSIZE_TICK
plt.rcParams["ytick.labelsize"] = FONTSIZE_TICK


def load_loss_csv(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("loss_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"loss_*.csv not found: {folder}")
    return pd.read_csv(csv_files[0])


def ensure_epoch(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df.columns:
        df = df.copy()
        df["epoch"] = np.arange(1, len(df) + 1)
    return df


def apply_component_custom_y_ticks(ax, *, row: int, col_index: int) -> None:
    """Customize y tick labels for selected energy-component panels only."""
    tick_map = {
        (0, 0): [-1e-2, -1e-4, -1e-6, 1e-6, 1e-4, 1e-2],
        (1, 0): [-1e-2, -1e-4, -1e-6, 1e-6, 1e-4, 1e-2],
        (0, 3): [-1e-2, -1e-4, -1e-6, 1e-6, 1e-4],
    }
    custom_ticks = tick_map.get((row, col_index))
    if custom_ticks is None:
        return

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


def plot_total_raw(case_dir: Path, pinn_df: pd.DataFrame, ps_df: pd.DataFrame) -> None:
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if f"{mode}_total" in pinn_df.columns
        and f"{mode}_total" in ps_df.columns
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 4.0))
    if len(modes) == 1:
        axes = [axes]
    for idx, (ax, mode) in enumerate(zip(axes, modes)):
        col = f"{mode}_total"
        # PINN - blue dashed line
        epoch_pinn = safe_epoch(pinn_df["epoch"].to_numpy())
        values_pinn = pinn_df[col].to_numpy()
        ax.plot(epoch_pinn, values_pinn, linestyle=LINESTYLE_PINN,
                color=COLOR_PINN, linewidth=LINEWIDTH, label="PINN")
        # PS-PINN - orange solid line
        epoch_ps = safe_epoch(ps_df["epoch"].to_numpy())
        values_ps = ps_df[col].to_numpy()
        ax.plot(epoch_ps, values_ps, linestyle=LINESTYLE_PS_PINN,
                color=COLOR_PS_PINN, linewidth=LINEWIDTH, label="PS-PINN")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Epoch", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Total Loss", fontsize=FONTSIZE_LABEL)
        mode_title = "Linear" if mode == "linear" else "Nonlinear"
        ax.set_title(f"{mode_title} Total Loss", fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
        # Subplot label
        ax.text(0.03, 0.95, f"({ascii_lowercase[idx]})", transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL, fontweight="bold", va="top", ha="left")
    fig.tight_layout()
    fig.savefig(case_dir / "total_raw.png", dpi=900)
    plt.close(fig)


def plot_relative_convergence_raw(
    case_dir: Path, pinn_df: pd.DataFrame, ps_df: pd.DataFrame
) -> None:
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if f"{mode}_total" in pinn_df.columns
        and f"{mode}_total" in ps_df.columns
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 4.0))
    if len(modes) == 1:
        axes = [axes]
    for idx, (ax, mode) in enumerate(zip(axes, modes)):
        col = f"{mode}_total"
        series_list = []
        # PINN
        epoch_pinn = safe_epoch(pinn_df["epoch"].to_numpy())
        values_pinn = pinn_df[col].to_numpy()
        ref_pinn = np.min(values_pinn[int(len(values_pinn) * 0.9) :])
        diff_pinn = np.abs(values_pinn - ref_pinn) + 1e-12
        ax.plot(epoch_pinn, diff_pinn, linestyle=LINESTYLE_PINN,
                color=COLOR_PINN, linewidth=LINEWIDTH, label="PINN")
        series_list.append(("PINN", epoch_pinn, diff_pinn))
        # PS-PINN
        epoch_ps = safe_epoch(ps_df["epoch"].to_numpy())
        values_ps = ps_df[col].to_numpy()
        ref_ps = np.min(values_ps[int(len(values_ps) * 0.9) :])
        diff_ps = np.abs(values_ps - ref_ps) + 1e-12
        ax.plot(epoch_ps, diff_ps, linestyle=LINESTYLE_PS_PINN,
                color=COLOR_PS_PINN, linewidth=LINEWIDTH, label="PS-PINN")
        series_list.append(("PS-PINN", epoch_ps, diff_ps))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("|Total - Total$_{ref}$|", fontsize=FONTSIZE_LABEL)
        mode_title = "Linear" if mode == "linear" else "Nonlinear"
        ax.set_title(f"{mode_title} Relative Convergence", fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
        # Subplot label
        ax.text(0.03, 0.95, f"({ascii_lowercase[idx]})", transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL, fontweight="bold", va="top", ha="left")

        # Inset figure - early stage
        inset_early = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        # Inset figure - late stage
        inset_late = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )

        early_count = max(50, int(len(series_list[0][1]) * 0.1))
        late_count = max(50, int(len(series_list[0][1]) * 0.1))

        early_min = np.inf
        early_max = -np.inf
        late_min = np.inf
        late_max = -np.inf

        colors = [COLOR_PINN, COLOR_PS_PINN]
        linestyles = [LINESTYLE_PINN, LINESTYLE_PS_PINN]
        for i, (label, epoch, diff) in enumerate(series_list):
            inset_early.plot(epoch[:early_count], diff[:early_count],
                           linestyle=linestyles[i], color=colors[i],
                           linewidth=LINEWIDTH * 0.8, label=label)
            inset_late.plot(epoch[-late_count:], diff[-late_count:],
                          linestyle=linestyles[i], color=colors[i],
                          linewidth=LINEWIDTH * 0.8, label=label)
            early_min = min(early_min, float(np.min(diff[:early_count])))
            early_max = max(early_max, float(np.max(diff[:early_count])))
            late_min = min(late_min, float(np.min(diff[-late_count:])))
            late_max = max(late_max, float(np.max(diff[-late_count:])))

        inset_early.set_xscale("log")
        inset_early.set_yscale("log")
        inset_early.set_title("Early", fontsize=8)
        inset_early.set_xticklabels([])
        inset_early.set_yticklabels([])
        inset_early.set_ylim(early_min * 0.8, early_max * 1.2)
        inset_early.tick_params(direction="in")

        inset_late.set_xscale("log")
        inset_late.set_yscale("log")
        inset_late.set_title("Late", fontsize=8)
        inset_late.set_xticklabels([])
        inset_late.set_yticklabels([])
        inset_late.set_ylim(late_min * 0.8, late_max * 1.2)
        inset_late.tick_params(direction="in")

        ax.annotate(
            "",
            xy=(1.02, 0.85),
            xycoords="axes fraction",
            xytext=(0.72, 0.85),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=0.8),
            clip_on=False,
        )
        ax.annotate(
            "",
            xy=(1.02, 0.15),
            xycoords="axes fraction",
            xytext=(0.72, 0.15),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=0.8),
            clip_on=False,
        )
    fig.tight_layout()
    fig.savefig(case_dir / "analysis_relative_convergence.png", dpi=900)
    plt.close(fig)


def plot_pi_all_raw(case_dir: Path, pinn_df: pd.DataFrame, ps_df: pd.DataFrame) -> None:
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if f"{mode}_Pi_all" in pinn_df.columns
        and f"{mode}_Pi_all" in ps_df.columns
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.0 * len(modes), 4.0))
    if len(modes) == 1:
        axes = [axes]
    for idx, (ax, mode) in enumerate(zip(axes, modes)):
        col = f"{mode}_Pi_all"
        # PINN - blue dashed line
        epoch_pinn = safe_epoch(pinn_df["epoch"].to_numpy())
        values_pinn = pinn_df[col].to_numpy()
        ax.plot(epoch_pinn, values_pinn, linestyle=LINESTYLE_PINN,
                color=COLOR_PINN, linewidth=LINEWIDTH, label="PINN")
        # PS-PINN - orange solid line
        epoch_ps = safe_epoch(ps_df["epoch"].to_numpy())
        values_ps = ps_df[col].to_numpy()
        ax.plot(epoch_ps, values_ps, linestyle=LINESTYLE_PS_PINN,
                color=COLOR_PS_PINN, linewidth=LINEWIDTH, label="PS-PINN")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Epoch", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(r"$\Pi_{all}$", fontsize=FONTSIZE_LABEL)
        mode_title = "Linear" if mode == "linear" else "Nonlinear"
        ax.set_title(f"{mode_title} " + r"$\Pi_{all}$", fontsize=FONTSIZE_TITLE)
        ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
        # Subplot label
        ax.text(0.03, 0.95, f"({ascii_lowercase[idx]})", transform=ax.transAxes,
                fontsize=FONTSIZE_PANEL, fontweight="bold", va="top", ha="left")
    fig.tight_layout()
    fig.savefig(case_dir / "pi_all_raw.png", dpi=900)
    plt.close(fig)


def plot_components_raw(case_dir: Path, pinn_df: pd.DataFrame, ps_df: pd.DataFrame) -> None:
    """
    Plot the energy component comparison figure (2x4 layout)
    Following the style of image 1.png:
    - Top row: Linear Energy
    - Bottom row: Nonlinear Energy
    - Four columns: Pi_all, Pi_str, Pi_w, -Pi_e
    """
    layout = [
        ("Pi_all", 1.0, r"$\Pi_{beam}$"),
        ("Pi_str", 1.0, r"$\Pi_{str}$"),
        ("Pi_w", 1.0, r"$\Pi_{w}$"),
        ("Pi_e", -1.0, r"$-\Pi_{e}$"),
    ]
    required = all(
        f"{mode}_{name}" in pinn_df.columns and f"{mode}_{name}" in ps_df.columns
        for mode in ("linear", "nonlinear")
        for name, _, _ in layout
    )
    if not required:
        return

    component_title_size = FONTSIZE_TITLE + 2
    component_label_size = FONTSIZE_LABEL + 2
    component_tick_size = FONTSIZE_TICK + 2
    component_legend_size = FONTSIZE_LEGEND + 2
    component_panel_size = FONTSIZE_PANEL + 2

    # Create 2x4 subplots, following the layout of image 1.png
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.0))

    panel_idx = 0
    for row, mode in enumerate(["linear", "nonlinear"]):
        for col_index, (name, sign, title) in enumerate(layout):
            ax = axes[row, col_index]
            col = f"{mode}_{name}"

            # Get the data
            pinn_epoch = safe_epoch(pinn_df["epoch"].to_numpy())
            pinn_values = pinn_df[col].to_numpy() * sign
            ps_epoch = safe_epoch(ps_df["epoch"].to_numpy())
            ps_values = ps_df[col].to_numpy() * sign

            if not (is_significant(pinn_values) or is_significant(ps_values)):
                ax.set_visible(False)
                panel_idx += 1
                continue

            # Plot the curves - PINN blue dashed line, PS-PINN orange solid line
            ax.plot(pinn_epoch, pinn_values, linestyle=LINESTYLE_PINN,
                    color=COLOR_PINN, linewidth=LINEWIDTH, label="PINN")
            ax.plot(ps_epoch, ps_values, linestyle=LINESTYLE_PS_PINN,
                    color=COLOR_PS_PINN, linewidth=LINEWIDTH, label="PS-PINN")

            # Set the Y-axis range
            y_min = min(float(np.min(pinn_values)), float(np.min(ps_values)))
            y_max = max(float(np.max(pinn_values)), float(np.max(ps_values)))
            y_pad = max((y_max - y_min) * 0.08, 1e-10)
            y_lower = y_min - y_pad
            y_upper = y_max + y_pad
            if col_index == 1:  # Pi_str
                y_lower = 10 ** (-4.5)
                y_upper = 1.0
            if col_index == 2:  # Pi_w
                y_lower = -1e-7
            if row == 1 and col_index == 3:  # nonlinear -Pi_e
                y_upper = -(10 ** (-4.5))
            if y_lower >= y_upper:
                y_lower = y_upper - max(abs(y_upper) * 0.1, 1e-6)
            ax.set_ylim(y_lower, y_upper)

            # Logarithmic axes
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-6)
            apply_component_custom_y_ticks(ax, row=row, col_index=col_index)
            ax.tick_params(axis="both", labelsize=component_tick_size)

            # X-axis label: shown only on the second row
            if row == 1:
                ax.set_xlabel("Epoch", fontsize=component_label_size)

            # Y-axis label: shown only on the first column
            if col_index == 0:
                mode_label = "Linear Energy" if mode == "linear" else "Nonlinear Energy"
                ax.set_ylabel(mode_label, fontsize=component_label_size)

            # Column title: show the math symbol only on the first row
            if row == 0:
                ax.set_title(title, fontsize=component_title_size)

            # Legend position: set a different position based on the column index, no background, flush with the axes
            # col 0 (a, e): flush bottom-left
            # col 1 (b, f): flush top-right
            # col 2 (c, g): flush bottom-right
            # col 3 (d, h): flush top-right
            # Use bbox_to_anchor for precise positioning, borderaxespad=0 to remove the margin
            if col_index == 0:  # bottom-left
                ax.legend(fontsize=component_legend_size, frameon=False,
                         loc='lower left', bbox_to_anchor=(0, 0), borderaxespad=0)
            elif col_index == 2:  # bottom-right
                ax.legend(fontsize=component_legend_size, frameon=False,
                         loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0)
            else:  # col 1, 3: top-right
                ax.legend(fontsize=component_legend_size, frameon=False,
                         loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)

            # Subplot labels (a), (b), ... in the top-left corner
            ax.text(
                0.03,
                0.95,
                f"({ascii_lowercase[panel_idx]})",
                transform=ax.transAxes,
                fontsize=component_panel_size,
                fontweight="bold",
                va="top",
                ha="left",
            )
            panel_idx += 1

    fig.tight_layout()
    fig.savefig(case_dir / "energy_components_raw.png", dpi=900)
    fig.savefig(case_dir / "energy_components_raw.svg", format="svg")
    fig.savefig(case_dir / "energy_components_raw.pdf", format="pdf")
    fig.savefig(case_dir / "energy_components_raw.eps", format="eps")
    plt.close(fig)


def safe_epoch(epoch: np.ndarray) -> np.ndarray:
    epoch = epoch.astype(float)
    epoch[epoch <= 0] = 1.0
    return epoch


def is_significant(series, tol: float = 1e-12) -> bool:
    values = np.asarray(series)
    return float(np.nanmax(np.abs(values))) > tol


def run_case(case_dir: Path) -> None:
    pinn_df = load_loss_csv(case_dir / "PINN")
    ps_df = load_loss_csv(case_dir / "PS-PINN")

    pinn_df = ensure_epoch(pinn_df)
    ps_df = ensure_epoch(ps_df)

    plot_pi_all_raw(case_dir, pinn_df, ps_df)
    plot_total_raw(case_dir, pinn_df, ps_df)
    plot_relative_convergence_raw(case_dir, pinn_df, ps_df)
    plot_components_raw(case_dir, pinn_df, ps_df)


__all__ = ["run_case"]
