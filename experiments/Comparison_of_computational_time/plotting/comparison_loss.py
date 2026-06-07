#!/usr/bin/env python3
"""
PINN vs PS-PINN loss curve comparison plotting script (enhanced version)

Functionality:
- Read the training loss logs from the PINN/ and PS-PINN/ subfolders in the current folder
- Compare the convergence curves of PINN and PS-PINN
- Support moving-average smoothing to reduce the visual interference of oscillations
- Professional academic-paper-style plotting

Author: Yang
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from pathlib import Path
from typing import Optional, Tuple

# ============================================================================
# Path configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.common import safe_mkdir
from utils.figs_plotting import add_subplot_labels

# ============================================================================
# Constant configuration
# ============================================================================
COLORS = {
    'PINN': '#E64B35',       # Scientific red
    'PS-PINN': '#4DBBD5',    # Scientific blue
    'PINN_light': '#E64B3540',
    'PS-PINN_light': '#4DBBD540',
}

# Moving-average window size
SMOOTH_WINDOW = 500

# Outlier filtering parameters (local outlier detection based on a sliding window)
OUTLIER_WINDOW = 100      # Local window size
OUTLIER_THRESHOLD = 3.0   # How many times above the local median is considered an outlier
FILTER_OUTLIERS = False   # Outlier filtering disabled by default


def setup_plot_style():
    """Set up the professional plotting style"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'figure.dpi': 150,
    })


# ============================================================================
# Data processing functions
# ============================================================================

def remove_outliers(data: np.ndarray, window: int = OUTLIER_WINDOW,
                    threshold: float = OUTLIER_THRESHOLD) -> np.ndarray:
    """Outlier filtering based on the local median of a sliding window

    Principle:
        For each data point, compute the median of its local window.
        If the value of that point exceeds the median by a factor of threshold,
        it is considered an outlier and replaced with the local median.

    Parameters:
        data: Original data array
        window: Sliding window size
        threshold: Outlier judgment threshold (how many times above the median is considered an outlier)

    Returns:
        The filtered data array
    """
    if len(data) < window:
        return data.copy()

    result = data.copy()
    half_win = window // 2

    for i in range(len(data)):
        # Compute the local window range
        start = max(0, i - half_win)
        end = min(len(data), i + half_win + 1)

        # Compute the local median
        local_median = np.median(result[start:end])

        # Outlier detection: if the current value exceeds the median by a factor of threshold
        if local_median > 0:
            if result[i] > local_median * threshold:
                result[i] = local_median

    return result


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute the moving average

    Parameters:
        data: Original data
        window: Window size

    Returns:
        The smoothed data (same length as the original data, boundaries filled with original values)
    """
    if len(data) < window:
        return data

    # Compute the moving average using convolution
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='valid')

    # Pad the boundaries to keep the data length unchanged
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left

    result = np.zeros_like(data)
    result[pad_left:len(data)-pad_right] = smoothed
    result[:pad_left] = data[:pad_left]
    result[len(data)-pad_right:] = data[len(data)-pad_right:]

    return result


def load_loss_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load the loss CSV file"""
    if not csv_path.exists():
        print(f"[WARN] Loss file does not exist: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded Loss data: {csv_path.name}, {len(df)} epochs")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None


def find_loss_csv_in_folder(folder: Path) -> Optional[Path]:
    """Find the loss CSV file in the folder"""
    csv_files = list(folder.glob("loss_*.csv"))
    if not csv_files:
        return None

    csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    if len(csv_files) > 1:
        print(f"[WARN] Multiple loss files found in {folder.name}, using the latest: {csv_files[0].name}")

    return csv_files[0]


# ============================================================================
# Plotting functions
# ============================================================================

def plot_single_loss(
    ax: plt.Axes,
    epochs: np.ndarray,
    loss: np.ndarray,
    color: str,
    color_light: str,
    label: str,
    smooth_window: int = SMOOTH_WINDOW,
    show_raw: bool = True,
    filter_outliers: bool = FILTER_OUTLIERS
) -> None:
    """Plot a single loss curve (with outlier filtering and smoothing)

    Parameters:
        ax: Axis object
        epochs: epoch array
        loss: loss array
        color: Main color
        color_light: Light color (used for the raw data)
        label: Legend label
        smooth_window: Smoothing window
        show_raw: Whether to show the raw data
        filter_outliers: Whether to filter outliers
    """
    # Filter out invalid values
    mask = np.isfinite(loss) & (loss > 0)
    if not np.any(mask):
        return

    epochs_valid = epochs[mask]
    loss_valid = loss[mask]

    # Outlier filtering (before smoothing)
    if filter_outliers and len(loss_valid) > OUTLIER_WINDOW:
        loss_filtered = remove_outliers(loss_valid, OUTLIER_WINDOW, OUTLIER_THRESHOLD)
    else:
        loss_filtered = loss_valid

    # Plot the raw data (semi-transparent, using the filtered data)
    if show_raw and len(loss_filtered) > smooth_window:
        ax.semilogy(epochs_valid, loss_filtered, '-', color=color,
                   alpha=0.15, linewidth=0.5, rasterized=True)

    # Compute and plot the smoothed curve
    if len(loss_filtered) > smooth_window:
        loss_smooth = moving_average(loss_filtered, smooth_window)
        ax.semilogy(epochs_valid, loss_smooth, '-', color=color,
                   linewidth=2.0, label=label)
    else:
        ax.semilogy(epochs_valid, loss_filtered, '-', color=color,
                   linewidth=1.5, label=label)


def plot_loss_comparison(
    pinn_data: Optional[pd.DataFrame],
    ps_pinn_data: Optional[pd.DataFrame],
    save_path: Path,
    smooth_window: int = SMOOTH_WINDOW,
    filter_outliers: bool = FILTER_OUTLIERS,
    dpi: int = 300,
    show_plot: bool = False
) -> None:
    """Plot the PINN vs PS-PINN loss comparison figure (enhanced version)"""
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    problem_configs = [
        ('linear_total', 'Linear Problem', axes[0]),
        ('nonlinear_total', 'Nonlinear Problem', axes[1]),
    ]

    for col_name, title, ax in problem_configs:
        has_data = False

        # Plot PINN
        if pinn_data is not None and col_name in pinn_data.columns:
            epochs = pinn_data['epoch'].values
            loss = pinn_data[col_name].values
            # Take the absolute value (the energy method may have negative values)
            loss_abs = np.abs(loss)
            plot_single_loss(ax, epochs, loss_abs, COLORS['PINN'],
                           COLORS['PINN_light'], 'PINN', smooth_window,
                           filter_outliers=filter_outliers)
            has_data = True

        # Plot PS-PINN
        if ps_pinn_data is not None and col_name in ps_pinn_data.columns:
            epochs = ps_pinn_data['epoch'].values
            loss = ps_pinn_data[col_name].values
            loss_abs = np.abs(loss)
            plot_single_loss(ax, epochs, loss_abs, COLORS['PS-PINN'],
                           COLORS['PS-PINN_light'], 'PS-PINN', smooth_window,
                           filter_outliers=filter_outliers)
            has_data = True

        # Set axis properties
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)

        # Set the grid
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)

        # Set the y-axis to logarithmic tick format
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))

        if has_data:
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

            # Add the final loss value annotation
            if pinn_data is not None and col_name in pinn_data.columns:
                final_loss = np.abs(pinn_data[col_name].iloc[-1])
                ax.axhline(y=final_loss, color=COLORS['PINN'], linestyle='--',
                          alpha=0.5, linewidth=1)

            if ps_pinn_data is not None and col_name in ps_pinn_data.columns:
                final_loss = np.abs(ps_pinn_data[col_name].iloc[-1])
                ax.axhline(y=final_loss, color=COLORS['PS-PINN'], linestyle='--',
                          alpha=0.5, linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='gray')

        # Adjust the margins
        ax.margins(x=0.02)

    add_subplot_labels(axes, fontsize=12)
    plt.tight_layout()

    safe_mkdir(save_path.parent)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"[OK] Image saved: {save_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def plot_loss_components(
    pinn_data: Optional[pd.DataFrame],
    ps_pinn_data: Optional[pd.DataFrame],
    save_path: Path,
    problem_type: str = "linear",
    smooth_window: int = SMOOTH_WINDOW,
    filter_outliers: bool = FILTER_OUTLIERS,
    dpi: int = 300,
    show_plot: bool = False
) -> None:
    """Plot the detailed loss component comparison figure (enhanced version)"""
    setup_plot_style()
    prefix = problem_type

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    components = [
        (f'{prefix}_total', 'Total Loss'),
        (f'{prefix}_Pi_all', r'Total Potential Energy ($\Pi_{all}$)'),
        (f'{prefix}_Pi_str', r'Strain Energy ($\Pi_{str}$)'),
        (f'{prefix}_Pi_str_T', r'Thermal Strain Energy ($\Pi_{str}^T$)'),
    ]

    for idx, (col_name, label) in enumerate(components):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        has_data = False

        # Plot PINN
        if pinn_data is not None and col_name in pinn_data.columns:
            epochs = pinn_data['epoch'].values
            loss = pinn_data[col_name].values
            loss_abs = np.abs(loss)
            mask = loss_abs > 1e-15
            if np.any(mask):
                plot_single_loss(ax, epochs, loss_abs, COLORS['PINN'],
                               COLORS['PINN_light'], 'PINN', smooth_window,
                               filter_outliers=filter_outliers)
                has_data = True

        # Plot PS-PINN
        if ps_pinn_data is not None and col_name in ps_pinn_data.columns:
            epochs = ps_pinn_data['epoch'].values
            loss = ps_pinn_data[col_name].values
            loss_abs = np.abs(loss)
            mask = loss_abs > 1e-15
            if np.any(mask):
                plot_single_loss(ax, epochs, loss_abs, COLORS['PS-PINN'],
                               COLORS['PS-PINN_light'], 'PS-PINN', smooth_window,
                               filter_outliers=filter_outliers)
                has_data = True

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)

        if has_data:
            ax.legend(loc='upper right', framealpha=0.9)
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')

    add_subplot_labels(axes, fontsize=12)
    plt.tight_layout()

    safe_mkdir(save_path.parent)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"[OK] Loss component plot saved: {save_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


# ============================================================================
# Main function
# ============================================================================

def main(show_plots: bool = False, smooth_window: int = SMOOTH_WINDOW,
         filter_outliers: bool = FILTER_OUTLIERS):
    """Main function"""
    print("=" * 60)
    print("PINN vs PS-PINN Loss Comparison Plotter (Enhanced)")
    print("=" * 60)
    print(f"[CONFIG] Smoothing window: {smooth_window}")
    print(f"[CONFIG] Outlier filtering: {'ON' if filter_outliers else 'OFF'}")
    if filter_outliers:
        print(f"         Outlier window: {OUTLIER_WINDOW}, threshold: {OUTLIER_THRESHOLD}x")

    base_dir = SCRIPT_DIR

    # 1. Load data
    print("\n[Step 1] Loading Loss data...")

    pinn_folder = base_dir / 'PINN'
    pinn_csv = find_loss_csv_in_folder(pinn_folder)
    pinn_data = load_loss_csv(pinn_csv) if pinn_csv else None

    ps_pinn_folder = base_dir / 'PS-PINN'
    ps_pinn_csv = find_loss_csv_in_folder(ps_pinn_folder)
    ps_pinn_data = load_loss_csv(ps_pinn_csv) if ps_pinn_csv else None

    # Check whether the data are identical
    if pinn_data is not None and ps_pinn_data is not None:
        if pinn_data.equals(ps_pinn_data):
            print("\n[WARNING] PINN and PS-PINN data are completely identical!")
            print("         Please check whether the wrong file was placed.")
            print("         Plotting will continue, but the two curves will overlap.\n")

    if pinn_data is None and ps_pinn_data is None:
        print("[ERROR] No Loss data files found!")
        return

    # 2. Plot the basic loss comparison figure
    print("\n[Step 2] Plotting Loss comparison...")
    save_path = base_dir / "comparison_loss.png"
    plot_loss_comparison(pinn_data, ps_pinn_data, save_path,
                        smooth_window=smooth_window,
                        filter_outliers=filter_outliers,
                        show_plot=show_plots)

    # 3. Plot the detailed loss component figures
    print("\n[Step 3] Plotting loss components...")

    comp_path_linear = base_dir / "loss_components_linear.png"
    plot_loss_components(pinn_data, ps_pinn_data, comp_path_linear,
                        "linear", smooth_window=smooth_window,
                        filter_outliers=filter_outliers,
                        show_plot=show_plots)

    comp_path_nonlinear = base_dir / "loss_components_nonlinear.png"
    plot_loss_components(pinn_data, ps_pinn_data, comp_path_nonlinear,
                        "nonlinear", smooth_window=smooth_window,
                        filter_outliers=filter_outliers,
                        show_plot=show_plots)

    print("\n" + "=" * 60)
    print("[SUCCESS] All Loss comparison plots completed!")
    print("=" * 60)

    print("\nGenerated files:")
    for f in base_dir.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    # Control the run behavior by modifying the CONFIG here (no command-line argument parsing provided)
    CONFIG = {
        "show_plots": False,
        "smooth_window": SMOOTH_WINDOW,
        "filter_outliers": FILTER_OUTLIERS,
    }

    main(
        show_plots=CONFIG["show_plots"],
        smooth_window=CONFIG["smooth_window"],
        filter_outliers=CONFIG["filter_outliers"],
    )
