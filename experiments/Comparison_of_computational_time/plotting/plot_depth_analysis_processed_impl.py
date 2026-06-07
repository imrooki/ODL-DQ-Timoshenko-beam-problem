from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

matplotlib.use("Agg")


@dataclass(frozen=True)
class ProcessConfig:
    tail_window: int = 500
    rolling_window: int = 200
    min_eps: float = 1e-12


def run_case(case_dir: Path) -> None:
    config = ProcessConfig()
    pinn_df = load_loss_csv(case_dir / "PINN")
    ps_df = load_loss_csv(case_dir / "PS-PINN")

    pinn_df = ensure_epoch(pinn_df)
    ps_df = ensure_epoch(ps_df)

    plot_pi_all_rolling(case_dir, pinn_df, ps_df, config)
    plot_relative_convergence_processed(case_dir, pinn_df, ps_df, config)
    plot_ebest(case_dir, pinn_df, ps_df, config)
    plot_components_rolling(case_dir, pinn_df, ps_df, config)
    plot_components_processed(case_dir, pinn_df, ps_df, config)


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


def plot_pi_all_rolling(
    case_dir: Path,
    pinn_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    config: ProcessConfig,
) -> None:
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if f"{mode}_Pi_all" in pinn_df.columns
        and f"{mode}_Pi_all" in ps_df.columns
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.2))
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        col = f"{mode}_Pi_all"
        for label, df in (("PINN", pinn_df), ("PS-PINN", ps_df)):
            epoch = safe_epoch(df["epoch"].to_numpy())
            values = df[col].to_numpy()
            smooth = rolling_mean(values, config.rolling_window)
            ax.plot(epoch, smooth, label=label)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pi_all (rolling mean)")
        ax.set_title(f"{mode} Pi_all processed")
        ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "pi_all_processed.png", dpi=300)
    plt.close(fig)


def plot_relative_convergence_processed(
    case_dir: Path,
    pinn_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    config: ProcessConfig,
) -> None:
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if f"{mode}_total" in pinn_df.columns
        and f"{mode}_total" in ps_df.columns
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.2))
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        col = f"{mode}_total"
        for label, df in (("PINN", pinn_df), ("PS-PINN", ps_df)):
            epoch = safe_epoch(df["epoch"].to_numpy())
            values = df[col].to_numpy()
            smooth = rolling_mean(values, config.rolling_window)
            ref = np.min(smooth[int(len(smooth) * 0.9) :])
            diff = np.abs(smooth - ref) + 1e-12
            ax.plot(epoch, diff, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("|Total - Total_ref|")
        ax.set_title(f"{mode} relative convergence")
        ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "analysis_relative_convergence_processed.png", dpi=300)
    plt.close(fig)


def plot_ebest(
    case_dir: Path,
    pinn_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    config: ProcessConfig,
) -> None:
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if f"{mode}_Pi_all" in pinn_df.columns
        and f"{mode}_Pi_all" in ps_df.columns
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.2))
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        col = f"{mode}_Pi_all"
        for label, df in (("PINN", pinn_df), ("PS-PINN", ps_df)):
            epoch = safe_epoch(df["epoch"].to_numpy())
            values = df[col].to_numpy()
            platform = platform_value(values, config.tail_window)
            ebest = compute_ebest(values, platform, config.min_eps)
            ax.plot(epoch, ebest, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("|Pi_all - Pi*| (best-so-far)")
        ax.set_title(f"{mode} e_best processed")
        ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "ebest_processed.png", dpi=300)
    plt.close(fig)


def plot_components_rolling(
    case_dir: Path,
    pinn_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    config: ProcessConfig,
) -> None:
    components = [
        ("Pi_str", 1.0),
        ("Pi_str_T", 1.0),
        ("Pi_w", 1.0),
        ("Pi_all", 1.0),
        ("Pi_e", -1.0),
    ]
    modes = [
        mode
        for mode in ("linear", "nonlinear")
        if any(
            f"{mode}_{name}" in pinn_df.columns and f"{mode}_{name}" in ps_df.columns
            for name, _ in components
        )
    ]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(7.2 * len(modes), 4.6))
    if len(modes) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, mode in zip(axes, modes):
        color_index = 0
        for name, sign in components:
            col = f"{mode}_{name}"
            if col not in pinn_df.columns or col not in ps_df.columns:
                continue
            if not (
                is_significant(pinn_df[col].to_numpy())
                or is_significant(ps_df[col].to_numpy())
            ):
                continue
            color = cmap(color_index % 10)
            color_index += 1
            for label, df, style in (
                ("PINN", pinn_df, "--"),
                ("PS-PINN", ps_df, "-"),
            ):
                epoch = safe_epoch(df["epoch"].to_numpy())
                values = df[col].to_numpy() * sign
                smooth = rolling_mean(values, config.rolling_window)
                display_name = f"{label}-{name if sign > 0 else f'-{name}'}"
                ax.plot(epoch, smooth, linestyle=style, color=color, label=display_name)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy (rolling mean)")
        ax.set_title(f"{mode} components processed")
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(case_dir / "energy_components_processed.png", dpi=300)
    plt.close(fig)


def plot_components_processed(
    case_dir: Path,
    pinn_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    config: ProcessConfig,
) -> None:
    layout = [
        ("Pi_all", 1.0, r"$\Pi_{all}$"),
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
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 7.5), sharex="col")
    panel_idx = 0
    for row, mode in enumerate(["linear", "nonlinear"]):
        for col_index, (name, sign, title) in enumerate(layout):
            ax = axes[row, col_index]
            col = f"{mode}_{name}"
            pinn_values = rolling_mean(pinn_df[col].to_numpy() * sign, config.rolling_window)
            ps_values = rolling_mean(ps_df[col].to_numpy() * sign, config.rolling_window)
            if not (is_significant(pinn_values) or is_significant(ps_values)):
                ax.set_visible(False)
                panel_idx += 1
                continue
            for label, values, style in (
                ("PINN", pinn_values, "--"),
                ("PS-PINN", ps_values, "-"),
            ):
                epoch = safe_epoch(pinn_df["epoch"].to_numpy())
                ax.plot(epoch, values, linestyle=style, label=label)
            y_min = min(float(np.min(pinn_values)), float(np.min(ps_values)))
            y_max = max(float(np.max(pinn_values)), float(np.max(ps_values)))
            y_pad = max((y_max - y_min) * 0.08, 1e-10)
            y_lower = y_min - y_pad
            y_upper = y_max + y_pad
            if col_index == 1:
                y_lower = 10 ** (-4.5)
                y_upper = 1.0
            if col_index == 2:
                y_lower = -1e-7
            if row == 1 and col_index == 3:
                y_upper = -(10 ** (-4.5))
            if y_lower >= y_upper:
                y_lower = y_upper - max(abs(y_upper) * 0.1, 1e-6)
            ax.set_ylim(y_lower, y_upper)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-6)
            if row == 1:
                ax.set_xlabel("Epoch")
            if col_index == 0:
                ax.set_ylabel(f"{mode} Energy")
            title_text = title if row == 0 else name
            ax.set_title(title_text)
            ax.legend(fontsize=8)
            ax.text(
                0.02,
                0.95,
                f"({ascii_lowercase[panel_idx]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
            panel_idx += 1
    fig.tight_layout()
    fig.savefig(case_dir / "energy_components_raw.png", dpi=300)
    plt.close(fig)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(pd.Series(values).rolling(window, min_periods=1).mean())


def platform_value(values: np.ndarray, tail_window: int) -> float:
    tail = values[-min(len(values), tail_window) :]
    return float(np.mean(tail))


def compute_ebest(values: np.ndarray, platform: float, min_eps: float) -> np.ndarray:
    errors = np.abs(values - platform)
    return np.minimum.accumulate(np.maximum(errors, min_eps))


def safe_epoch(epoch: np.ndarray) -> np.ndarray:
    epoch = epoch.astype(float)
    epoch[epoch <= 0] = 1.0
    return epoch


def is_significant(series, tol: float = 1e-12) -> bool:
    values = np.asarray(series)
    return float(np.nanmax(np.abs(values))) > tol


__all__ = ["run_case"]
