#!/usr/bin/env python3
"""
Shared plotting helpers for figures under `Figs/`.

Goals:
- Keep consistent matplotlib style (Times New Roman, stix math).
- Add subplot labels like (a), (b), (c) ... in row-major order.
- Provide colorbar settings consistent with the reference figure in
  `Figs/Comparison_u_w_phi/k1=k2=0/comparison_linear_2d.png`.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt

try:
    from matplotlib.axes import Axes
except Exception:  # pragma: no cover
    Axes = Any  # type: ignore[misc,assignment]


def set_matplotlib_style(font_size: int = 11) -> None:
    """Apply a consistent matplotlib style for figures under `Figs/`."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = int(font_size)


def _index_to_letters(index: int) -> str:
    """0 -> a, 25 -> z, 26 -> aa ..."""
    index = int(index)
    if index < 0:
        raise ValueError("index must be non-negative")

    letters = ""
    n = index + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        letters = chr(ord("a") + r) + letters
    return letters


def _flatten_axes(axes: Any) -> List[Axes]:
    if axes is None:
        return []
    if isinstance(axes, Axes):
        return [axes]
    if hasattr(axes, "flat"):
        return [ax for ax in axes.flat if ax is not None]
    if isinstance(axes, (list, tuple)):
        out: List[Axes] = []
        for item in axes:
            out.extend(_flatten_axes(item))
        return out
    return []


def add_subplot_labels(
    axes: Any,
    *,
    start_index: int = 0,
    x: float = 0.02,
    y: float = 0.98,
    fontsize: int = 12,
    fontweight: str = "bold",
    color: str = "black",
    ha: str = "left",
    va: str = "top",
) -> None:
    """
    Add subplot labels like (a), (b), (c) ... to a collection of axes.

    The order is row-major for 2D axes arrays (matplotlib default).
    """
    axes_list = _flatten_axes(axes)
    for i, ax in enumerate(axes_list):
        tag = _index_to_letters(start_index + i)
        ax.text(
            x,
            y,
            f"({tag})",
            transform=ax.transAxes,
            fontsize=int(fontsize),
            fontweight=fontweight,
            color=color,
            ha=ha,
            va=va,
        )


def add_colorbar_like_reference(
    fig,
    mappable,
    ax,
    *,
    shrink: float = 0.6,
    pad: float = 0.02,
    tick_labelsize: int = 9,
    label: Optional[str] = None,
    label_fontsize: int = 11,
    **kwargs,
):
    """Create a colorbar with length and tick style similar to the reference figure."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=float(shrink), pad=float(pad), **kwargs)
    cbar.ax.tick_params(labelsize=int(tick_labelsize))
    if label is not None:
        cbar.set_label(str(label), fontsize=int(label_fontsize))
    return cbar


def set_zh_ticks_like_reference(
    ax: Axes,
    *,
    ylabel: Optional[str] = "z/h",
    ylabel_fontsize: int = 12,
    tick_labelsize: int = 10,
    ticks: Sequence[float] = (-1.0, -0.5, 0.0, 0.5),
) -> None:
    """
    Apply y-axis formatting for z/h-like plots.

    - Sets ylabel to "z/h" by default
    - Uses tick values similar to the reference figure when they fall inside current ylim
    """
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=int(ylabel_fontsize))
    ax.tick_params(axis="y", labelsize=int(tick_labelsize))

    y_min, y_max = ax.get_ylim()
    candidates = [float(t) for t in ticks]
    in_range = [t for t in candidates if (y_min - 1e-9) <= t <= (y_max + 1e-9)]
    if in_range:
        ax.set_yticks(in_range)
