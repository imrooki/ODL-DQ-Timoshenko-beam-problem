"""
Experiment orchestration helper (experiment)

Responsibilities:
- Build the OutputManager and file prefix from parameters and material parameters.
- Provide helper functions for result summaries and index updates, reducing the
  burden on main.py.
"""

from __future__ import annotations

from typing import Dict, Optional
import os
import numpy as np

# Support running the script directly: fall back to absolute import if the relative import fails
try:
    from .output_manager import OutputManager
except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from utils.output_manager import OutputManager


def build_output_manager(params, material_params) -> tuple[OutputManager, str]:
    """Build the OutputManager and the standard filename prefix."""

    script_name = os.path.splitext(os.path.basename(params.__file__ if hasattr(params, "__file__") else "main"))[0]
    # If executed through main, use the running script name
    if script_name == "params":
        # When run from main.py, params.__file__ may be params.py; here we use "main" as the script name
        script_name = "main"

    param_folder = OutputManager.make_param_folder(
        params.W_Gr, params.T, params.H_Gr, params.q, material_params["lambda_val"]
    )
    om = OutputManager(
        base_dir="results",
        script_name=script_name,
        bc_type=params.bc_type,
        distribution=params.distribution,
        param_folder=param_folder,
    )
    prefix = om.generate_filename(params.W_Gr, params.T, params.H_Gr, params.q)
    return om, prefix


def summarize_results(linear: Optional[Dict], nonlinear: Optional[Dict]) -> Dict:
    """Build the summary dictionary used by print_summary."""

    summary = {}
    if linear is not None:
        summary["linear"] = {
            "max_w": float(np.abs(linear[1]).max()),
            "final_loss": None,
        }
    if nonlinear is not None:
        summary["nonlinear"] = {
            "max_w": float(np.abs(nonlinear[1]).max()),
            "final_loss": None,
        }
    return summary


def update_index_via_manager(
    output_manager: OutputManager,
    prefix: str,
    params_dict: Dict,
    logs: Dict[str, Optional[np.ndarray]],
    results: Dict[str, Optional[tuple]],
    plots_enabled: bool,
) -> None:
    """Call OutputManager.update_index to write the index."""

    import numpy as _np
    import hashlib

    def best_meta(log_arr):
        if log_arr is None or len(log_arr) == 0:
            return None
        idx = int(_np.argmin(log_arr[:, 1]))  # loss column
        def g(arr, i, j, default=_np.nan):
            try:
                return float(arr[i, j])
            except Exception:
                return float(default)
        meta = {
            "best_epoch": int(log_arr[idx, 0]),
            "best_loss": g(log_arr, idx, 1),
            "final_loss": g(log_arr, -1, 1),
        }
        # Extended columns if present: [epoch, total, Pi_all, bc, Pi_str, Pi_str_T, Pi_w, Pi_e, pseudo]
        if log_arr.shape[1] >= 3:
            meta.update({
                "best_Pi_all": g(log_arr, idx, 2),
                "final_Pi_all": g(log_arr, -1, 2),
            })
        if log_arr.shape[1] >= 4:
            meta.update({
                "best_bc": g(log_arr, idx, 3),
                "final_bc": g(log_arr, -1, 3),
            })
        if log_arr.shape[1] >= 5:
            meta.update({
                "best_Pi_str": g(log_arr, idx, 4),
                "final_Pi_str": g(log_arr, -1, 4),
            })
        if log_arr.shape[1] >= 6:
            meta.update({
                "best_Pi_str_T": g(log_arr, idx, 5),
                "final_Pi_str_T": g(log_arr, -1, 5),
            })
        if log_arr.shape[1] >= 7:
            meta.update({
                "best_Pi_w": g(log_arr, idx, 6),
                "final_Pi_w": g(log_arr, -1, 6),
            })
        if log_arr.shape[1] >= 8:
            meta.update({
                "best_Pi_e": g(log_arr, idx, 7),
                "final_Pi_e": g(log_arr, -1, 7),
            })
        if log_arr.shape[1] >= 9:
            meta.update({
                "best_pseudo": g(log_arr, idx, 8),
                "final_pseudo": g(log_arr, -1, 8),
            })
        return meta

    def resolve_model_path(models_dir: str, full_prefix: str, short_prefix: str, filename: str) -> str:
        """Resolve model path consistently with utils/training_core.py shortening strategy."""
        max_path_limit = 255

        full_path = os.path.join(models_dir, f"{full_prefix}_{filename}.pth")
        if len(os.path.abspath(full_path)) <= max_path_limit:
            return full_path

        short_path = os.path.join(models_dir, f"{short_prefix}_{filename}.pth")
        if len(os.path.abspath(short_path)) <= max_path_limit:
            return short_path

        hash_str = hashlib.md5(filename.encode()).hexdigest()[:12]
        return os.path.join(models_dir, f"{short_prefix}_{hash_str}.pth")

    linear_meta = best_meta(logs.get("linear"))
    nonlinear_meta = best_meta(logs.get("nonlinear"))
    if linear_meta is not None and results.get("linear") is not None:
        linear_meta.update(
            {
                "max_w": float(_np.abs(results["linear"][1]).max()),
                "model_path": resolve_model_path(output_manager.models_dir, "Linearw", "Lw", prefix),
            }
        )
    if nonlinear_meta is not None and results.get("nonlinear") is not None:
        nonlinear_meta.update(
            {
                "max_w": float(_np.abs(results["nonlinear"][1]).max()),
                "model_path": resolve_model_path(output_manager.models_dir, "Nonlinearw", "NLw", prefix),
            }
        )

    plots = {}
    if plots_enabled:
        plots = {
            "loss": os.path.join(output_manager.plots_dir, f"{prefix}_loss.png"),
            "u": os.path.join(output_manager.plots_dir, f"{prefix}_u.png"),
            "w": os.path.join(output_manager.plots_dir, f"{prefix}_w.png"),
            "phi": os.path.join(output_manager.plots_dir, f"{prefix}_phi.png"),
        }

    data_files = {
        "displacement_csv": os.path.join(output_manager.data_dir, f"w_{prefix}.csv"),
        "loss_csv": os.path.join(output_manager.logs_dir, f"loss_{prefix}.csv"),
        "plots": plots,
    }

    output_manager.update_index(
        prefix,
        params_dict,
        linear_meta=linear_meta,
        nonlinear_meta=nonlinear_meta,
        data_files=data_files,
    )


__all__ = ["build_output_manager", "summarize_results", "update_index_via_manager"]
