"""
PINN parameter configuration (self-contained inside Comparison_of_computational_time).

Defines the *outer product* sweep
    scenarios x optimizers
that is consumed by `run_pinn.py`. The base PINN config is reused from
`params_pure_pinn_legacy.py`. Each (scenario, optimizer) call to `get_params_for_run()`
overrides the foundation parameters and the optimizer-specific knobs
on top of that base config.

This module does not import the project root params.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

# Legacy PINN base config.
import params_pure_pinn_legacy as _legacy  # noqa: E402


# ============================================================
# Foundation scenarios
# ============================================================
scenarios: List[Dict[str, Any]] = [
    {
        "name": "with_foundation",
        "k1": 0.01,
        "k2": 0.001,
        "description": "Winkler-Pasternak foundation (paper default)",
    },
    {
        "name": "no_foundation",
        "k1": 0.0,
        "k2": 0.0,
        "description": "No elastic foundation (isolated benchmark)",
    },
]


# ============================================================
# Optimizer variants
# ------------------------------------------------------------
# Two variants per scenario, total = 2 x 2 = 4 PINN runs.
#
# - Adam variant uses the upstream-tested baseline (lr=8e-5, 100k epochs)
#   and *does not* track best loss for the deliverable: the FINAL-epoch
#   model state is what gets reported and saved (`restore_best=False`).
#
# - LBFGS variant uses a much larger budget cap (50k epochs) but with
#   patience-based early stopping: if BOTH the linear and nonlinear
#   models fail to improve for `patience` consecutive epochs, training
#   stops. The deliverable is the BEST-loss model (`restore_best=True`).
#
# Each variant gets its own upstream `script_name` so OutputManager
# nests results separately under
#   results/pinn/<script_name>/C-C/X/<param-folder>/
# ============================================================
optimizers: List[Dict[str, Any]] = [
    {
        "name": "adam",
        "script_name": "pure_pinn_adam",
        "label": "PINN-Adam (final model)",
        "params": {
            "optimizer_type": "Adam",
            "lr": 8e-5,
            "epochs": 100000,
            "patience": None,         # disable early stopping
            "restore_best": False,    # use FINAL-epoch model
        },
    },
]


def get_params_for_run(scen: Dict[str, Any],
                       opt: Dict[str, Any]) -> Dict[str, Any]:
    """Build the params dict for one (scenario, optimizer) PINN run.

    Starts from the upstream legacy config, then overrides foundation
    `(k1, k2)` and the optimizer-specific keys defined in `opt['params']`.
    """
    params = deepcopy(_legacy.get_params())
    # Foundation override
    params["k1"] = float(scen["k1"])
    params["k2"] = float(scen["k2"])
    # Optimizer override (last write wins)
    for k, v in opt["params"].items():
        params[k] = v
    return params


def print_sweep_summary() -> None:
    print("=" * 76)
    print("PINN sweep matrix (scenarios x optimizers)")
    print("=" * 76)
    for s in scenarios:
        for o in optimizers:
            print(f"  - {s['name']:18s}  x  {o['label']:36s}  "
                  f"(k1={s['k1']}, k2={s['k2']}, opt={o['name']})")
    print("=" * 76)


if __name__ == "__main__":
    print_sweep_summary()
