"""
PINN sweep runner — scenarios x optimizers.

Runs the energy-based Pure PINN training on the outer product

    scenarios x optimizers   =   2 x 2   =   4 runs

defined in `params_pinn.py`:

    Scenarios:
      - with_foundation : k1=0.01, k2=0.001 (paper default)
      - no_foundation   : k1=0,    k2=0

    Optimizers:
      - adam  : Adam, lr=8e-5, 100,000 epochs, FINAL model used
                (no patience, no best-state restore)
      - lbfgs : LBFGS, lr=1.0, max-iter=50,000 with patience=10,000
                early stop, BEST model used (best-loss state restored)

The two optimizer variants land in distinct top-level folders thanks
to upstream OutputManager respecting `script_name`:

    results/pinn/pure_pinn_adam/C-C/X/<param>/...
    results/pinn/pure_pinn_lbfgs/C-C/X/<param>/...

so `compare.py` can find them independently.

Usage (from this directory):
    conda activate claude_test
    python run_pinn.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path bootstrap: this script lives at <project>/experiments/Comparison_of_computational_time/pinn/
# Add this directory to sys.path so `utils.*`, `modules.*`, and
# `params_pure_pinn_legacy` resolve to the *copied* PINN code under Comparison_of_computational_time/pinn.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.training_core import run_training_core, validate_params_dict  # noqa: E402
from utils.common import safe_mkdir  # noqa: E402

import params_pinn  # noqa: E402  (local sweep config)


def _train_one_run(scen: dict, opt: dict, base_dir: str) -> dict:
    """Run one PINN training at the given (scenario, optimizer)."""
    params_dict = params_pinn.get_params_for_run(scen, opt)
    params_dict["scenario_name"] = scen["name"]
    params_dict["optimizer_label"] = opt["label"]

    print()
    print("#" * 76)
    print(f"# PINN  scenario={scen['name']}  optimizer={opt['name']}")
    print(f"#       k1={scen['k1']}, k2={scen['k2']}")
    print(f"#       optimizer_type={params_dict.get('optimizer_type')} "
          f"lr={params_dict.get('lr')} "
          f"epochs={params_dict.get('epochs')} "
          f"patience={params_dict.get('patience')} "
          f"restore_best={params_dict.get('restore_best')}")
    print("#" * 76)

    if not validate_params_dict(params_dict):
        return {"success": False, "error": "param validation failed",
                "scenario": scen["name"], "optimizer": opt["name"]}

    t0 = time.time()
    result = run_training_core(
        params_dict=params_dict,
        script_name=opt["script_name"],
        verbose=True,
        base_dir=base_dir,
    )
    elapsed = time.time() - t0

    if not isinstance(result, dict):
        result = {"success": False,
                  "error": f"unexpected return type: {type(result)!r}"}
    result["elapsed_time"] = elapsed
    result["scenario"] = scen["name"]
    result["k1"] = scen["k1"]
    result["k2"] = scen["k2"]
    result["optimizer"] = opt["name"]
    result["script_name"] = opt["script_name"]

    if result.get("skipped"):
        print(f"[SKIP] {scen['name']}/{opt['name']} already trained, skipping.")
    elif result.get("success"):
        print(f"[OK]   {scen['name']}/{opt['name']} done in {elapsed:.1f} s "
              f"({elapsed/60:.1f} min)")
    else:
        print(f"[FAIL] {scen['name']}/{opt['name']}: "
              f"{result.get('error', 'Unknown')}")

    # Free GPU memory between runs
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()

    return result


def _parse_cli():
    """Parse a small CLI: --only-optimizer adam|lbfgs to filter which
    runs are executed. Default: run everything in params_pinn.optimizers.
    """
    import argparse
    p = argparse.ArgumentParser(description="PINN sweep")
    p.add_argument("--only-optimizer", choices=["adam", "lbfgs"], default=None,
                   help="If set, only run this optimizer (preserve previous results "
                        "for the other variant).")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()

    print("=" * 76)
    print("PINN sweep (scenarios x optimizers)")
    if args.only_optimizer is not None:
        print(f"FILTER: --only-optimizer={args.only_optimizer}")
    print("=" * 76)
    params_pinn.print_sweep_summary()

    results_root = SCRIPT_DIR.parent / "results" / "pinn"
    safe_mkdir(str(results_root))
    print(f"\n[OUTPUT] PINN results -> {results_root}")

    # Apply filter
    optimizers_to_run = params_pinn.optimizers
    if args.only_optimizer is not None:
        optimizers_to_run = [o for o in params_pinn.optimizers
                             if o["name"] == args.only_optimizer]
        print(f"[FILTER] Running only optimizers: "
              f"{[o['name'] for o in optimizers_to_run]}")

    all_results = []
    total_t0 = time.time()
    for scen in params_pinn.scenarios:
        for opt in optimizers_to_run:
            all_results.append(_train_one_run(scen, opt, str(results_root)))

    total_elapsed = time.time() - total_t0

    index_path = results_root / "sweep_index.json"

    new_runs = [
        {
            "scenario": r.get("scenario"),
            "k1": r.get("k1"),
            "k2": r.get("k2"),
            "optimizer": r.get("optimizer"),
            "script_name": r.get("script_name"),
            "success": r.get("success", False),
            "skipped": r.get("skipped", False),
            "elapsed_s": r.get("elapsed_time"),
            "error": r.get("error"),
        }
        for r in all_results
    ]

    # Merge with existing index when filter is active (don't overwrite
    # successful runs of the other optimizer variant)
    if args.only_optimizer is not None and index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            old_runs = existing.get("runs", [])
            # Keep old runs that are NOT for the filtered optimizer
            kept = [r for r in old_runs
                    if r.get("optimizer") != args.only_optimizer]
            merged = kept + new_runs
            payload = {
                "total_elapsed_s": (existing.get("total_elapsed_s", 0.0)
                                    or 0.0) + total_elapsed,
                "runs": merged,
            }
            print(f"[INDEX] Merged: kept {len(kept)} old run(s) for the other "
                  f"optimizer + {len(new_runs)} new run(s).")
        except Exception as e:
            print(f"[WARN] Could not merge sweep_index.json ({e}); writing fresh.")
            payload = {"total_elapsed_s": total_elapsed, "runs": new_runs}
    else:
        payload = {"total_elapsed_s": total_elapsed, "runs": new_runs}

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print(f"PINN sweep finished in {total_elapsed:.1f} s "
          f"({total_elapsed/60:.1f} min)")
    print(f"Index: {index_path}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
