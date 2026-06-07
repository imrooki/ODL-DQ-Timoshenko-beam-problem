#!/usr/bin/env python3
"""
Pure PINN - Main training script

Author: Yang
Version: 1.0

Functionality:
Train a pure PINN model (no pseudo-supervision, no transfer learning)
- Use the MC integrator
- Disable pseudo-supervision
- Disable transfer learning
- Results are saved to Figs/Pure_PINN/results/

Usage:
    cd Figs/Pure_PINN
    python main_pure_pinn.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from utils.training_core import run_training_core, validate_params_dict
from utils.common import safe_mkdir
from params_pure_pinn_legacy import get_params, print_config_summary


def remove_nul_file(project_root: Path) -> None:
    """Remove 'nul' file if exists (Windows issue)"""
    nul_path = project_root / "nul"
    if nul_path.exists() and nul_path.is_file():
        try:
            nul_path.unlink()
            print("[INFO] Removed 'nul' file")
        except (PermissionError, OSError) as exc:
            print(f"[WARN] Could not remove 'nul' file: {exc}")


def train_pure_pinn(params_dict: dict, base_dir: str) -> dict:
    """
    Train a pure PINN model

    Args:
        params_dict: Parameter dictionary
        base_dir: Output base directory

    Returns:
        Training result dictionary
    """
    print(f"\n{'='*70}")
    print(f"Training Pure PINN (No Pseudo-Supervision, No Transfer Learning)")
    print(f"{'='*70}")

    start_time = time.time()

    # Validate parameters
    if not validate_params_dict(params_dict):
        print(f"[ERROR] Parameter validation failed")
        return {'success': False, 'error': 'Validation failed'}

    # Print key configuration
    use_ps = params_dict.get('use_pseudo_supervision', False)
    tf_freq = params_dict.get('transfer_freq', 0)
    integrator = params_dict.get('integrator', 'mc')
    print(f"  Pseudo-Supervision: {'Enabled' if use_ps else 'DISABLED'}")
    print(f"  Transfer Learning: {'Enabled' if tf_freq > 0 else 'DISABLED'}")
    print(f"  Integrator: {integrator.upper()}")

    # Call the training core
    result = run_training_core(
        params_dict=params_dict,
        script_name='pure_pinn',
        verbose=True,
        base_dir=base_dir
    )

    elapsed_time = time.time() - start_time
    result['elapsed_time'] = elapsed_time
    result['method'] = 'Pure PINN'

    if result['success']:
        print(f"\n[SUCCESS] Pure PINN training completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    else:
        print(f"\n[ERROR] Pure PINN training failed: {result.get('error', 'Unknown')}")

    return result


def load_training_results(results_dir: Path) -> dict:
    """Load training results"""
    method_dir = results_dir / "pure_pinn"

    if not method_dir.exists():
        return None

    # Find the parameter folder
    bc_dirs = list(method_dir.glob("*"))
    if not bc_dirs:
        return None

    for bc_dir in bc_dirs:
        dist_dirs = list(bc_dir.glob("*"))
        for dist_dir in dist_dirs:
            param_dirs = list(dist_dir.glob("W*"))
            if param_dirs:
                param_dir = param_dirs[0]
                result = {'method': 'Pure PINN', 'path': str(param_dir)}

                # Load displacement data
                data_files = list((param_dir / "data").glob("w_*.csv"))
                if data_files:
                    df = pd.read_csv(data_files[0])
                    result['x'] = df['x'].values
                    result['linear_w'] = df.get('linear_w', df.get('w_linear', pd.Series())).values
                    result['nonlinear_w'] = df.get('nonlinear_w', df.get('w_nonlinear', pd.Series())).values
                    result['max_linear_w'] = float(np.abs(result['linear_w']).max()) if len(result['linear_w']) else 0
                    result['max_nonlinear_w'] = float(np.abs(result['nonlinear_w']).max()) if len(result['nonlinear_w']) else 0

                # Load loss data
                log_files = list((param_dir / "logs").glob("loss_*.csv"))
                if log_files:
                    loss_df = pd.read_csv(log_files[0])
                    if 'linear_total' in loss_df.columns:
                        result['linear_loss'] = loss_df['linear_total'].values
                        result['final_linear_loss'] = float(loss_df['linear_total'].iloc[-1])
                    if 'nonlinear_total' in loss_df.columns:
                        result['nonlinear_loss'] = loss_df['nonlinear_total'].values
                        result['final_nonlinear_loss'] = float(loss_df['nonlinear_total'].iloc[-1])
                    result['epochs'] = loss_df['epoch'].values

                return result
    return None


def generate_report(result: dict, data: dict, output_dir: Path):
    """Generate the training report"""
    report_path = output_dir / "training_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Pure PINN Training Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("[Configuration]\n")
        f.write("-" * 40 + "\n")
        f.write("Method: Pure PINN\n")
        f.write("  - Pseudo-Supervision: DISABLED\n")
        f.write("  - Transfer Learning: DISABLED\n")
        f.write("  - Integrator: MC (Monte Carlo)\n\n")

        f.write("[Training Time]\n")
        f.write("-" * 40 + "\n")
        t = result.get('elapsed_time', 0)
        f.write(f"Total Time: {t:.1f}s ({t/60:.1f} min)\n\n")

        if data:
            f.write("[Results]\n")
            f.write("-" * 40 + "\n")
            f.write(f"Linear Model:\n")
            f.write(f"  Max |w|: {data.get('max_linear_w', 0):.6f}\n")
            f.write(f"  Final Loss: {data.get('final_linear_loss', float('nan')):.6e}\n")
            f.write(f"Nonlinear Model:\n")
            f.write(f"  Max |w|: {data.get('max_nonlinear_w', 0):.6f}\n")
            f.write(f"  Final Loss: {data.get('final_nonlinear_loss', float('nan')):.6e}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"[REPORT] Saved: {report_path}")
    return report_path


def main():
    """Main function: train a pure PINN model"""
    # Clean up the nul file
    remove_nul_file(PROJECT_ROOT)

    print("\n" + "=" * 70)
    print("Pure PINN Training")
    print("=" * 70)

    # Print the configuration summary
    print_config_summary()

    # Set up the output directory
    results_dir = SCRIPT_DIR / "results"
    safe_mkdir(results_dir)

    print(f"\n[OUTPUT] Results will be saved to: {results_dir}")

    total_start = time.time()

    # ===================== Train the pure PINN =====================
    print("\n" + "#" * 70)
    print("# Pure PINN Training (No Pseudo-Supervision, No Transfer Learning)")
    print("#" * 70)

    params = get_params()
    result = train_pure_pinn(params, str(results_dir))

    # ===================== Generate the report =====================
    print("\n" + "#" * 70)
    print("# Generating Training Report")
    print("#" * 70)

    # Load results
    data = load_training_results(results_dir)

    # Generate the report
    report_dir = results_dir / "summary"
    safe_mkdir(report_dir)
    generate_report(result, data, report_dir)

    # ===================== Summary =====================
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nPure PINN: {'SUCCESS' if result.get('success') else 'FAILED'}")
    print(f"  Training Time: {result.get('elapsed_time', 0):.1f}s")

    if data:
        print(f"  Max Linear |w|: {data.get('max_linear_w', 0):.6f}")
        print(f"  Max Nonlinear |w|: {data.get('max_nonlinear_w', 0):.6f}")

    print(f"\n[OUTPUT] Results saved to: {results_dir}")
    print(f"  - Model & Data: {results_dir}/pure_pinn/...")
    print(f"  - Report: {results_dir}/summary/")
    print("=" * 70)


if __name__ == "__main__":
    main()
