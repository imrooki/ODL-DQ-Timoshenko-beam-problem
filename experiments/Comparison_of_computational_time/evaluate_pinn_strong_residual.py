"""
evaluate_pinn_strong_residual.py
=================================
Evaluate the strong-form PDE residual L2 norm of the 4 trained PINN-Adam models
on 13 DQ nodes, so that PINN and ODIL can be compared under the same metric
(||R_PDE||).

Usage (after conda activate claude_test):
    cd <Comparison_of_computational_time root directory>
    python evaluate_pinn_strong_residual.py

Output:
    Comparison_of_computational_time/results/pinn_vs_odil_strong_residual.csv
    Console prints a side-by-side comparison table: the PINN values vs the ODIL
    values, all evaluated under the same strong-form residual metric ||R_PDE||.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

# ============================================================================
# Path setup
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent      # = Comparison_of_computational_time root directory
EXP_ROOT = SCRIPT_DIR
PROJECT_ROOT = EXP_ROOT.parent.parent             # = FSDT_bending root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Upper-level modules (DQ + residuals + material + boundary mask)
from modules.dq_core import cheb_lobatto_nodes, weighting_coefficients_negsum  # noqa: E402
from modules.residuals import TimoshenkoBeamResiduals                            # noqa: E402
from modules.material_properties import compute_material_params_for_solver       # noqa: E402
from utils.tensor_ops import create_boundary_mask                                # noqa: E402


# ============================================================================
# Load PINN nets.py by path using importlib (to avoid a package-name conflict with the upper-level modules)
# ============================================================================
def _load_pinn_nets():
    nets_path = EXP_ROOT / "pinn" / "modules" / "nets.py"
    spec = importlib.util.spec_from_file_location("pinn_nets_mod", nets_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pinn_nets = _load_pinn_nets()


# ============================================================================
# Windows MAX_PATH compatibility
# ============================================================================
def _long_path(p: Path) -> str:
    s = str(p.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


# ============================================================================
# Physical constants (consistent with the Comparison_of_computational_time ODIL/PINN main pipeline)
# ============================================================================
H_BEAM = 0.1
L_BEAM = 20 * H_BEAM
NUM_LAYERS = 10
W_GR = 0.025
H_GR = 0.8
T_TEMP = 300.0
DISTRIBUTION_TYPE = "X"
Q_LOAD = -0.08
N_DQ = 13

# Paths of the 4 PINN models and their corresponding scenarios
PINN_BASE = EXP_ROOT / "results" / "pinn" / "pure_pinn_adam" / "C-C" / "X"
ODIL_BASE = EXP_ROOT / "results" / "odil"

SCENARIOS = {
    "with_foundation": {
        "k1": 0.01, "k2": 0.001,
        "pinn_model_dir": PINN_BASE / "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh-k0.01_0.001" / "models",
        "pinn_linear_glob": "Lw_*.pth",
        "pinn_nonlinear_glob": "NLw_*.pth",
        "odil_dir": ODIL_BASE / "with_foundation",
    },
    "no_foundation": {
        "k1": 0.0, "k2": 0.0,
        "pinn_model_dir": PINN_BASE / "W0.025-T300.0-H0.8-qn0.08-L20h-Tanh" / "models",
        "pinn_linear_glob": "Lw_*.pth",
        "pinn_nonlinear_glob": "NLw_*.pth",
        "odil_dir": ODIL_BASE / "no_foundation",
    },
}

ODIL_OPTIMIZERS = [
    ("levenberg-marquardt", "LM"),
    ("gauss-newton",        "GN"),
    ("lbfgs",               "L-BFGS"),
]


# ============================================================================
# PINN model loading and node prediction
# ============================================================================
def load_pinn_model(pth_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Construct the SharedEncoder + 3-head network identical to the one used during
    training and load the state_dict.
    Architecture: encoder=[1,32,64,128], heads=[128,64,32,1], activation Tanh, in_dim=1.
    The saved state_dict comes from EnergyPINNStatic.state_dict() (which includes
    the 'net.' prefix); this script uses only the net part, so the 'net.' prefix is
    stripped before loading.
    """
    model = pinn_nets.build_timoshenko_net(
        in_dim=1,
        activation_type="Tanh",
        encoder_dims_shared=[1, 32, 64, 128],
        head_dims=[128, 64, 32, 1],
    )
    state = torch.load(_long_path(pth_path), map_location=device, weights_only=True)
    # Strip the 'net.' prefix to match the pure net structure returned by build_timoshenko_net
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    if not cleaned:
        # Fault tolerance: if the original state has no prefix, use it directly
        cleaned = state
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model


def load_odil_displacement(scen_dir: Path, opt_name: str, mode: str,
                            device: torch.device, dtype: torch.dtype) -> tuple:
    """
    Load the displacement field from Comparison_of_computational_time/results/odil/<scenario>/<optimizer>/data/displacement.csv.
    CSV columns: x, linear_u, linear_w, linear_phi, nonlinear_u, nonlinear_w, nonlinear_phi
    Returns (u, w, phi), shape=(N,), dtype=float64 on the device.
    Used only for the max|w| report (rebuttal sanity check), not for residual computation.
    """
    p = scen_dir / opt_name / "data" / "displacement.csv"
    if not p.exists():
        raise FileNotFoundError(f"No ODIL displacement.csv: {p}")
    df = pd.read_csv(_long_path(p))
    u = torch.tensor(df[f"{mode}_u"].values, device=device, dtype=dtype)
    w = torch.tensor(df[f"{mode}_w"].values, device=device, dtype=dtype)
    phi = torch.tensor(df[f"{mode}_phi"].values, device=device, dtype=dtype)
    return u, w, phi


def _read_summary_r_pde_norm(scen_dir: Path, opt_name: str, mode: str):
    """
    Best-effort read of the R_PDE_norm precomputed by run_odil.py from
    <scen>/<opt>/summary.json. Used only as a runtime consistency check against
    the value this script recomputes from displacement.csv -- it should match,
    because run_odil.py uses the identical TimoshenkoBeamResiduals + boundary
    mask + sqrt(sum of squares) formula. Returns None if the file or the
    summary[mode]["R_PDE_norm"] field is absent (the check is informational,
    never fatal).
    """
    p = scen_dir / opt_name / "summary.json"
    if not p.exists():
        return None
    with open(_long_path(p), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    block = data.get(mode)
    if not isinstance(block, dict):
        return None
    return block.get("R_PDE_norm")


def pinn_predict_at_nodes(model: torch.nn.Module, x: torch.Tensor) -> tuple:
    """
    Forward at the 13 DQ nodes and apply the C-C boundary condition poly lifting.
    poly lifting (vL=vR=0): v(x) = x*(1-x)*v_hat
    Returns (u, w, phi), shape=(N,), dtype consistent with x.

    The dtype of the model weights is determined by the state_dict of the .pth file
    (here it is float64 when saved). This script uniformly makes the input dtype
    follow the weight dtype to avoid Float/Double mixing errors.
    """
    # Detect the dtype of the current model weights (after load_state_dict it follows the state)
    weight_dtype = next(model.parameters()).dtype
    x_in = x.to(weight_dtype).unsqueeze(-1)  # (N, 1)
    with torch.no_grad():
        raw = model(x_in)  # (N, 3): raw_u, raw_w, raw_phi
    raw_u = raw[:, 0]
    raw_w = raw[:, 1]
    raw_phi = raw[:, 2]
    x1d = x_in.squeeze(-1)
    lift = x1d * (1.0 - x1d)  # x*(1-x)
    u = (lift * raw_u).to(x.dtype)
    w = (lift * raw_w).to(x.dtype)
    phi = (lift * raw_phi).to(x.dtype)
    return u, w, phi


# ============================================================================
# Main evaluation
# ============================================================================
def evaluate(device: torch.device) -> pd.DataFrame:
    # 1) Material parameters (dimensionless, shared)
    mat_dict = compute_material_params_for_solver(
        h=H_BEAM, L=L_BEAM, num_layers=NUM_LAYERS,
        W_Gr=W_GR, H_Gr=H_GR, T=T_TEMP,
        distribution_type=DISTRIBUTION_TYPE, q=Q_LOAD,
    )
    print(f"[INFO] material params: a11={mat_dict['a11']:.4f} b11={mat_dict['b11']:.4f} "
          f"d11={mat_dict['d11']:.4f} a55={mat_dict['a55']:.4f} "
          f"lambda={mat_dict['lambda_val']:.4f} n_xT={mat_dict['n_xT']:.6f}")

    # 2) DQ nodes + derivative matrices (consistent with the ODIL main pipeline: N=13, Cheb-Lobatto, negative-sum)
    x = cheb_lobatto_nodes(N_DQ, device=device)  # (13,) float64
    A, B, _C, _D = weighting_coefficients_negsum(x)
    mask = create_boundary_mask(N_DQ, device, x.dtype)

    # 3) Compute ||R_PDE||_2 under the same metric across all methods
    rows = []

    def _compute_residual_norm(u, w, phi, mode, residual_calc):
        if mode == "linear":
            R1, R2, R3 = residual_calc.compute_linear(u, w, phi, A, B)
        else:
            R1, R2, R3 = residual_calc.compute_nonlinear(u, w, phi, A, B)
        sq = (R1 * mask).pow(2).sum() + (R2 * mask).pow(2).sum() + (R3 * mask).pow(2).sum()
        return float(torch.sqrt(sq).item())

    for scen_name, scen in SCENARIOS.items():
        k1, k2 = scen["k1"], scen["k2"]
        residual_calc = TimoshenkoBeamResiduals(mat_dict, q=Q_LOAD, k1=k1, k2=k2)
        print(f"\n=== Scenario: {scen_name} (k1={k1}, k2={k2}) ===")

        # ---- (a) PINN-Adam: 2 .pth models ----
        pinn_dir = scen["pinn_model_dir"]
        linear_pths = sorted(pinn_dir.glob(scen["pinn_linear_glob"]))
        nonlinear_pths = sorted(pinn_dir.glob(scen["pinn_nonlinear_glob"]))
        if linear_pths and nonlinear_pths:
            for mode, pth_path in [("linear", linear_pths[0]),
                                   ("nonlinear", nonlinear_pths[0])]:
                model = load_pinn_model(pth_path, device)
                u, w, phi = pinn_predict_at_nodes(model, x)
                r_norm = _compute_residual_norm(u, w, phi, mode, residual_calc)
                w_max = float(w.abs().max().item())
                print(f"  [PINN-Adam   ] {mode:10s} ||R_PDE||={r_norm:.6e}  "
                      f"max|w|={w_max:.6f}  src={pth_path.name}")
                rows.append({
                    "scenario": scen_name, "k1": k1, "k2": k2,
                    "method": "PINN-Adam", "optimizer": "Adam",
                    "mode": mode,
                    "source": pth_path.name,
                    "max_abs_w": w_max,
                    "R_PDE_L2_norm": r_norm,
                })
        else:
            print(f"  [PINN-Adam   ] missing pth files in {pinn_dir}")

        # ---- (b) ODIL-DQ: 3 optimizers x 2 modes ----
        # Same metric as PINN: load the ODIL displacement field (u, w, phi) from
        # displacement.csv -- it lives exactly on the 13 Cheb-Lobatto DQ nodes, in
        # the same ascending order as cheb_lobatto_nodes() -- and run it through the
        # SAME _compute_residual_norm() used for PINN. This recomputes ||R_PDE||_2
        # under one identical evaluator and reproduces the R_PDE_norm that
        # run_odil.py already stored in summary.json (printed as a consistency check).
        odil_dir = scen["odil_dir"]
        for opt_name, opt_label in ODIL_OPTIMIZERS:
            for mode in ("linear", "nonlinear"):
                try:
                    u, w, phi = load_odil_displacement(odil_dir, opt_name, mode,
                                                       device, x.dtype)
                except FileNotFoundError as e:
                    print(f"  [ODIL-{opt_label:6s}] missing -> {e}")
                    continue

                r_norm = _compute_residual_norm(u, w, phi, mode, residual_calc)
                w_max = float(w.abs().max().item())

                # Consistency check against the value run_odil.py precomputed.
                ref = _read_summary_r_pde_norm(odil_dir, opt_name, mode)
                ref_str = f"{ref:.6e}" if ref is not None else "n/a"
                print(f"  [ODIL-{opt_label:6s}] {mode:10s} ||R_PDE||={r_norm:.6e}  "
                      f"max|w|={w_max:.6f}  (summary.json R_PDE_norm={ref_str})  "
                      f"src=displacement.csv")
                rows.append({
                    "scenario": scen_name, "k1": k1, "k2": k2,
                    "method": f"ODIL-DQ ({opt_label})", "optimizer": opt_label,
                    "mode": mode,
                    "source": "recomputed from displacement.csv (same metric as PINN)",
                    "max_abs_w": w_max,
                    "R_PDE_L2_norm": r_norm,
                })

    return pd.DataFrame(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] EXP_ROOT = {EXP_ROOT}")
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")

    df = evaluate(device)

    out_csv = EXP_ROOT / "results" / "pinn_vs_odil_strong_residual.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_long_path(out_csv), index=False)

    print("\n" + "=" * 100)
    print("Strong-form PDE residual ||R_PDE||_2 on 13 Cheb-Lobatto DQ nodes (interior masked)")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\n[OK] saved -> {out_csv}")


if __name__ == "__main__":
    main()
