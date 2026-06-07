# ODIL-FSDT: Bending of Functionally Graded Timoshenko Beams

Static bending of FG graphene-platelet-reinforced **Timoshenko beams** (First-order Shear
Deformation Theory) on a Winkler–Pasternak elastic foundation, solved with the
**ODIL (Optimizing a Discrete Loss)** method: the governing PDEs are discretized and the
resulting **discrete residual loss** is minimized. Both **linear** and **nonlinear**
(von Kármán) formulations are supported.

## Features

- **Unknowns:** axial `u`, deflection `w`, rotation `φ`; all derivatives via **autodiff** (CPU/CUDA).
- **Discretization:** `dq` (differential quadrature), `taylor` (Taylor/Fornberg), `spline`
  *(GPR and RBF-FD are not included in this release)*.
- **Optimizers:** `lbfgs` (with adaptive perturbation-restart), `adam`, `gauss-newton`, `levenberg-marquardt`.
- **Material:** FG graphene-platelet composite (Halpin–Tsai), through-thickness patterns `X` / `U` / `O`.
- **Elastic foundation:** Winkler `k1` + Pasternak `k2`.
- **Boundary conditions:** `C-C`, `S-S`, `H-H`, `C-S`, `C-H`.

## Install

Python 3.13 (tested on 3.13.3):

```bash
pip install -r requirements.txt
```

For GPU, install the matching CUDA build of PyTorch (tested with `torch==2.8.0+cu128`); the CPU build also works.

## Usage

Edit **`params.py`** (geometry, material, method, optimizer, BC, load, foundation, `mode`), then run:

```bash
python main.py
```

Outputs are written to `results/main/<bc_type>/<distribution>/<parameter-combo>/`: `data/`
(displacement CSVs), `models/` (best-loss linear & nonlinear checkpoints), `plots/`
(deflection & loss curves), `logs/`. Results are deterministic for a fixed `seed` on the same hardware.

Key `params.py` settings:

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `method` | discretization: `dq` / `taylor` / `spline` | `dq` |
| `optim_name` | `lbfgs` / `adam` / `gauss-newton` / `levenberg-marquardt` | `levenberg-marquardt` |
| `mode` | `linear` / `nonlinear` / `both` | `both` |
| `bc_type` / `distr_type` | boundary condition / material distribution | `C-C` / `X` |
| `q` | dimensionless distributed load | `-0.08` |
| `N` | number of nodes | `13` |
| `k1`, `k2` | Winkler / Pasternak stiffness | `0.01`, `0.001` |
| `seed` | random seed | `42` |

## Layout

```
Opensource/
├── main.py          # entry point (reads params.py)
├── params.py        # all configuration
├── modules/         # solver_odil, residuals, dq/taylor/spline cores, material_properties
├── utils/           # boundary conditions, output manager, logger, helpers
└── experiments/
    ├── Comparison_of_computational_time/   # ODIL-vs-PINN study (self-contained)
    └── ODL-LBFGS/                           # LBFGS-only sweep, adaptive perturbation-restart disabled
```

## ODIL-vs-PINN comparison

`experiments/Comparison_of_computational_time/` reproduces a wall-clock / accuracy comparison
against a Pure-PINN baseline, reusing the core `modules/` and `utils/`:

```bash
python odil/run_odil.py     # ODIL (DQ) sweep over scenarios / optimizers
python pinn/run_pinn.py     # Pure-PINN baseline
python compare.py           # aggregate and compare
```

## Notes

`material_properties.py` is the validated reference material model. Add your preferred license before publishing.
