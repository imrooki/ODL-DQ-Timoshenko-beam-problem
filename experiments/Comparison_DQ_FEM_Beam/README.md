# Conventional Newton-Raphson baseline for Associate-Editor Q2

**Q2 (Associate Editor):** *"ODL is essentially a classical discretization-based
nonlinear solver with a small number of unknowns... How does ODL compare against a
conventional Newton-Raphson FEM/FDM/DQ solver?"*

This self-contained package provides exactly that comparison baseline: a **conventional
Newton-Raphson (and Picard) solver** for the *same* Timoshenko/FSDT bending problem
solved by our ODIL method, so the two can be compared apples-to-apples on identical
material and identical boundary-value problem. For the **Direct DQ** baseline the
discretisation is *identical* to ODIL (N=13 CGL + negative-sum); the **FEM** baseline is
an independent, mesh-converged weak-form solver that cross-checks the same BVP through a
*different* discretisation. (The FEM residual and tangent are obtained by automatic
differentiation of the finite-element energy — this is a conventional displacement FEM,
not a neural-network solver.)

It contains **two independent conventional solvers**:

| Solver | Form | Discretisation | Nonlinear strategy |
|---|---|---|---|
| **Direct DQ** (`solver/dq/`) | strong form | Differential Quadrature, N=13 Chebyshev–Gauss–Lobatto + negative-sum (identical to ODIL) | **Newton-Raphson** (tangent stiffness, **default**); Picard (fixed-point) kept as cross-check |
| **FEM** (`solver/fem/`) | weak / energy form | 2-node linear elements + selective reduced integration | **standard Newton-Raphson** (single load, full Newton steps from the chosen initial guess; backtracking line-search retained only as a safeguard) |

## Scope (what this baseline covers)

- **Bending only.** No vibration / eigenvalue code.
- **No geometric defect** (`A0 = 0`): the formulation is purely bending — no `defect_params`,
  no `dw0` branches/helpers, and no `defect_functions` dependency.
- Benchmark: `T = 300 K =` reference temperature, so `ΔT = 0 → n_xT = m_xT = 0`.
- Boundary condition: **C-C** (clamped-clamped; pure Dirichlet, cleanest DQ/FEM/ODIL comparison).
- Two elastic-foundation scenarios: with `(k1, k2) = (0.01, 0.001)` and without `(0, 0)`.

## Apples-to-apples guarantee

The material coefficients (`a11, b11, d11, a55, λ, n_xT, m_xT`) are computed from the
**same `material_properties.py`** (see `material/`) that feeds the ODIL solver. Both run
scripts print these coefficients so the reader can confirm they match the ODIL run. The DQ
grid (N=13 CGL + negative-sum) also matches ODIL.

## How to run (conda env `claude_test`)

```powershell
conda activate claude_test
python run_dq_newton.py    # Direct DQ: linear + nonlinear Newton-Raphson (default) + Picard (cross-check)
python run_fem_newton.py   # FEM: linear + nonlinear standard Newton-Raphson (single load)
```
Outputs go to `results/dq_baseline.csv` and `results/fem_baseline.csv`.

## Verified results (mid-span deflection `w(x=0.5)`)

| Scenario | Method | linear | nonlinear | reference (ODIL) |
|---|---|---|---|---|
| with foundation | DQ-Newton | −0.468340 | −0.428953 | −0.46834 / −0.42895 |
| with foundation | DQ-Picard | −0.468340 | −0.428953 | (same) |
| with foundation | FEM-Newton (n_elem=1500) | −0.468340 | −0.428952 | (same, O(h²)) |
| no foundation | DQ-Newton | −0.521127 | −0.465201 | −0.52113 / −0.46520 |
| no foundation | FEM-Newton (n_elem=1500) | −0.521126 | −0.465200 | (same, O(h²)) |

All three conventional solvers (DQ-Newton, DQ-Picard, weak-form FEM-Newton from the
linear warm start) and ODIL converge to the **same** solution. Within the DQ baseline, Newton-Raphson and Picard
reach an identical solution (`|Δw| ~ 1e-11`); Newton converges in fewer iterations and
less wall-time. This is the empirical basis for the Q2 reply: on this forward,
well-posed bending problem the ODIL solve and a conventional Newton-Raphson DQ solve are
in the *same* solver class and produce the *same* result.

## Directory layout

```
conventional_baseline/
├── params_q2.py            # benchmark parameters
├── run_dq_newton.py        # Direct DQ baseline runner
├── run_fem_newton.py       # FEM baseline runner
├── solver/
│   ├── dq/                 # Direct DQ (strong form)
│   │   ├── direct_dq.py    #   Newton-Raphson / Picard bending solver (strong form)
│   │   ├── dq_stiffness.py #   bending stiffness assembly (linear + nonlinear)
│   │   └── dq_core.py      #   DQ weight matrices (Chebyshev-Lobatto + negative-sum)
│   ├── fem/
│   │   └── fem_timoshenko.py  # weak-form FEM (standard single-load Newton-Raphson)
│   └── odil/                  # the proposed method (ODIL)
│       ├── run_odil.py        #   ODIL sweep runner
│       ├── params_odil.py     #   ODIL config (seed=42; LM / GN / L-BFGS)
│       ├── modules/           #   ODIL stack (solver_odil, residuals, dq_core, ...)
│       └── utils/             #   ODIL utils (nonlinear_solving, tensor_ops, ...)
├── material/
│   └── material_properties.py # the material_properties.py used by the ODIL solver
└── results/                # generated CSVs (dq_baseline.csv, fem_baseline.csv; ODIL -> results/odil/)

```

## ODIL solver (the proposed method) — `solver/odil/`

For a self-contained head-to-head, the **ODIL** solver (the method this paper proposes) is
included under `solver/odil/`. Only the package root and output directory in `run_odil.py`
are set for this layout.

Run it (conda env `claude_test`):

```powershell
python solver/odil/run_odil.py                                      # full sweep (LM, GN, L-BFGS) x 2 scenarios
python solver/odil/run_odil.py --only-optimizer levenberg-marquardt # one optimizer
```
Outputs go to `results/odil/<scenario>/<optimizer>/`.

**Reproducibility.** The ODIL sweep is fully deterministic (`seed = 42`, CPU — the optional
GPU-acceleration module is not shipped in this package, so the run falls back to CPU
regardless of the `use_cuda` flag in `params_odil.py`). The three
optimizers (LM, GN, L-BFGS) and both foundation scenarios reproduce the deflection values
in the table above; on this strong-form residual L-BFGS needs many more iterations
(≈24600 / 18100) than LM or GN.

See `provenance.md` for the source of every file.
