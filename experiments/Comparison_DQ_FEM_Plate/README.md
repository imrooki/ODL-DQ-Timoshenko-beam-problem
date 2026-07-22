# Focused plate comparison — ODL vs conventional Newton solvers (Associate-Editor Q3)

This package is the **focused head-to-head comparison** for the Associate Editor's Q3
(extend the beam comparison to a larger PDE system): **one** FSDT (Mindlin) plate bending
boundary-value problem — a square GOEAM graphene-FGM plate under uniform transverse load,
von Kármán geometric nonlinearity, Winkler–Pasternak elastic foundation — solved by

* **ODL (proposed)** — strong-residual form, soft (penalty) BC, swept over the three
  optimizers **LM / GN / L-BFGS** (L-BFGS uses the ODIL multigrid decomposition);
* **Direct-DQ Newton** — conventional strong-form Newton-Raphson, hand-coded consistent
  tangent, **identical spectral discretisation** (CGL N=21) to the ODL rows;
* **weak-form FEM Newton** — independent Q4 Mindlin displacement FEM (selective reduced
  integration), at its primary O(h²)-converged mesh nx=96,

for **CCCC and SSSS** boundary conditions, **with and without** the elastic foundation.
Every problem / discretisation / solver parameter is pinned in `params_q3.py` and held
identical across rows; the only thing that varies is the method (and, for ODL, the
optimizer). See the docstring of `one_start_comp.py` for the full protocol, including
the residual, timing-scope, and device conventions.

## How to run (conda env `claude_test`)

```powershell
conda activate claude_test
python one_start_comp.py
```

**Memory warning:** the FEM rows run the primary mesh `FEM_NX = 96` with the dense solver
(~50 GB RAM linear / ~68 GB nonlinear). On a smaller machine lower `FEM_NX` (nx=64 ≈ 13 GB,
nx=32 ≈ 0.2 GB) or set `FEM_SPARSE = True` in `one_start_comp.py` (sparse path,
dense-identical to ~1e-15). The `discretization` column always records the actual mesh.

## What ships

* `compare_results/comparison.csv` — one row per (BC, scenario, method, optimizer):
  linear and nonlinear center deflection, iterations, method-native residual,
  and wall-clock (`lin_*` / `nl_*` column pairs; full pipeline = `lin_total_s + nl_total_s`).
* `compare_results/logs/` — the 12 ODL optimisation loss histories.
* `provenance.md` — source of every module; `nondim_derivation.md` — the
  A110/D0 nondimensionalization used by all solvers (`params_q3.NONDIM = True`).

## Shipped result (nonlinear center deflection, `|w|/h`)

All five solves agree on the deflection to ≤ 5e-4 relative on every case
(CCCC: 0.3498 with / 0.3948 without foundation; SSSS: 0.7958 / 0.9752).

**Nonlinear solve wall-clock (s)** from the shipped `comparison.csv`
(DQ totals include assembly; FEM totals include its interleaved assembly; ODL rows are
pure solve time on GPU — see the timing-scope NOTE in `one_start_comp.py`):

| case | DQ-Newton | FEM (nx=96) | ODL-LM | ODL-GN | ODL-L-BFGS |
|---|---|---|---|---|---|
| CCCC, with foundation | 1.44 | 433 | 1.09 | 1.10 | 375 |
| CCCC, no foundation   | 1.49 | 444 | 1.11 | 1.09 | 505 |
| SSSS, with foundation | 1.94 | 618 | 23.7 | 116  | 1517 |
| SSSS, no foundation   | 2.32 | 924 | 47.0 | 141  | 1505 |

Reading the table:

* Residual columns are **method-native and not cross-comparable** (`residual_kind`):
  DQ uses the absolute strong-residual L2 (< 1e-10), FEM the relative weak residual
  (< 1e-8), ODL the row-scaled soft residual (< 1e-3). The physically comparable
  quantities are the center deflection and the wall-clock.
* On the clamped plate the second-order ODL optimizers (LM/GN) are on par with
  Direct-DQ Newton; the gap opens on SSSS, where the natural-boundary rows make the
  soft system harder. The FEM cost reflects mesh convergence of an independent
  discretisation, not solver inefficiency.
* The L-BFGS rows reach the 1e-3 engineering bar of the row-scaled soft residual
  (SSSS: ~1.5e-4 at the 50000-evaluation budget), not the ~1e-9 floor of LM/GN;
  they demonstrate that a first-order optimizer configured per the ODIL multigrid
  prescription reaches the same solution at plate scale, while **LM/GN remain the
  practical strong-form ODL optimizers**.

## Scope note

This is the focused-comparison subset of a larger internal baseline package;
`provenance.md` also documents drivers (mesh-sweep FEM runner, four-solver
cross-validation, plotting) that are not required by `one_start_comp.py` and are
not included here.
