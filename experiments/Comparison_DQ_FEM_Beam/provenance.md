# Provenance & file sources — conventional_baseline

This package was assembled by copying from existing code bases and adding a small,
documented set of edits plus a few newly-written files. No original file in any source
project was modified.

## 1. Source of every file

| File in this package | Source | Edited? |
|---|---|---|
| `solver/dq/direct_dq.py` | Work_5 `FEM_Code/modules/direct_dq.py` | yes — see §2.1 |
| `solver/dq/dq_stiffness.py` | Work_5 `FEM_Code/modules/vibration_matrices.py` (renamed) | yes — bending-only extraction, §2.2 |
| `solver/dq/dq_core.py` | Work_5 `FEM_Code/modules/dq_core.py` | no |
| `solver/fem/fem_timoshenko.py` | Work_5 `FEM_Code/fem_timoshenko.py` | yes — one import re-pointed, §2.3 |
| `material/material_properties.py` | Work_2 `modules/material_properties.py` | no |
| `params_q2.py`, `run_dq_newton.py`, `run_fem_newton.py`, `README.md`, `provenance.md` | newly written for this package | n/a |

Work_5 = `…/Work_5/D-PINNs/2_D_PINNs_vibration_only/FSDT_vibration_val`
Work_2 = `…/Work_2/…/FSDT_bending` (this project)

## 2. Edits applied to the copies

### 2.1 `solver/dq/direct_dq.py`
- The three vibration wrapper functions (`solve_vibration_linear_direct`,
  `solve_vibration_linear_batch`, `solve_vibration_nonlinear_direct`) and their vibration
  branches in the `solve_direct_dq` dispatcher and in `__all__` are not included (bending only).
- Defect code is not included: the defect helpers (`_compute_dw0_array_numpy`,
  `_compute_dw0_full_arrays`), the `defect_params` / `x_np` parameters, and the `has_defect`
  branches in `_compute_bending_residual_and_tangent` (R1/R2/R3 + K-blocks), in the BC row
  helpers, and in `_apply_bc_newton` are absent. The formulation is purely bending (`A0 = 0`);
  the bending expressions are unchanged.
- Import re-pointed: `from vibration_matrices import …` → `from dq_stiffness import …`
  (the renamed, bending-only assembly module).
- The module docstring describes the bending-only baseline.
- Default nonlinear method is Newton-Raphson (`iteration_method` default `'newton'`); the
  obsolete `'direct'` alias is dropped from the validation list.
- Consistent tangent (Newton branch `K22`): the von-Kármán tangent prefactor is `3a11/(2λ²)`
  — the exact consistent tangent (the Work_5 vibration copy used a Newton-like `3a11/λ²`).
  The converged root `R = 0` is unchanged (`w(0.5)` identical to ≥10 digits and to the Picard
  cross-check); the exact tangent restores quadratic convergence (Newton 4–5 iterations).
- Otherwise the bending solver core (linear solve, Picard loop, residual, BCs) is the Work_5
  original.

### 2.2 `solver/dq/dq_stiffness.py` (from `vibration_matrices.py`)
- Contains only the bending pieces: base backend helpers, `assemble_linear_stiffness_matrix`,
  `assemble_nonlinear_stiffness_matrix`, `get_boundary_dofs`.
- Vibration code is not included: mass-matrix assembly, generalized-eigenvalue solvers,
  reduced-eigen / mode machinery, the `scipy.linalg.eig` import, the `dataclass` import.
- Defect code is not included (`_has_active_defect`, `_compute_defect_derivatives`,
  `_row_scale`, `_to_numpy`, `_to_same_backend`, the `defect_params` / `x` parameters, and the
  `defect_active` branches in both `assemble_*` functions). The bending assembly expressions
  are unchanged.

### 2.3 `solver/fem/fem_timoshenko.py`
- Only `load_material_params` was re-pointed: it imports the copy of `material_properties.py`
  in `material/` instead of reaching into a main project. The energy functional, element, SRI,
  and autograd residual/tangent are unchanged.
- Default nonlinear method is standard Newton-Raphson (`n_load_steps = 1`: single full-load
  Newton from the chosen initial guess). The backtracking line-search is retained only as an
  inactive safeguard; the reported Q2 runs accept full Newton steps (`|R|∞ ~ 1e-13`).
- `fem_core.py` from Work_5 is not copied (the FEM solver does not import it).

## 3. Integrity & correctness

- Originals are untouched: the Work_5 `direct_dq.py`, `vibration_matrices.py`, `dq_core.py`,
  `fem_timoshenko.py`, `fem_core.py` and the Work_2 `material_properties.py` are unchanged
  after the copy/edit work.
- Apples-to-apples material: both runners print `a11,b11,d11,a55,λ,n_xT,m_xT`; for the
  benchmark they are `a11=1.4815…, b11≈3.9e-17, d11=0.16400…, a55=0.76940…, λ=20,
  n_xT=m_xT=0` — identical to the ODIL run, confirming the same problem.
- Numerical match (mid-span `w(0.5)`): DQ-Newton and DQ-Picard reproduce the ODIL / reference
  values to all printed digits (e.g. −0.468340 vs −0.46834); weak-form FEM-Newton matches to
  O(h²) at n_elem=1500. DQ Newton vs Picard agree to `|Δw| ~ 1e-11`.

## 4. Nonlinear initial guess, timing, and physics notes

- Nonlinear initial-guess option in DQ (`solve_bending_nonlinear_direct`) and FEM
  (`solve_nonlinear`): `init_guess` ∈ {`random`(default), `gauss`, `zero`, `linear`}, with
  `init_seed` (reproducible RNG) and `init_scale` (default 0.01). The DQ run script sweeps all
  four init guesses × {Newton, Picard}; the FEM baseline is reported with the linear warm start.
  - *Physics note*: the von-Kármán discrete residual is a multi-root system; Newton from a
    large-amplitude random start converges (`‖R‖→0`) to a non-physical root (e.g. DQ
    `scale=1.0` → w=−31.96). `init_scale=0.01` keeps the random start inside the physical
    solution's basin. Picard is init-robust (any init → physical).
- Timing: a wall-clock timer inside both solvers (around the solve; caller-prepared
  material / DQ-weights excluded) returns `info["elapsed_s"]` / `out["elapsed_s"]`, reported by
  the run scripts.
- Physics details:
  1. FEM `random/gauss` init assigns values only to the free DOFs (`d[~free] = 0.0` on the
     fixed Dirichlet DOFs, which Newton never updates); the reported FEM baseline uses the
     linear warm start.
  2. In the DQ natural-BC rows (H/F) the thermal terms are `n_xT` / `m_xT`, consistent with the
     `N`/`M` definitions and with the interior usage (`+n_xT·w''`, no `λ`). For the C-C / T=300
     benchmark this has no effect (C-C has no natural-BC rows and `n_xT=m_xT=0`).
  3. The FEM `K_T` is the Hessian of a scalar potential and is symmetric.
- Physics verification:
  - Cross-method: DQ-Newton = DQ-Picard = FEM = ODIL reference (FEM from the linear warm
    start); all four init guesses converge to the physical solution for DQ.
  - The FSDT von-Kármán governing equations (R1/R2/R3), the consistent tangent `K_T`, the
    C/H/F boundaries, the FEM energy/SRI, and the linear/nonlinear assembly were checked and
    confirmed correct.

## 5. ODIL solver — `solver/odil/`

- The ODIL solver (`modules/` 8 files + `utils/` 12 files) and the comparison driver
  (`run_odil.py`, `params_odil.py`).
- Included under `solver/odil/` by copy: `modules/`, `utils/`, `params_odil.py` unchanged;
  `run_odil.py` has two edits only — `PROJECT_ROOT = THIS_DIR` (so `modules.*`/`utils.*`
  resolve to the local copies) and `results_root = PKG_ROOT/results/odil` (= conventional_baseline/results/odil). No reach-back to a
  source bundle or main project at runtime.
- Device: the optional GPU-acceleration module is absent, so the ODIL run falls back to CPU
  (`dq_core.py` try/except), fully deterministic with `seed=42`.
- Results (`summary.json` numerical fields `nonlinear.w_max`, `R_PDE_norm`, `final_loss`,
  `iterations`; both scenarios):
  - LM: nl.w_max −0.42895266602417653 / −0.4652009185710251.
  - GN: 45 iters; 0.4289526660399755 / 0.4652009185710251.
  - L-BFGS: 24629 / 18141 iterations; nl.w_max 0.42895260084038483 / 0.46520084020996216.

## 6. Comment wording cleanup

Comment and docstring wording across this package was tidied for publication
(terminology and phrasing only). No code, parameter, or numerical content was
changed by this pass.

