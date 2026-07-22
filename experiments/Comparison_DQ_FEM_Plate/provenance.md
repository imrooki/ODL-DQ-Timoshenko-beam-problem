# Provenance & file sources — conventional_baseline (Associate-Editor Q3, plate)

> **Scope of this release.** This distribution contains the focused-comparison subset
> driven by `one_start_comp.py` (its dependency closure plus `compare_results/`).
> Sections below that reference other drivers of the full internal package
> (`run_dq_newton.py`, `run_fem_newton.py`, `compare_plate.py`, `run_odl_plate.py` as a
> standalone sweep, and their `results/` outputs) document the same code base, but those
> drivers and outputs are not required by `one_start_comp.py` and are not included here.

This package was assembled by copying from existing code bases, plus a small documented set
of edits and two newly-written solvers. No original file in any source project was modified.

Sources:
- **Work_4** = `…/Work_4/Plates` — `1_Plates/` (Python ODL plate solver, the proposed
  method) and `2_DQ_code/` (MATLAB DQ plate solver).
- **Q2** = `…/Response2reviewers/1_Associate_Editor_Q2/conventional_baseline` (the beam
  package this one mirrors).

## 1. Source of every file

| File in this package | Source | Edited? |
|---|---|---|
| `solver/odl/modules/*` (10 files) | Work_4 `1_Plates/modules/*` (bending closure) | no (byte-identical at copy time) |
| `solver/dq/dq_core.py` | Work_4 `1_Plates/modules/dq_core.py` | no (byte-identical) |
| `solver/dq/plate_strong.py` | Work_4 `1_Plates/modules/plate_strong.py` | yes — one import re-pointed (§2.1) |
| `material/material_plate.py` | Work_4 `1_Plates/modules/material_plate.py` | yes — A110/D0 reference functions added (§6) |
| `solver/dq/direct_dq_plate.py` | new — Python port of Work_4 `2_DQ_code/strong_form/strong_nonlinear.m` | new (§2.2) |
| `solver/fem/fem_plate_fsdt.py` | new — Q4 Mindlin FEM (no FSDT plate FEM existed in Work_4) | new (§2.3) |
| `solver/odl/run_odl_plate.py` | new — driver adapted from Work_4 `1_bending/main_bending.py` | new (§2.4) |
| `params_q3.py`, `run_dq_newton.py`, `run_fem_newton.py`, `compare_plate.py`, `README.md`, `provenance.md`, `nondim_derivation.md` | newly written for this package | n/a |

## 2. Edits applied & new code

### 2.1 `solver/dq/plate_strong.py`
- Only the dq_core import was re-pointed: `from .dq_core import …` → `from dq_core import …`,
  so the file works in the flat `solver/dq/` layout. Otherwise identical to the ODL
  `modules/plate_strong.py` copy (the linear strong system, foundation, and boundary map are
  unchanged).

### 2.2 `solver/dq/direct_dq_plate.py` (new)
- **Strong-form von-Kármán residual** ported from `strong_nonlinear.m` (`residual_full` /
  `strong_kinematics`): membrane balance `Dx·Nx+Dy·Nxy`, `Dx·Nxy+Dy·Ny`; transverse
  `Dx·Qx+Dy·Qy + Dx·(Nx·w,x+Nxy·w,y)+Dy·(Nxy·w,x+Ny·w,y) + FoundW·w`; moment
  `Dx·Mx+Dy·Mxy−Qx`, `Dx·Mxy+Dy·My−Qy`; `FoundW = −k_w I + k_s(Dxx+Dyy)`.
- **Hand-coded analytical consistent tangent** `K = dR/dd` (the 2D, five-field extension of
  the Q2 beam tangent), assembled from the strain-derivative operators, the resultant
  derivatives, and the W-equation product-rule + foundation terms.
- Newton–Raphson + backtracking line search + init-guess options (linear/zero/random/gauss)
  + relative convergence; `check_tangent_fd` central-FD gate.
- **Linear** solve reuses `build_strong_form_plate_system` → identical discretisation to ODL.
- The imperfection-slope terms (`Wbarx/Wbary`) of `strong_nonlinear.m` are omitted
  (`A0 = 0` bending benchmark), as in Q2.
- **Nonlinear natural-boundary rows** (for SSSS): `_residual_full` and the hand-coded
  `_tangent_full` impose the conjugate-resultant conditions at natural (free) boundary DOFs —
  `Nx/Ny/Nxy/Mx/My/Mxy = 0` (comp 1-6, e.g. SSSS) and the von-Kármán effective transverse
  shear `TwX = Qx + Nx w,x + Nxy w,y` / `TwY` + free-edge Pasternak (comp 7/8, free edges).
- **`build_strong_vk_residual`** (torch, vmap-safe): the reduced strong residual used by the
  hard-BC strong-form ODL solver; equals the numpy `_residual_full` to 2.5e-16.
- **`build_strong_vk_residual_soft`** (torch, vmap-safe): the soft-BC strong residual over the
  full DOF vector — interior PDE collocation + natural-edge resultant rows + a Dirichlet
  penalty `bc_weight·d` on essential DOFs, row-scaled (PDE/`s_pde`, penalty/`d_scale`) so the
  dimensional plate is well-conditioned.

### 2.3 `solver/fem/fem_plate_fsdt.py` (new)
- 4-node bilinear Mindlin Q4 element, von-Kármán energy, **selective reduced integration**
  (2×2 Gauss for membrane/bending/foundation/load, 1×1 reduced for transverse shear).
- **Hand-coded analytical element residual + consistent tangent** — material `BᵀCB` +
  von-Kármán geometric stiffness (`σ·∂²ε`) + foundation; no autograd.
  **Dense assembly + `np.linalg.solve` by default**; optional `scipy.sparse` path via
  `sparse=True` (same parameter; dense vs sparse agree ~1e-15). Standard Newton–Raphson +
  backtracking line search + linear-solution warm start. Material stiffness `(A,B,D,As)` from
  the shared `material/material_plate.py`. Mesh sweep `O(h²)`; the dense default sizes the sweep
  to `[8,16,32]` (nx=32 → 5445 DOF, ~0.24 GB; nx=64 dense ≈ 3.6 GB — use `sparse=True` for a
  finer mesh).
- **Boundary conditions** `_edge_essential`/`_essential_dofs` match the spectral
  `plate_strong._edge_essential_fields` (CCCC = all five fields; SSSS `ss1` = `{V,W,phiy}` on
  x-edges, `{U,W,phix}` on y-edges, union at corners), so the FEM enforces the *same* BC.
- **Linear-solution warm start** in `solve_nonlinear` (the von-Kármán tangent at `d=0` is the
  linear stiffness), so standard single-step Newton converges for the soft, strongly-nonlinear
  SSSS plate.

### 2.4 `solver/odl/run_odl_plate.py` (new)
- Driver only; the ODL `modules/` stack is identical to Work_4. Runs the ODL formulations over
  the CCCC/SSSS × foundation sweep: **strong-form ODL (hard BC, GN)**; **strong-form ODL (soft
  BC penalty)** over the three optimizers LM / GN / L-BFGS; and **energy-form ODL** (L-BFGS Ritz
  energy linear + LM weak von-Kármán residual nonlinear). The `odl_plate_baseline.csv` has
  `form`, `optimizer`, `converged`, `elapsed_s` columns. Also writes loss history and the
  optimised discrete fields (the ODL "models").

### 2.5 `params_q3.py`, `run_dq_newton.py`, `run_fem_newton.py`, `compare_plate.py`
- `params_q3.py` adds `BOUNDARY_CONDITIONS = [CCCC, SSSS(ss1)]`; `make_config(... , bc=None)`
  builds the per-BC config. The runners loop boundary condition × foundation scenario;
  `compare_plate.py` consolidates all four solves and plots the center-line deflection profiles
  (linear & nonlinear) per BC/scenario.
- `params_q3.py` also adds `use_cuda` (default `True`) + `resolve_device()`: the torch-based
  **ODL** solvers run on **GPU** when available (else CPU); the conventional DQ-Newton (numpy)
  and FEM (scipy sparse) stay CPU. The vendored `odl_config.py` is unmodified (still
  `DEFAULT_DEVICE="cpu"`); the GPU device is supplied via the per-call `device=` override the
  module documents. ODL-on-GPU reproduces the CPU result to ~1e-8. Measured: CPU is ~4× faster
  at N=15 on this RTX 3070 (consumer float64), so the device is a config knob, not a
  result-affecting choice.

## 3. Integrity

- The Work_4 source files are unchanged after copying.
- `solver/odl/modules/*`, `solver/dq/dq_core.py`, and `material/material_plate.py` were copied
  byte-identical from the Work_4 `1_Plates` sources at copy time; `solver/dq/plate_strong.py`
  differs from its source only by the one import line in §2.1. The nondim path (§6) later adds
  explicit branches to `plate_weak.py`, `plate_nonlinear.py`, and `material_plate.py`.

## 4. Cross-validation

- **Numerical cross-validation.** The four solves — strong-form ODL, energy-form ODL,
  Direct-DQ Newton, weak-form FEM — agree to **≤ 9.2e-4** on the center deflection for CCCC and
  SSSS, with and without foundation, linear and nonlinear
  (`results/comparison/plate_comparison.csv`; nondim default ON, dense FEM nx=32). The spread is
  dominated by the FEM's `O(h²)` mesh error; the **three spectral solvers agree to ~4e-5**, and
  strong-form ODL ≡ Direct-DQ Newton (same discrete residual, optimised vs rooted) to **≤ 1e-7**
  — also for the soft-BC strong-ODL, whose nonlinear LM/GN match the hard-BC solution.
  The FEM converges `O(h²)`.
- **Boundary conditions.** SSSS uses the `ss1` convention (free normal rotation → `M_n = 0`), a
  valid simply-supported BC; the movable `starter` SS is singular for pure bending (in-plane
  rigid-body modes) and is not used.
- **Optimizer sweep (strong-form soft-BC ODL).** Over LM / GN / L-BFGS: LM and GN reach the
  DQ-Newton solution to ~1e-7 in 4–57 iterations. Plain L-BFGS minimises `‖R‖²` (inheriting
  `cond²`) and is pathological on the strong collocation residual — kept switchable
  (`params_q3.LBFGS_MULTIGRID = False`) as the squared-residual conditioning exhibit. With
  the ODIL multigrid decomposition (`LBFGS_MULTIGRID = True`, the default) L-BFGS reaches
  the same plate solution (see `compare_results/comparison.csv`), though its wall-clock
  stays well above LM/GN; it also converges efficiently on the energy form. LM/GN remain
  the practical strong-form ODL optimizers.

## 5. ODL solver — the proposed method

- The `solver/odl/modules/` stack is the **bending closure** (10 files) of Work_4
  `1_Plates/modules/`. The vibration/dynamics modules (`eigen_solver.py`, `dynamics_odl.py`)
  are excluded as out of scope (bending only), consistent with Q2's bending-only scope.
  `run_odl_plate.py` is the only new file in `solver/odl/`; the numerical solvers are unchanged
  from the source.

## 6. Nondimensionalization (explicit A110/D0 path; default ON)

An explicit `NONDIM` path is provided in **all four** solvers (gated by `params_q3.NONDIM`,
default `True`); the dimensional path (`NONDIM=False`) is preserved and still selectable. The
full derivation, per-form coefficients, and verification table are in **`nondim_derivation.md`**.
Summary:

- **Scaling.** x=X/a, y=Y/b ∈ [0,1]²; u,v,w = U,V,W/h; λ₁=a/h, λ₂=a/b. **Pure-copper**
  references A110 = E_Cu·h/(1−ν²), D0 = E_Cu·h³/(12(1−ν²)) (`material_plate.compute_A110_reference`
  / `compute_D0_reference`); scaled stiffness a_ij=A/A110, b_ij=B/(h·A110), d_ij=D/(h²·A110),
  a_s=As/A110.
- **Strong** (Direct-DQ Newton, strong-ODL): the five λ-factored equations on the [0,1]² grid;
  load q·a²/(h·A110); foundation cw=k_w·a²/A110, cs=k_s/A110.
  **Weak** (energy-ODL, FEM): nondim energy = E_dim/(A110·a·b) → load q·h/A110; global-weak
  foundation cw=k_w·h²/A110, cs=k_s·h²/(A110·a²); FEM foundation cs=k_s/A110 (effective-operator
  form, differs by λ₁²). DQ + FEM use hand-coded analytical nondim tangents (no autograd); ODL
  (strong + energy) autograd-differentiate the torch residual/energy.
- **Equivalence.** The nondim minimiser is `disp_scale⁻¹·d_dim`; every runner recovers the
  physical displacement via `direct_dq_plate.to_physical` (× h on u,v,w; no-op when dimensional)
  before `extract_center_deflection`, so reported `w/h` is identical for both paths.
- **Foundation D0 convention.** The pure-copper `D0` (≈0.51× the laminate `D[0,0]`): with
  foundation, deflections are larger; no-foundation is unaffected. Both paths consume the same
  physical `(k_w, k_s)`.
- **FEM mesh.** The hand-coded dense(default)/sparse Newton FEM (§2.3) is reported at dense
  nx=32; the `nondim == dimensional` equivalence is mesh-independent (verified at nx=4), and the
  FEM↔spectral agreement at nx=32 is the FEM's own O(h²) discretisation error.

**Verification (square a=b=1, h=0.05, N=15).** DQ `nondim==dim` ~1e-16 & FD tangent ~1e-11;
strong-ODL torch==numpy ~1e-16; energy-ODL linear ~1e-8 / nonlinear ~1e-5 (LM floor); FEM
`nondim==dim` linear ~1e-14, dense==sparse ~1e-15, nonlinear ~5e-7 (NR floor); full runner path
(`to_physical → extract → /h`) DQ exact, ODL ~1e-5. The soft-BC strong-ODL carries the explicit
λ-factored strains and five equations (mirroring `residual_nd`); its LM/GN reach the hard-BC
solution (`w/h` 0.3498 CCCC / 0.7958 SSSS).

**Known limitations** (outside the CCCC/SSSS, `ΔT=0` benchmark scope):
1. **FREE-edge natural BC.** The nondim natural rows enforce the bare resultants (exact for
   SSSS `M_n=0`); the dimensional path's FREE-edge effective-shear (`TwX/TwY`) + Pasternak
   (`k_s ∂w/∂n`) natural terms are not mirrored in the nondim path. The benchmark uses only
   CCCC/SSSS.
2. **Thermal resultants in the FEM.** The FEM element omits `NT/MT` (its bending-only scope;
   inert for `ΔT=0`); the DQ/ODL paths scale thermal (`NT/A110`, `MT/(h·A110)`). For `ΔT≠0` the
   FEM would need the thermal term to stay consistent with the other three solvers.

## 7. Comment wording cleanup

Comment and docstring wording across this package was tidied for publication
(terminology and phrasing only). No code, parameter, or numerical content was
changed by this pass.

