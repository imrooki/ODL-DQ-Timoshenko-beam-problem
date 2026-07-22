# Q3 Plate — Nondimensionalization Derivation & Verification

Technical reference for the explicit `NONDIM` path added to all four Q3 FSDT-plate solvers
(strong-ODL, energy-ODL, Direct-DQ Newton, Q4 FEM). It documents the scaling, the per-form
coefficient derivations, the `nondim == dimensional` equivalence, the conditioning rationale,
the verification numbers, and the known limitations. Reproducible from the cited
file/function locations.

---

## 1. Why nondimensionalize

The proposed method (ODL) minimizes the **discrete loss** `||R||²` by Gauss–Newton/LM. In
dimensional units the plate residual rows span ~`1e5–1e12` (load `q ~ 2.5e7`, stiffness
`A ~ 1e9`, `D ~ 1e6`, `As ~ 1e8`), so the Gauss–Newton normal matrix `JᵀJ` spans ~`1e10–1e12`
and is badly conditioned. Nondimensionalizing on the pure-copper references makes **every**
coefficient `O(0.02–1.5)` (e.g. the strong load becomes `f = q·a²/(h·A110) ≈ −0.13`), which
balances the rows and conditions the least-squares loss. The conventional Newton solvers
(DQ, FEM) are unaffected by conditioning but are nondimensionalized identically so the four
solvers remain apples-to-apples.

## 2. Scaling

| quantity | scaling |
|---|---|
| coordinates | `x = X/a`, `y = Y/b ∈ [0,1]²` |
| displacements | `u, v, w = U, V, W/h`; `phix, phiy` unchanged |
| aspect ratios | `λ₁ = a/h` (side-to-thickness), `λ₂ = a/b` (in-plane) |
| references (pure copper) | `A110 = E_Cu·h/(1−ν_Cu²)`, `D0 = E_Cu·h³/(12(1−ν_Cu²))` |
| stiffness identity | `D0 = A110·h²/12` (checked in `material_plate.compute_*`) |
| scaled stiffness | `a_ij = A/A110`, `b_ij = B/(h·A110)`, `d_ij = D/(h²·A110)`, `a_s = As/A110` |

`A110`, `D0` come from `material/material_plate.py::compute_A110_reference / compute_D0_reference`.

## 3. Nondim strains (membrane physical, curvature ×h)

```
exx = (1/λ₁)(u_x + (1/(2λ₁)) w_x²)          gxz = (1/λ₁) w_x + phix
eyy = (λ₂/λ₁)(v_y + (λ₂/(2λ₁)) w_y²)        gyz = (λ₂/λ₁) w_y + phiy
gxy = (1/λ₁)(λ₂ u_y + v_x + (λ₂/λ₁) w_x w_y)
kxx = (1/λ₁) phix_x   kyy = (λ₂/λ₁) phiy_y   kxy = (1/λ₁)(λ₂ phix_y + phiy_x)
```

The membrane strains equal the **physical** strains; the curvatures carry an extra `h`
(`kxx = (1/λ₁)phix_x = (h/a)phix_x = h·κ_xx`). This extra `h` pairs with `d_ij = D/(h²·A110)`
so the bending energy density `d·(h·κ)² = (D/A110)·κ²` is consistently `(1/A110)×` the
physical density — the key fact that makes the energy form scale cleanly (§5).

Constitutive: `{n;m} = [[a,b],[b,d]]{e;k}`, `{qx,qy} = a_s{gxz,gyz}`.

## 4. Strong form (Direct-DQ Newton, strong-ODL)

`plate_strong.py::_build_strong_form_nondim` (linear), `direct_dq_plate.py::_residual_full_nd`
/ `_tangent_full_nd` (nonlinear, **hand-coded** tangent), `build_strong_vk_residual` (torch).

The five collocation equations (= 0), on the `[0,1]²` CGL grid:

```
(1) n_xx,x + λ₂ n_xy,y
(2) n_xy,x + λ₂ n_yy,y
(3) λ₁ q_x,x + λ₁λ₂ q_y,y + (n_xx w_x),x + λ₂(n_xy w_y),x + λ₂(n_xy w_x),y + λ₂²(n_yy w_y),y
    − cw·w + cs·(w_xx + λ₂² w_yy) + f
(4) m_xx,x + λ₂ m_xy,y − λ₁ q_x
(5) m_xy,x + λ₂ m_yy,y − λ₁ q_y
```

- **Load** `f = q·a²/(h·A110)`.
- **Foundation** `cw = k_w·a²/A110 = Kw/(12λ₁²)`, `cs = k_s/A110 = Ks/(12λ₁²)`.

These follow from dividing the dimensional `W`-equation by `A110·h/a²` (and the in-plane /
moment equations by the corresponding factors). `k_w, k_s` use the pure-copper `D0`
(`k_w = Kw·D0/a⁴`, `k_s = Ks·D0/a²`, `params_q3.foundation_dimensional`).

## 5. Weak / energy form (energy-ODL, FEM)

The weak forms minimize the **potential energy**, not the collocation residual, so their
nondim coefficients differ from the strong form. Writing the nondim strain energy with the
`[0,1]²` quadrature and the scaled stiffness gives, per §3,

```
E_nd = E_dim / (A110 · a · b)
```

(strain-energy density consistently `1/A110`; the `[0,1]` integral is `1/(ab)×` the `[0,a]×[0,b]`
integral). `E_nd` is a positive constant multiple of `E_dim`, so its minimizer is the nondim
displacement. From this single scaling:

- **Load** `f = q·h/A110` (`= strong load / λ₁²`, since the strong divides the `W`-equation by
  `A110·h/a²` while the weak energy divides by `A110·a·b`).
- **Global-weak foundation** (`plate_weak.py::_build_weak_form_nondim`): energy
  `cw·∫w² + cs·∫(w_x² + λ₂² w_y²)` with `cw = k_w·h²/A110`, `cs = k_s·h²/(A110·a²)`.
- **FEM foundation** (`fem_plate_fsdt.py`, effective ops `dXe = (1/λ₁)dX_nd`,
  `dYe = (λ₂/λ₁)dY_nd`): `cw = k_w·h²/A110`, `cs = k_s/A110`. `cs` differs from the global-weak
  value by `λ₁²` because the FEM Pasternak term acts on `(dXe·w)²`, which already carries the
  `1/λ₁²`.

`plate_nonlinear.py` (energy-ODL residual/energy) reuses the strong nonlinear residual code
with the effective operators `Dxe = (1/λ₁)Dx_nd`, `Dye = (λ₂/λ₁)Dy_nd` and the scaled
stiffness — the torch residual/energy are **autograd-differentiated** by the LM/L-BFGS
optimizers (ODL is the autograd method; only the conventional DQ/FEM Newton use hand-coded
tangents).

## 6. nondim == dimensional equivalence

The nondim minimizer is `d_nd = disp_scale⁻¹ · d_dim`, where `disp_scale = h` on the
`u,v,w` blocks and `1` on the rotations. The physical displacement is recovered by
`to_physical(d, system)` (`direct_dq_plate.py`), which multiplies by `disp_scale` for a
nondim system and is a **no-op** for the dimensional path. Every runner converts to physical
*before* `extract_center_deflection`, so the reported `w/h` is identical for `NONDIM = True`
and `False`. (For the square benchmark `a = b = 1`, the `[0,1]` center `0.5` coincides with
`0.5·a`, so the center interpolation is unchanged.)

## 7. Verification (square benchmark `a = b = 1`, `h = 0.05`, `N = 15`)

| solver | check | result |
|---|---|---|
| DQ-Newton | `nondim == dim` w/h (linear & nonlinear) | `~1e-16` |
| DQ-Newton | hand tangent vs central FD | `~1e-11` |
| strong-ODL | torch residual vs numpy `_residual_full_nd` | `~1e-16` |
| energy-ODL | linear weak `nondim == dim` | `~1e-8` |
| energy-ODL | nonlinear `nondim == dim` | `~1e-5` (LM optimiser floor) |
| FEM | element residual/tangent vs FD | `~1e-11` |
| FEM | `nondim == dim` linear | `~1e-14` |
| FEM | dense vs sparse assembly | `~1e-15` |
| FEM | `nondim == dim` nonlinear | `~5e-7` (NR floor) |
| runners | DQ+ODL `to_physical → extract → /h`, `nondim == dim` | DQ exact, ODL `~1e-5` |

The optimiser-floor agreements (energy-ODL `~1e-5`, FEM NR `~5e-7`) are the iterative
solvers' own convergence floors, well inside the 4-solver cross-check tolerance; the exact
(Newton/linear-solve) paths agree to machine precision.

Full four-solver comparison (`compare_plate.py`, NONDIM default, dense FEM nx=32): spread
**≤ 9.2e-4** (dominated by the FEM `O(h²)` mesh error), with the **three spectral solvers
(strong-ODL, energy-ODL, DQ) agreeing to ~4e-5**. The soft-BC strong-ODL variant
(`build_strong_vk_residual_soft`) reproduces the hard-BC solution (LM/GN → 0.3498 CCCC /
0.7958 SSSS) via its explicit nondim λ-branch.

## 8. Conditioning

Nondimensionalizing balances the residual rows from `~1e5–1e12` to `O(0.02–1.5)`, so the
Gauss–Newton / LM normal matrix `JᵀJ` is well-conditioned and the strong-ODL / energy-ODL
converge cleanly. L-BFGS on the *squared* collocation residual inherits `cond²` regardless of
scaling and stays pathological; nondim balances its
rows but cannot remove the squaring.

## 9. Known limitations

Two items are outside the benchmark scope (CCCC/SSSS, `T = 300 K → ΔT = 0`):

1. **FREE-edge natural BC.** The nondim natural rows enforce the bare resultants
   (`n_xx = 0`, `m_xx = 0`, …), which is exact for SSSS. The dimensional path additionally adds
   the FREE-edge effective-shear and Pasternak (`k_s ∂w/∂n`) natural terms (`plate_strong.py`,
   comp 7/8); these are not mirrored in the nondim path. The benchmark uses only CCCC/SSSS.
2. **Thermal resultants in the FEM.** The DQ/ODL paths scale the thermal resultants
   (`NT/A110`, `MT/(h·A110)`); the FEM element (`_elem_residual`) omits `NT/MT`, matching its
   original bending-only scope. Inert for `ΔT = 0` (the benchmark); for `ΔT ≠ 0` the FEM would
   need the thermal term to stay consistent with the other three solvers.
