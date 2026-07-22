



import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from material_plate import compute_A110_reference, compute_D0_reference


def _mesh(nx, ny):
    def nid(i, j):
        return i * (ny + 1) + j
    nnode = (nx + 1) * (ny + 1)
    conn = np.empty((nx * ny, 4), dtype=np.int64)
    e = 0
    for i in range(nx):
        for j in range(ny):
            conn[e] = (nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)); e += 1
    return nnode, conn, nid


def _gauss_data(hx, hy):
    g = 1.0 / np.sqrt(3.0)
    full = [(-g, -g), (g, -g), (g, g), (-g, g)]

    def shp(xi, eta):
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return N, (2.0 / hx) * dxi, (2.0 / hy) * deta
    Nf, dXf, dYf = [], [], []
    for (xi, eta) in full:
        N, dx, dy = shp(xi, eta); Nf.append(N); dXf.append(dx); dYf.append(dy)
    Nr, dXr, dYr = shp(0.0, 0.0)
    return (np.array(Nf), np.array(dXf), np.array(dYf), np.ones(4), Nr, dXr, dYr, 4.0, hx * hy / 4.0)


def _elem_residual(de, q, K):
    u = de[0:4]; v = de[4:8]; w = de[8:12]; px = de[12:16]; py = de[16:20]
    Nf, dXf, dYf, fw = K["Nf"], K["dXf"], K["dYf"], K["fw"]
    Nr, dXr, dYr, wred, detJ = K["Nr"], K["dXr"], K["dYr"], K["wred"], K["detJ"]
    C6, As, kw, ks, nl = K["C6"], K["As"], K["k_w"], K["k_s"], K["nl"]
    R = np.zeros(20)
    for g in range(4):
        dX, dY, N = dXf[g], dYf[g], Nf[g]
        wx = dX @ w; wy = dY @ w; wv = N @ w
        ex = dX @ u + (0.5 * wx * wx if nl else 0.0)
        ey = dY @ v + (0.5 * wy * wy if nl else 0.0)
        gxy = dY @ u + dX @ v + (wx * wy if nl else 0.0)
        s = C6 @ np.array([ex, ey, gxy, dX @ px, dY @ py, dY @ px + dX @ py])
        c = fw[g] * detJ
        R[0:4] += (s[0] * dX + s[2] * dY) * c
        R[4:8] += (s[1] * dY + s[2] * dX) * c
        R[12:16] += (s[3] * dX + s[5] * dY) * c
        R[16:20] += (s[4] * dY + s[5] * dX) * c
        Rw = kw * wv * N + ks * (wx * dX + wy * dY) - q * N
        if nl:
            Rw = Rw + s[0] * wx * dX + s[1] * wy * dY + s[2] * (wy * dX + wx * dY)
        R[8:12] += Rw * c
    q2 = As @ np.array([dXr @ w + Nr @ px, dYr @ w + Nr @ py])
    cs = wred * detJ
    R[8:12] += (q2[0] * dXr + q2[1] * dYr) * cs
    R[12:16] += q2[0] * Nr * cs
    R[16:20] += q2[1] * Nr * cs
    return R


def _elem_tangent(de, K):
    u = de[0:4]; v = de[4:8]; w = de[8:12]; px = de[12:16]; py = de[16:20]
    Nf, dXf, dYf, fw = K["Nf"], K["dXf"], K["dYf"], K["fw"]
    Nr, dXr, dYr, wred, detJ = K["Nr"], K["dXr"], K["dYr"], K["wred"], K["detJ"]
    C6, As, kw, ks, nl = K["C6"], K["As"], K["k_w"], K["k_s"], K["nl"]
    Kt = np.zeros((20, 20))
    for g in range(4):
        dX, dY, N = dXf[g], dYf[g], Nf[g]
        wx = dX @ w; wy = dY @ w
        B = np.zeros((6, 20))
        B[0, 0:4] = dX; B[2, 0:4] = dY
        B[1, 4:8] = dY; B[2, 4:8] = dX
        B[3, 12:16] = dX; B[5, 12:16] = dY
        B[4, 16:20] = dY; B[5, 16:20] = dX
        if nl:
            B[0, 8:12] = wx * dX; B[1, 8:12] = wy * dY; B[2, 8:12] = wy * dX + wx * dY
        c = fw[g] * detJ
        Kt += (B.T @ C6 @ B) * c
        Kt[8:12, 8:12] += (kw * np.outer(N, N) + ks * (np.outer(dX, dX) + np.outer(dY, dY))) * c
        if nl:
            ex = dX @ u + 0.5 * wx * wx; ey = dY @ v + 0.5 * wy * wy
            gxy = dY @ u + dX @ v + wx * wy
            s = C6 @ np.array([ex, ey, gxy, dX @ px, dY @ py, dY @ px + dX @ py])
            Kt[8:12, 8:12] += (s[0] * np.outer(dX, dX) + s[1] * np.outer(dY, dY)
                               + s[2] * (np.outer(dX, dY) + np.outer(dY, dX))) * c
    Bs = np.zeros((2, 20))
    Bs[0, 8:12] = dXr; Bs[0, 12:16] = Nr
    Bs[1, 8:12] = dYr; Bs[1, 16:20] = Nr
    Kt += (Bs.T @ As @ Bs) * (wred * detJ)
    return Kt


_FIELD_IDX = {"U": 0, "V": 1, "W": 2, "phix": 3, "phiy": 4}


def _edge_essential(bc_type, convention, edge_normal):
    bc = str(bc_type).strip().upper(); conv = str(convention).strip().lower()
    tdisp, trot = ("V", "phiy") if edge_normal == "x" else ("U", "phix")
    if bc in ("FREE", "F-F"):
        return []
    if bc in ("CC", "C-C"):
        return ["U", "V", "W", "phix", "phiy"]
    if bc in ("SS", "S-S", "HH", "H-H"):
        return [tdisp, "W", trot] if conv == "ss1" else ["W", trot]
    raise ValueError(f"unsupported boundary condition: {bc_type}")


def _essential_dofs(nx, ny, nnode, nid, x_bc, y_bc, x_conv, y_conv):
    x_ess = set(_edge_essential(x_bc, x_conv, "x")); y_ess = set(_edge_essential(y_bc, y_conv, "y"))
    fixed = set()
    for i in range(nx + 1):
        for j in range(ny + 1):
            on_x = i in (0, nx); on_y = j in (0, ny)
            if not (on_x or on_y):
                continue
            ess = set()
            if on_x:
                ess |= x_ess
            if on_y:
                ess |= y_ess
            for fld in ess:
                fixed.add(_FIELD_IDX[fld] * nnode + nid(i, j))
    return np.array(sorted(fixed), dtype=np.int64)


class FEMPlate:
    def __init__(self, nx, ny, material, a=1.0, b=1.0, k_w=0.0, k_s=0.0, q=-1.0,
                 nonlinear=True, h=None, nondim=False, sparse=False,
                 x_bc_type="CC", y_bc_type="CC", x_bc_convention="starter", y_bc_convention="starter"):
        self.nx, self.ny, self.a, self.b, self.sparse, self.nondim = nx, ny, a, b, bool(sparse), bool(nondim)
        self.nnode, conn, nid = _mesh(nx, ny)
        self.ndof = 5 * self.nnode
        A, B, D, As = material["A"], material["B"], material["D"], material["As"]
        if nondim:
            A110 = compute_A110_reference(h); lam1 = a / h; lam2 = a / b
            Nf, dXf, dYf, fw, Nr, dXr, dYr, wred, detJ = _gauss_data(1.0 / nx, 1.0 / ny)
            dXf = (1.0 / lam1) * dXf; dYf = (lam2 / lam1) * dYf
            dXr = (1.0 / lam1) * dXr; dYr = (lam2 / lam1) * dYr
            C6 = np.block([[A / A110, B / (h * A110)], [B / (h * A110), D / (h ** 2 * A110)]])
            As2 = As / A110
            self.qload = q * h / A110
            kw = k_w * h ** 2 / A110; ks = k_s / A110
            self.disp_scale = np.ones(self.ndof); self.disp_scale[0:3 * self.nnode] = h
        else:
            Nf, dXf, dYf, fw, Nr, dXr, dYr, wred, detJ = _gauss_data(a / nx, b / ny)
            C6 = np.block([[A, B], [B, D]]); As2 = As
            self.qload = q; kw = k_w; ks = k_s
            self.disp_scale = np.ones(self.ndof)
        self._K = {"Nf": Nf, "dXf": dXf, "dYf": dYf, "fw": fw, "Nr": Nr, "dXr": dXr, "dYr": dYr,
                   "wred": float(wred), "detJ": float(detJ), "C6": C6, "As": As2,
                   "k_w": float(kw), "k_s": float(ks), "nl": bool(nonlinear)}
        self.conn = conn; self.ne = conn.shape[0]
        gidx = np.empty((self.ne, 20), dtype=np.int64)
        for f in range(5):
            for n in range(4):
                gidx[:, f * 4 + n] = f * self.nnode + conn[:, n]
        self.gidx = gidx
        if self.sparse:
            self._rows = np.repeat(gidx, 20, axis=1).reshape(-1)
            self._cols = np.tile(gidx, (1, 20)).reshape(-1)
        fixed = _essential_dofs(nx, ny, self.nnode, nid, x_bc_type, y_bc_type, x_bc_convention, y_bc_convention)
        mask = np.ones(self.ndof, dtype=bool); mask[fixed] = False
        self.free = np.where(mask)[0]
        self.center_dof = 2 * self.nnode + nid(nx // 2, ny // 2)

    def residual(self, d, ls=1.0):
        q = ls * self.qload
        R = np.zeros(self.ndof)
        for e in range(self.ne):
            g = self.gidx[e]
            R[g] += _elem_residual(d[g], q, self._K)
        return R

    def tangent(self, d):
        if self.sparse:
            vals = np.empty(self.ne * 400)
            for e in range(self.ne):
                vals[e * 400:(e + 1) * 400] = _elem_tangent(d[self.gidx[e]], self._K).reshape(-1)
            return sp.coo_matrix((vals, (self._rows, self._cols)), shape=(self.ndof, self.ndof)).tocsr()
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.ne):
            g = self.gidx[e]
            K[np.ix_(g, g)] += _elem_tangent(d[g], self._K)
        return K

    def _solve(self, A, rhs):
        free = self.free
        if self.sparse:
            return spsolve(A[free][:, free].tocsc(), rhs[free])
        return np.linalg.solve(A[np.ix_(free, free)], rhs[free])

    def to_physical(self, d):
        return self.disp_scale * d


def solve_linear(nx, ny, material, a=1.0, b=1.0, k_w=0.0, k_s=0.0, q=-1.0,
                 h=None, nondim=False, sparse=False,
                 x_bc_type="CC", y_bc_type="CC", x_bc_convention="starter", y_bc_convention="starter"):
    t0 = time.time()
    fem = FEMPlate(nx, ny, material, a, b, k_w, k_s, q, nonlinear=False, h=h, nondim=nondim, sparse=sparse,
                   x_bc_type=x_bc_type, y_bc_type=y_bc_type,
                   x_bc_convention=x_bc_convention, y_bc_convention=y_bc_convention)
    d = np.zeros(fem.ndof)
    R0 = fem.residual(d); K = fem.tangent(d)
    d[fem.free] = fem._solve(K, -R0)
    R = fem.residual(d)
    d_phys = fem.to_physical(d)
    return {"w_center": d_phys[fem.center_dof], "d": d_phys, "ndof": fem.ndof,
            "residual": float(np.max(np.abs(R[fem.free]))), "elapsed_s": time.time() - t0, "fem": fem}


def solve_nonlinear(nx, ny, material, a=1.0, b=1.0, k_w=0.0, k_s=0.0, q=-1.0,
                    load_steps=1, rtol=1.0e-10, max_iter=50, h=None, nondim=False, sparse=False, verbose=False,
                    x_bc_type="CC", y_bc_type="CC", x_bc_convention="starter", y_bc_convention="starter"):
    t0 = time.time()
    fem = FEMPlate(nx, ny, material, a, b, k_w, k_s, q, nonlinear=True, h=h, nondim=nondim, sparse=sparse,
                   x_bc_type=x_bc_type, y_bc_type=y_bc_type,
                   x_bc_convention=x_bc_convention, y_bc_convention=y_bc_convention)
    free = fem.free
    f_scale = max(abs(fem.qload), 1.0)
    d = np.zeros(fem.ndof)
    R0 = fem.residual(d); K0 = fem.tangent(d)
    try:
        d[free] = fem._solve(K0, -R0)
    except Exception:
        pass
    total_iter = 0
    res_history = []
    for ls in np.linspace(1.0 / load_steps, 1.0, load_steps):
        thresh = rtol * max(abs(ls) * f_scale, 1.0); prev = np.inf
        for _ in range(max_iter):
            R = fem.residual(d, ls); rnorm = np.max(np.abs(R[free]))
            res_history.append(float(rnorm))
            if rnorm < thresh or rnorm > (1.0 - 1.0e-10) * prev:
                break
            prev = rnorm
            delta = fem._solve(fem.tangent(d), -R)
            alpha = 1.0; dvec = np.zeros(fem.ndof)
            for _ls_it in range(30):
                dvec[free] = alpha * delta
                if np.max(np.abs(fem.residual(d + dvec, ls)[free])) < rnorm or alpha < 1.0e-10:
                    break
                alpha *= 0.5
            d = d + dvec; total_iter += 1
        if verbose:
            print("    [FEM-Newton] ls=%.3f max|R_f|=%.3e rel=%.2e iters=%d"
                  % (ls, rnorm, rnorm / f_scale, total_iter))
    rnorm = float(np.max(np.abs(fem.residual(d)[free])))
    d_phys = fem.to_physical(d)
    return {"w_center": d_phys[fem.center_dof], "d": d_phys, "ndof": fem.ndof,
            "residual_inf": rnorm, "residual_rel": rnorm / f_scale,
            "converged": bool(rnorm / f_scale < 1.0e-8), "residual_history": res_history,
            "iterations": total_iter, "load_steps": int(load_steps), "elapsed_s": time.time() - t0}
