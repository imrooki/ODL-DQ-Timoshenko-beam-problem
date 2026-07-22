




import time

import numpy as np

from plate_strong import build_strong_form_plate_system, strong_boundary_map
from material_plate import compute_A110_reference

__all__ = [
    "build_strong_form_plate_system",
    "solve_bending_linear_dq",
    "solve_bending_nonlinear_dq",
    "check_tangent_fd",
    "build_strong_vk_residual",
    "build_strong_vk_residual_soft",
]


def _prep(system):
    Ng = system["Ng"]
    ops = system["ops"]
    mat = system["material"]
    A, B, D, As = mat["A"], mat["B"], mat["D"], mat["As"]
    C6 = np.block([[A, B], [B, D]])
    thermal6 = np.concatenate([mat.get("NT", np.zeros(3)), mat.get("MT", np.zeros(3))])
    cfg = system["config"]
    k_w = cfg.get("k_w", 0.0)
    k_s = cfg.get("k_s", 0.0)
    FoundW = -k_w * np.eye(Ng) + k_s * (ops["Dxx"] + ops["Dyy"])
    _, natural = strong_boundary_map(cfg, system["Nx"], system["Ny"])
    Pd = {
        "Ng": Ng,
        "Dx": ops["Dx"], "Dy": ops["Dy"],
        "C6": C6, "As2": As, "thermal6": thermal6,
        "FoundW": FoundW, "k_s": k_s, "natural": natural,
        "T": system["transform"], "f_full": system["f_full"],
    }
    if not cfg.get("NONDIM"):
        return Pd
    lam1 = system["lam1"]; lam2 = system["lam2"]; A110 = system["A110"]
    a = cfg["a"]; h = cfg["h"]
    cw = k_w * a ** 2 / A110
    cs = k_s / A110
    FoundW_nd = -cw * np.eye(Ng) + cs * (ops["Dxx"] + lam2 ** 2 * ops["Dyy"])
    NT = mat.get("NT", np.zeros(3)); MT = mat.get("MT", np.zeros(3))
    thermal6_nd = np.concatenate([NT / A110, MT / (h * A110)])
    disp_scale = np.ones(5 * Ng); disp_scale[0:3 * Ng] = h
    return {
        "Ng": Ng, "Dx": ops["Dx"], "Dy": ops["Dy"],
        "C6": system["C6_nd"], "As2": system["As2_nd"], "thermal6": thermal6_nd,
        "FoundW": FoundW_nd, "k_s": k_s, "natural": natural,
        "T": system["transform"], "f_full": system["f_full"],
        "nondim": True, "lam1": lam1, "lam2": lam2, "disp_scale": disp_scale,
    }


def _kinematics(d_full, P):
    Ng, Dx, Dy = P["Ng"], P["Dx"], P["Dy"]
    U = d_full[0:Ng]; V = d_full[Ng:2 * Ng]; W = d_full[2 * Ng:3 * Ng]
    Fx = d_full[3 * Ng:4 * Ng]; Fy = d_full[4 * Ng:5 * Ng]
    Wx = Dx @ W; Wy = Dy @ W
    ex = Dx @ U + 0.5 * Wx * Wx
    ey = Dy @ V + 0.5 * Wy * Wy
    gxy = Dy @ U + Dx @ V + Wx * Wy
    kx = Dx @ Fx; ky = Dy @ Fy; kxy = Dy @ Fx + Dx @ Fy
    gxz = Wx + Fx; gyz = Wy + Fy
    res6 = np.stack([ex, ey, gxy, kx, ky, kxy], axis=1) @ P["C6"].T - P["thermal6"]
    resS = np.stack([gxz, gyz], axis=1) @ P["As2"].T
    Nx, Ny, Nxy = res6[:, 0], res6[:, 1], res6[:, 2]
    Qx, Qy = resS[:, 0], resS[:, 1]
    return {
        "W": W, "Wx": Wx, "Wy": Wy,
        "Nx": Nx, "Ny": Ny, "Nxy": Nxy,
        "Mx": res6[:, 3], "My": res6[:, 4], "Mxy": res6[:, 5],
        "Qx": Qx, "Qy": Qy,
        "TwX": Qx + Nx * Wx + Nxy * Wy,
        "TwY": Qy + Nxy * Wx + Ny * Wy,
    }


def _residual_full(d_full, P, load_scale=1.0):
    if P.get("nondim"):
        return _residual_full_nd(d_full, P, load_scale)
    Ng, Dx, Dy = P["Ng"], P["Dx"], P["Dy"]
    k = _kinematics(d_full, P)
    Nx, Ny, Nxy = k["Nx"], k["Ny"], k["Nxy"]
    Mx, My, Mxy = k["Mx"], k["My"], k["Mxy"]
    Qx, Qy, Wx, Wy = k["Qx"], k["Qy"], k["Wx"], k["Wy"]
    Fint = np.empty(5 * Ng)
    Fint[0:Ng]        = Dx @ Nx + Dy @ Nxy
    Fint[Ng:2 * Ng]   = Dx @ Nxy + Dy @ Ny
    Fint[2 * Ng:3 * Ng] = (Dx @ Qx + Dy @ Qy
                           + Dx @ (Nx * Wx + Nxy * Wy)
                           + Dy @ (Nxy * Wx + Ny * Wy)
                           + P["FoundW"] @ k["W"])
    Fint[3 * Ng:4 * Ng] = Dx @ Mx + Dy @ Mxy - Qx
    Fint[4 * Ng:5 * Ng] = Dx @ Mxy + Dy @ My - Qy
    R_full = -Fint - load_scale * P["f_full"]
    if P["natural"]:
        res = [Nx, Ny, Nxy, Mx, My, Mxy, k["TwX"], k["TwY"]]
        k_s = P["k_s"]
        for dof, comp, node in P["natural"]:
            v = res[comp - 1][node]
            if comp == 7:
                v = v + k_s * Wx[node]
            elif comp == 8:
                v = v + k_s * Wy[node]
            R_full[dof] = v
    return R_full


def _tangent_full(d_full, P):
    if P.get("nondim"):
        return _tangent_full_nd(d_full, P)
    Ng, Dx, Dy = P["Ng"], P["Dx"], P["Dy"]
    C6, As2, FoundW = P["C6"], P["As2"], P["FoundW"]
    k = _kinematics(d_full, P)
    Wx, Wy = k["Wx"], k["Wy"]
    Nx, Ny, Nxy = k["Nx"], k["Ny"], k["Nxy"]
    I = np.eye(Ng)
    Z = np.zeros((Ng, Ng))

    def dscale(v, M):
        return v[:, None] * M

    def Bvec(U=Z, V=Z, W=Z, Fx=Z, Fy=Z):
        return [U, V, W, Fx, Fy]

    B_ex  = Bvec(U=Dx, W=dscale(Wx, Dx))
    B_ey  = Bvec(V=Dy, W=dscale(Wy, Dy))
    B_gxy = Bvec(U=Dy, V=Dx, W=dscale(Wy, Dx) + dscale(Wx, Dy))
    B_kx  = Bvec(Fx=Dx)
    B_ky  = Bvec(Fy=Dy)
    B_kxy = Bvec(Fx=Dy, Fy=Dx)
    B_strain = [B_ex, B_ey, B_gxy, B_kx, B_ky, B_kxy]

    B_gxz = Bvec(W=Dx, Fx=I)
    B_gyz = Bvec(W=Dy, Fy=I)
    B_shear = [B_gxz, B_gyz]

    def comb(coeff_row, B_list):
        out = [Z.copy() for _ in range(5)]
        for j, Bj in enumerate(B_list):
            c = coeff_row[j]
            if c == 0.0:
                continue
            for f in range(5):
                out[f] = out[f] + c * Bj[f]
        return out

    dNx = comb(C6[0], B_strain); dNy = comb(C6[1], B_strain); dNxy = comb(C6[2], B_strain)
    dMx = comb(C6[3], B_strain); dMy = comb(C6[4], B_strain); dMxy = comb(C6[5], B_strain)
    dQx = comb(As2[0], B_shear); dQy = comb(As2[1], B_shear)

    K = np.zeros((5 * Ng, 5 * Ng))
    for f in range(5):
        dFU  = Dx @ dNx[f] + Dy @ dNxy[f]
        dFV  = Dx @ dNxy[f] + Dy @ dNy[f]
        dFFx = Dx @ dMx[f] + Dy @ dMxy[f] - dQx[f]
        dFFy = Dx @ dMxy[f] + Dy @ dMy[f] - dQy[f]
        dFW = (Dx @ dQx[f] + Dy @ dQy[f]
               + Dx @ (dscale(Wx, dNx[f]) + dscale(Wy, dNxy[f]))
               + Dy @ (dscale(Wx, dNxy[f]) + dscale(Wy, dNy[f])))
        if f == 2:
            dFW = (dFW
                   + Dx @ (dscale(Nx, Dx) + dscale(Nxy, Dy))
                   + Dy @ (dscale(Nxy, Dx) + dscale(Ny, Dy))
                   + FoundW)
        K[0 * Ng:1 * Ng, f * Ng:(f + 1) * Ng] = -dFU
        K[1 * Ng:2 * Ng, f * Ng:(f + 1) * Ng] = -dFV
        K[2 * Ng:3 * Ng, f * Ng:(f + 1) * Ng] = -dFW
        K[3 * Ng:4 * Ng, f * Ng:(f + 1) * Ng] = -dFFx
        K[4 * Ng:5 * Ng, f * Ng:(f + 1) * Ng] = -dFFy

    if P["natural"]:
        dRes6 = [dNx, dNy, dNxy, dMx, dMy, dMxy]
        k_s = P["k_s"]
        for dof, comp, node in P["natural"]:
            row = np.zeros(5 * Ng)
            if comp <= 6:
                blocks = dRes6[comp - 1]
                for f in range(5):
                    row[f * Ng:(f + 1) * Ng] = blocks[f][node, :]
            else:
                if comp == 7:
                    dQ, dNa, dNb, Na, Nb, dWself = dQx, dNx, dNxy, Nx[node], Nxy[node], Dx
                else:
                    dQ, dNa, dNb, Na, Nb, dWself = dQy, dNxy, dNy, Nxy[node], Ny[node], Dy
                for f in range(5):
                    row[f * Ng:(f + 1) * Ng] = (dQ[f][node, :]
                                                + Wx[node] * dNa[f][node, :]
                                                + Wy[node] * dNb[f][node, :])
                row[2 * Ng:3 * Ng] += Na * Dx[node, :] + Nb * Dy[node, :] + k_s * dWself[node, :]
            K[dof, :] = row
    return K


def _kinematics_nd(d_full, P):
    Ng, Dx, Dy = P["Ng"], P["Dx"], P["Dy"]
    lam1, lam2 = P["lam1"], P["lam2"]
    u = d_full[0:Ng]; v = d_full[Ng:2 * Ng]; w = d_full[2 * Ng:3 * Ng]
    px = d_full[3 * Ng:4 * Ng]; py = d_full[4 * Ng:5 * Ng]
    wx = Dx @ w; wy = Dy @ w
    exx = (1.0 / lam1) * (Dx @ u + (1.0 / (2.0 * lam1)) * wx * wx)
    eyy = (lam2 / lam1) * (Dy @ v + (lam2 / (2.0 * lam1)) * wy * wy)
    gxy = (1.0 / lam1) * (lam2 * (Dy @ u) + (Dx @ v) + (lam2 / lam1) * wx * wy)
    kxx = (1.0 / lam1) * (Dx @ px)
    kyy = (lam2 / lam1) * (Dy @ py)
    kxy = (1.0 / lam1) * (lam2 * (Dy @ px) + (Dx @ py))
    gxz = (1.0 / lam1) * wx + px
    gyz = (lam2 / lam1) * wy + py
    res6 = np.stack([exx, eyy, gxy, kxx, kyy, kxy], axis=1) @ P["C6"].T - P["thermal6"]
    resS = np.stack([gxz, gyz], axis=1) @ P["As2"].T
    Nx, Ny, Nxy = res6[:, 0], res6[:, 1], res6[:, 2]
    Qx, Qy = resS[:, 0], resS[:, 1]
    return {"W": w, "wx": wx, "wy": wy, "Nx": Nx, "Ny": Ny, "Nxy": Nxy,
            "Mx": res6[:, 3], "My": res6[:, 4], "Mxy": res6[:, 5], "Qx": Qx, "Qy": Qy}


def _residual_full_nd(d_full, P, load_scale=1.0):
    Ng, Dx, Dy = P["Ng"], P["Dx"], P["Dy"]
    lam1, lam2 = P["lam1"], P["lam2"]
    k = _kinematics_nd(d_full, P)
    Nx, Ny, Nxy = k["Nx"], k["Ny"], k["Nxy"]
    Mx, My, Mxy = k["Mx"], k["My"], k["Mxy"]
    Qx, Qy, wx, wy = k["Qx"], k["Qy"], k["wx"], k["wy"]
    Fint = np.empty(5 * Ng)
    Fint[0:Ng]        = Dx @ Nx + lam2 * (Dy @ Nxy)
    Fint[Ng:2 * Ng]   = Dx @ Nxy + lam2 * (Dy @ Ny)
    Fint[2 * Ng:3 * Ng] = (lam1 * (Dx @ Qx) + lam1 * lam2 * (Dy @ Qy)
                           + Dx @ (Nx * wx) + lam2 * (Dx @ (Nxy * wy))
                           + lam2 * (Dy @ (Nxy * wx)) + lam2 ** 2 * (Dy @ (Ny * wy))
                           + P["FoundW"] @ k["W"])
    Fint[3 * Ng:4 * Ng] = Dx @ Mx + lam2 * (Dy @ Mxy) - lam1 * Qx
    Fint[4 * Ng:5 * Ng] = Dx @ Mxy + lam2 * (Dy @ My) - lam1 * Qy
    R_full = -Fint - load_scale * P["f_full"]
    if P["natural"]:
        res = [Nx, Ny, Nxy, Mx, My, Mxy, Qx, Qy]
        for dof, comp, node in P["natural"]:
            R_full[dof] = res[comp - 1][node]
    return R_full


def _tangent_full_nd(d_full, P):
    Ng, Dx, Dy = P["Ng"], P["Dx"], P["Dy"]
    C6, As2, FoundW = P["C6"], P["As2"], P["FoundW"]
    lam1, lam2 = P["lam1"], P["lam2"]
    k = _kinematics_nd(d_full, P)
    wx, wy = k["wx"], k["wy"]
    Nx, Ny, Nxy = k["Nx"], k["Ny"], k["Nxy"]
    I = np.eye(Ng); Z = np.zeros((Ng, Ng))

    def dscale(vv, M):
        return vv[:, None] * M

    def Bvec(U=Z, V=Z, W=Z, Fx=Z, Fy=Z):
        return [U, V, W, Fx, Fy]

    B_ex  = Bvec(U=(1.0 / lam1) * Dx, W=(1.0 / lam1 ** 2) * dscale(wx, Dx))
    B_ey  = Bvec(V=(lam2 / lam1) * Dy, W=(lam2 ** 2 / lam1 ** 2) * dscale(wy, Dy))
    B_gxy = Bvec(U=(lam2 / lam1) * Dy, V=(1.0 / lam1) * Dx,
                 W=(lam2 / lam1 ** 2) * (dscale(wy, Dx) + dscale(wx, Dy)))
    B_kx  = Bvec(Fx=(1.0 / lam1) * Dx)
    B_ky  = Bvec(Fy=(lam2 / lam1) * Dy)
    B_kxy = Bvec(Fx=(lam2 / lam1) * Dy, Fy=(1.0 / lam1) * Dx)
    B_strain = [B_ex, B_ey, B_gxy, B_kx, B_ky, B_kxy]
    B_gxz = Bvec(W=(1.0 / lam1) * Dx, Fx=I)
    B_gyz = Bvec(W=(lam2 / lam1) * Dy, Fy=I)
    B_shear = [B_gxz, B_gyz]

    def comb(coeff_row, B_list):
        out = [Z.copy() for _ in range(5)]
        for j, Bj in enumerate(B_list):
            c = coeff_row[j]
            if c == 0.0:
                continue
            for f in range(5):
                out[f] = out[f] + c * Bj[f]
        return out

    dNx = comb(C6[0], B_strain); dNy = comb(C6[1], B_strain); dNxy = comb(C6[2], B_strain)
    dMx = comb(C6[3], B_strain); dMy = comb(C6[4], B_strain); dMxy = comb(C6[5], B_strain)
    dQx = comb(As2[0], B_shear); dQy = comb(As2[1], B_shear)

    K = np.zeros((5 * Ng, 5 * Ng))
    for f in range(5):
        dFU  = Dx @ dNx[f] + lam2 * (Dy @ dNxy[f])
        dFV  = Dx @ dNxy[f] + lam2 * (Dy @ dNy[f])
        dFFx = Dx @ dMx[f] + lam2 * (Dy @ dMxy[f]) - lam1 * dQx[f]
        dFFy = Dx @ dMxy[f] + lam2 * (Dy @ dMy[f]) - lam1 * dQy[f]
        dFW = (lam1 * (Dx @ dQx[f]) + lam1 * lam2 * (Dy @ dQy[f])
               + Dx @ dscale(wx, dNx[f]) + lam2 * (Dx @ dscale(wy, dNxy[f]))
               + lam2 * (Dy @ dscale(wx, dNxy[f])) + lam2 ** 2 * (Dy @ dscale(wy, dNy[f])))
        if f == 2:
            dFW = (dFW
                   + Dx @ dscale(Nx, Dx) + lam2 * (Dx @ dscale(Nxy, Dy))
                   + lam2 * (Dy @ dscale(Nxy, Dx)) + lam2 ** 2 * (Dy @ dscale(Ny, Dy))
                   + FoundW)
        K[0 * Ng:1 * Ng, f * Ng:(f + 1) * Ng] = -dFU
        K[1 * Ng:2 * Ng, f * Ng:(f + 1) * Ng] = -dFV
        K[2 * Ng:3 * Ng, f * Ng:(f + 1) * Ng] = -dFW
        K[3 * Ng:4 * Ng, f * Ng:(f + 1) * Ng] = -dFFx
        K[4 * Ng:5 * Ng, f * Ng:(f + 1) * Ng] = -dFFy

    if P["natural"]:
        dRes = [dNx, dNy, dNxy, dMx, dMy, dMxy, dQx, dQy]
        for dof, comp, node in P["natural"]:
            blocks = dRes[comp - 1]
            row = np.zeros(5 * Ng)
            for f in range(5):
                row[f * Ng:(f + 1) * Ng] = blocks[f][node, :]
            K[dof, :] = row
    return K


def _linear_q(system):
    return np.linalg.solve(system["KL_red"], system["f_red"])


def _initial_guess(system, P, init, seed, scale):
    n_free = P["T"].shape[1]
    if init == "zero":
        return np.zeros(n_free)
    q_lin = _linear_q(system)
    if init == "linear":
        return q_lin.copy()
    ref = np.max(np.abs(q_lin))
    if ref == 0.0:
        ref = 1.0
    rng = np.random.default_rng(seed)
    if init == "random":
        return scale * ref * (2.0 * rng.random(n_free) - 1.0)
    if init == "gauss":
        return scale * ref * rng.standard_normal(n_free)
    raise ValueError(f"unknown init guess: {init}")


def to_physical(d_full, system):
    if not system.get("nondim"):
        return d_full
    Ng = len(d_full) // 5
    cfg = system.get("config", {})
    h = cfg.get("h", system.get("h"))
    s = np.ones(5 * Ng); s[0:3 * Ng] = h
    return s * d_full


def solve_bending_linear_dq(system):
    t0 = time.time()
    q = np.linalg.solve(system["KL_red"], system["f_red"])
    d = to_physical(system["transform"] @ q, system)
    res = float(np.linalg.norm(system["KL_red"] @ q - system["f_red"]))
    return {
        "displacement_full": d, "q_red": q,
        "residual_norm": res, "elapsed_s": time.time() - t0,
    }


def solve_bending_nonlinear_dq(system, load_steps=1, init="linear", init_seed=0,
                               init_scale=0.5, rtol=1.0e-10, max_iter=50, verbose=False):

    P = _prep(system)
    T = P["T"]
    nondim = bool(P.get("nondim"))
    f_scale = max(float(np.max(np.abs(system["f_red"]))), 1.0)
    t0 = time.time()
    q = _initial_guess(system, P, init, init_seed, init_scale)
    schedule = np.linspace(1.0 / load_steps, 1.0, load_steps)
    total_iter = 0
    res_history = []
    for ls in schedule:
        thresh = rtol
        prev = np.inf
        for _ in range(max_iter):
            d = T @ q
            R = T.T @ _residual_full(d, P, ls)
            rnorm = float(np.linalg.norm(R))
            res_history.append(float(rnorm))
            if rnorm < thresh:
                break
            if rnorm > (1.0 - 1.0e-10) * prev:
                break
            prev = rnorm
            K = T.T @ _tangent_full(d, P) @ T
            try:
                delta = np.linalg.solve(K, -R)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(K, -R, rcond=None)[0]
            alpha = 1.0
            q_try = q + delta
            for _ls_it in range(30):
                q_try = q + alpha * delta
                if float(np.linalg.norm(T.T @ _residual_full(T @ q_try, P, ls))) < rnorm or alpha < 1.0e-10:
                    break
                alpha *= 0.5
            q = q_try
            total_iter += 1
        if verbose:
            print(f"    [DQ-Newton] load_scale={ls:.3f}  ||R||_2={rnorm:.3e}  iters={total_iter}")
    d = T @ q
    R_final = T.T @ _residual_full(d, P, 1.0)
    rnorm = float(np.linalg.norm(R_final))
    rinf = float(np.max(np.abs(R_final)))
    rel = rnorm / f_scale
    if nondim:
        d = P["disp_scale"] * d
    return {
        "displacement_full": d, "q_red": q,
        "residual_l2": rnorm, "residual_inf": rinf, "residual_rel": rel,
        "iterations": total_iter,
        "residual_history": res_history,
        "converged": bool(rnorm < 1.0e-10),
        "load_steps": int(load_steps), "init": init,
        "elapsed_s": time.time() - t0,
    }


def check_tangent_fd(system, eps=1.0e-6, seed=1, amp=0.3):
    P = _prep(system)
    T = P["T"]
    n = T.shape[1]
    q_lin = _linear_q(system)
    ref = np.max(np.abs(q_lin))
    if ref == 0.0:
        ref = 1.0
    rng = np.random.default_rng(seed)
    q = amp * ref * rng.standard_normal(n)
    Ka = T.T @ _tangent_full(T @ q, P) @ T
    Kfd = np.zeros((n, n))
    for j in range(n):
        s = eps * max(1.0, abs(q[j]))
        qp = q.copy(); qp[j] += s
        qm = q.copy(); qm[j] -= s
        Rp = T.T @ _residual_full(T @ qp, P, 1.0)
        Rm = T.T @ _residual_full(T @ qm, P, 1.0)
        Kfd[:, j] = (Rp - Rm) / (2.0 * s)
    denom = np.max(np.abs(Kfd))
    if denom == 0.0:
        denom = 1.0
    return float(np.max(np.abs(Ka - Kfd)) / denom)


def build_strong_vk_residual(system, device=None):
    import torch
    torch.set_default_dtype(torch.float64)
    if device is None:
        try:
            from odl_config import DEFAULT_DEVICE
            device = DEFAULT_DEVICE
        except Exception:
            device = "cpu"
    P = _prep(system)
    Ng = P["Ng"]

    def t(M):
        return torch.as_tensor(np.asarray(M, dtype=float), device=device)

    Dx = t(P["Dx"]); Dy = t(P["Dy"])
    C6 = t(P["C6"]); As2 = t(P["As2"]); FoundW = t(P["FoundW"])
    thermal6 = t(P["thermal6"])
    Tt = t(P["T"]); TTt = t(P["T"].T)
    f_full = t(P["f_full"])
    k_s = float(P["k_s"])
    natural = P["natural"]
    has_nat = bool(natural)
    if has_nat:
        free_dofs = np.asarray(system["free_dofs"])
        pos = {int(d): i for i, d in enumerate(free_dofs)}
        nat_comp = np.array([c for _, c, _ in natural], dtype=int)
        nat_node = np.array([n for _, _, n in natural], dtype=int)
        nat_free_idx = np.array([pos[int(d)] for d, _, _ in natural], dtype=int)
        n_free = free_dofs.size
        Smat = np.zeros((n_free, nat_comp.size))
        Smat[nat_free_idx, np.arange(nat_comp.size)] = 1.0
        S = t(Smat)
        comp_idx = torch.as_tensor(nat_comp - 1, device=device, dtype=torch.long)
        node_idx = torch.as_tensor(nat_node, device=device, dtype=torch.long)
        free_is_nat = torch.zeros(n_free, dtype=torch.bool, device=device)
        free_is_nat[torch.as_tensor(nat_free_idx, device=device, dtype=torch.long)] = True
        mask7 = t((nat_comp == 7).astype(float))
        mask8 = t((nat_comp == 8).astype(float))

    def residual(d_red, load_scale=1.0):
        d = Tt @ d_red
        U = d[0:Ng]; V = d[Ng:2 * Ng]; W = d[2 * Ng:3 * Ng]
        Fx = d[3 * Ng:4 * Ng]; Fy = d[4 * Ng:5 * Ng]
        Wx = Dx @ W; Wy = Dy @ W
        ex = Dx @ U + 0.5 * Wx * Wx
        ey = Dy @ V + 0.5 * Wy * Wy
        gxy = Dy @ U + Dx @ V + Wx * Wy
        kx = Dx @ Fx; ky = Dy @ Fy; kxy = Dy @ Fx + Dx @ Fy
        gxz = Wx + Fx; gyz = Wy + Fy
        res6 = torch.stack([ex, ey, gxy, kx, ky, kxy], dim=1) @ C6.T - thermal6
        Nx = res6[:, 0]; Ny = res6[:, 1]; Nxy = res6[:, 2]
        Mx = res6[:, 3]; My = res6[:, 4]; Mxy = res6[:, 5]
        resS = torch.stack([gxz, gyz], dim=1) @ As2.T
        Qx = resS[:, 0]; Qy = resS[:, 1]
        Fint_U = Dx @ Nx + Dy @ Nxy
        Fint_V = Dx @ Nxy + Dy @ Ny
        Fint_W = (Dx @ Qx + Dy @ Qy
                  + Dx @ (Nx * Wx + Nxy * Wy) + Dy @ (Nxy * Wx + Ny * Wy) + FoundW @ W)
        Fint_Fx = Dx @ Mx + Dy @ Mxy - Qx
        Fint_Fy = Dx @ Mxy + Dy @ My - Qy
        Fint = torch.cat([Fint_U, Fint_V, Fint_W, Fint_Fx, Fint_Fy])
        R_red = TTt @ (-Fint - load_scale * f_full)
        if has_nat:
            TwX = Qx + Nx * Wx + Nxy * Wy
            TwY = Qy + Nxy * Wx + Ny * Wy
            res_stack = torch.stack([Nx, Ny, Nxy, Mx, My, Mxy, TwX, TwY], dim=0)
            nat_vals = res_stack[comp_idx, node_idx] + k_s * (mask7 * Wx[node_idx] + mask8 * Wy[node_idx])
            R_red = torch.where(free_is_nat, S @ nat_vals, R_red)
        return R_red

    if not system["config"].get("NONDIM"):
        return residual
    lam1 = P["lam1"]; lam2 = P["lam2"]

    def residual_nd(d_red, load_scale=1.0):
        d = Tt @ d_red
        u = d[0:Ng]; v = d[Ng:2 * Ng]; w = d[2 * Ng:3 * Ng]
        px = d[3 * Ng:4 * Ng]; py = d[4 * Ng:5 * Ng]
        wx = Dx @ w; wy = Dy @ w
        exx = (1.0 / lam1) * (Dx @ u + (1.0 / (2.0 * lam1)) * wx * wx)
        eyy = (lam2 / lam1) * (Dy @ v + (lam2 / (2.0 * lam1)) * wy * wy)
        gxy = (1.0 / lam1) * (lam2 * (Dy @ u) + (Dx @ v) + (lam2 / lam1) * wx * wy)
        kxx = (1.0 / lam1) * (Dx @ px)
        kyy = (lam2 / lam1) * (Dy @ py)
        kxy = (1.0 / lam1) * (lam2 * (Dy @ px) + (Dx @ py))
        gxz = (1.0 / lam1) * wx + px
        gyz = (lam2 / lam1) * wy + py
        res6 = torch.stack([exx, eyy, gxy, kxx, kyy, kxy], dim=1) @ C6.T - thermal6
        Nx = res6[:, 0]; Ny = res6[:, 1]; Nxy = res6[:, 2]
        Mx = res6[:, 3]; My = res6[:, 4]; Mxy = res6[:, 5]
        resS = torch.stack([gxz, gyz], dim=1) @ As2.T
        Qx = resS[:, 0]; Qy = resS[:, 1]
        Fint_U = Dx @ Nx + lam2 * (Dy @ Nxy)
        Fint_V = Dx @ Nxy + lam2 * (Dy @ Ny)
        Fint_W = (lam1 * (Dx @ Qx) + lam1 * lam2 * (Dy @ Qy)
                  + Dx @ (Nx * wx) + lam2 * (Dx @ (Nxy * wy))
                  + lam2 * (Dy @ (Nxy * wx)) + lam2 ** 2 * (Dy @ (Ny * wy)) + FoundW @ w)
        Fint_Fx = Dx @ Mx + lam2 * (Dy @ Mxy) - lam1 * Qx
        Fint_Fy = Dx @ Mxy + lam2 * (Dy @ My) - lam1 * Qy
        Fint = torch.cat([Fint_U, Fint_V, Fint_W, Fint_Fx, Fint_Fy])
        R_red = TTt @ (-Fint - load_scale * f_full)
        if has_nat:
            res_stack = torch.stack([Nx, Ny, Nxy, Mx, My, Mxy, Qx, Qy], dim=0)
            nat_vals = res_stack[comp_idx, node_idx]
            R_red = torch.where(free_is_nat, S @ nat_vals, R_red)
        return R_red

    return residual_nd


def build_strong_vk_residual_soft(system, bc_weight=1.0e3, device=None, is_nonlinear=True):
    import torch
    torch.set_default_dtype(torch.float64)
    if device is None:
        try:
            from odl_config import DEFAULT_DEVICE
            device = DEFAULT_DEVICE
        except Exception:
            device = "cpu"
    P = _prep(system)
    Ng = P["Ng"]
    ndof = 5 * Ng
    nondim = bool(P.get("nondim"))
    lam1 = P.get("lam1"); lam2 = P.get("lam2")
    essential, natural = strong_boundary_map(system["config"], system["Nx"], system["Ny"])
    ess = np.array(sorted(set(essential)), dtype=int)
    s_pde = max(float(np.max(np.abs(system["f_full"]))), 1.0)
    d_lin = system["transform"] @ np.linalg.solve(system["KL_red"], system["f_red"])
    d_scale = max(float(np.max(np.abs(d_lin))), 1.0e-30)

    def t(M):
        return torch.as_tensor(np.asarray(M, dtype=float), device=device)

    def tl(a):
        return torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)

    Dx = t(P["Dx"]); Dy = t(P["Dy"])
    C6 = t(P["C6"]); As2 = t(P["As2"]); FoundW = t(P["FoundW"])
    thermal6 = t(P["thermal6"]); f_full = t(P["f_full"]); k_s = float(P["k_s"])
    ess_mask = torch.zeros(ndof, dtype=torch.bool, device=device)
    if ess.size:
        ess_mask[tl(ess)] = True
    bc_coef = bc_weight / d_scale
    has_nat = bool(natural)
    if has_nat:
        nat_dofs = np.array([d for d, _, _ in natural], dtype=int)
        nat_comp = np.array([c for _, c, _ in natural], dtype=int)
        nat_node = np.array([n for _, _, n in natural], dtype=int)
        nat_mask = torch.zeros(ndof, dtype=torch.bool, device=device)
        nat_mask[tl(nat_dofs)] = True
        Smat = np.zeros((ndof, nat_dofs.size))
        Smat[nat_dofs, np.arange(nat_dofs.size)] = 1.0
        S = t(Smat)
        comp_idx = tl(nat_comp - 1); node_idx = tl(nat_node)
        mask7 = t((nat_comp == 7).astype(float)); mask8 = t((nat_comp == 8).astype(float))

    def residual(d_full):
        nl = 1.0 if is_nonlinear else 0.0
        U = d_full[0:Ng]; V = d_full[Ng:2 * Ng]; W = d_full[2 * Ng:3 * Ng]
        Fx = d_full[3 * Ng:4 * Ng]; Fy = d_full[4 * Ng:5 * Ng]
        Wx = Dx @ W; Wy = Dy @ W
        if nondim:
            ex = (1.0 / lam1) * (Dx @ U + nl * (1.0 / (2.0 * lam1)) * Wx * Wx)
            ey = (lam2 / lam1) * (Dy @ V + nl * (lam2 / (2.0 * lam1)) * Wy * Wy)
            gxy = (1.0 / lam1) * (lam2 * (Dy @ U) + (Dx @ V) + nl * (lam2 / lam1) * Wx * Wy)
            kx = (1.0 / lam1) * (Dx @ Fx); ky = (lam2 / lam1) * (Dy @ Fy)
            kxy = (1.0 / lam1) * (lam2 * (Dy @ Fx) + (Dx @ Fy))
            gxz = (1.0 / lam1) * Wx + Fx; gyz = (lam2 / lam1) * Wy + Fy
        else:
            ex = Dx @ U + nl * 0.5 * Wx * Wx
            ey = Dy @ V + nl * 0.5 * Wy * Wy
            gxy = Dy @ U + Dx @ V + nl * Wx * Wy
            kx = Dx @ Fx; ky = Dy @ Fy; kxy = Dy @ Fx + Dx @ Fy
            gxz = Wx + Fx; gyz = Wy + Fy
        res6 = torch.stack([ex, ey, gxy, kx, ky, kxy], dim=1) @ C6.T - thermal6
        Nx = res6[:, 0]; Ny = res6[:, 1]; Nxy = res6[:, 2]
        Mx = res6[:, 3]; My = res6[:, 4]; Mxy = res6[:, 5]
        resS = torch.stack([gxz, gyz], dim=1) @ As2.T
        Qx = resS[:, 0]; Qy = resS[:, 1]
        if nondim:
            Fint_U = Dx @ Nx + lam2 * (Dy @ Nxy)
            Fint_V = Dx @ Nxy + lam2 * (Dy @ Ny)
            Fint_W = (lam1 * (Dx @ Qx) + lam1 * lam2 * (Dy @ Qy)
                      + nl * (Dx @ (Nx * Wx) + lam2 * (Dx @ (Nxy * Wy))
                              + lam2 * (Dy @ (Nxy * Wx)) + lam2 ** 2 * (Dy @ (Ny * Wy))) + FoundW @ W)
            Fint_Fx = Dx @ Mx + lam2 * (Dy @ Mxy) - lam1 * Qx
            Fint_Fy = Dx @ Mxy + lam2 * (Dy @ My) - lam1 * Qy
        else:
            Fint_U = Dx @ Nx + Dy @ Nxy
            Fint_V = Dx @ Nxy + Dy @ Ny
            Fint_W = (Dx @ Qx + Dy @ Qy
                      + nl * (Dx @ (Nx * Wx + Nxy * Wy) + Dy @ (Nxy * Wx + Ny * Wy)) + FoundW @ W)
            Fint_Fx = Dx @ Mx + Dy @ Mxy - Qx
            Fint_Fy = Dx @ Mxy + Dy @ My - Qy
        Fint = torch.cat([Fint_U, Fint_V, Fint_W, Fint_Fx, Fint_Fy])
        R = (-Fint - f_full) / s_pde
        if has_nat:
            TwX = Qx + nl * (Nx * Wx + Nxy * Wy)
            TwY = Qy + nl * (Nxy * Wx + Ny * Wy)
            res_stack = torch.stack([Nx, Ny, Nxy, Mx, My, Mxy, TwX, TwY], dim=0)
            nat_vals = (res_stack[comp_idx, node_idx]
                        + k_s * (mask7 * Wx[node_idx] + mask8 * Wy[node_idx])) / s_pde
            R = torch.where(nat_mask, S @ nat_vals, R)
        R = torch.where(ess_mask, bc_coef * d_full, R)
        return R

    return residual, d_lin
