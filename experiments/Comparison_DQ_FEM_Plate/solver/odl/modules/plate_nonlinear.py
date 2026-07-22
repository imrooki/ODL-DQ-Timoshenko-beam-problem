import numpy as np

from .odl_config import DEFAULT_DEVICE
from .dq_core import (
    dq_first_derivative_matrix, quadrature_weights_for_nodes,
)


def _imperfection_slopes(system, Dx, Dy):
    cfg = system["config"]
    mu = cfg.get("imperfection_amplitude", 0.0)
    Ng = system["Ng"]
    if abs(mu) < 1.0e-300:
        return np.zeros(Ng), np.zeros(Ng)
    a, b, h = system["a"], system["b"], cfg["h"]
    xg = np.asarray(system["x_nodes"], dtype=float)
    yg = np.asarray(system["y_nodes"], dtype=float)
    wbar = mu * h * np.kron(np.sin(np.pi * xg / a), np.sin(np.pi * yg / b))
    return Dx @ wbar, Dy @ wbar


def build_weak_vk_residual(system, device=None):

    import torch
    torch.set_default_dtype(torch.float64)
    if device is None:
        device = DEFAULT_DEVICE

    Nx, Ny, Ng = system["Nx"], system["Ny"], system["Ng"]
    x_nodes, y_nodes = system["x_nodes"], system["y_nodes"]
    D1x = dq_first_derivative_matrix(x_nodes)
    D1y = dq_first_derivative_matrix(y_nodes)
    wx = quadrature_weights_for_nodes(x_nodes)
    wy = quadrature_weights_for_nodes(y_nodes)
    Dx = np.kron(D1x, np.eye(Ny))
    Dy = np.kron(np.eye(Nx), D1y)
    w = np.kron(wx, wy)

    mat = system["material"]
    C6 = np.block([[mat["A"], mat["B"]], [mat["B"], mat["D"]]])
    As2 = mat["As"]
    nt6 = np.concatenate([mat.get("NT", np.zeros(3)), mat.get("MT", np.zeros(3))])
    if system["config"].get("NONDIM"):
        lam1 = system["lam1"]; lam2 = system["lam2"]; A110 = system["A110"]; hh = system["config"]["h"]
        Dx = (1.0 / lam1) * Dx; Dy = (lam2 / lam1) * Dy
        C6 = system["C6_nd"]; As2 = system["As2_nd"]
        nt6 = np.concatenate([mat.get("NT", np.zeros(3)) / A110, mat.get("MT", np.zeros(3)) / (hh * A110)])

    def t(M):
        return torch.as_tensor(np.asarray(M, dtype=float), device=device)

    Dx_t, Dy_t = t(Dx), t(Dy)
    DxT, DyT = t(Dx.T), t(Dy.T)
    w_t = t(w)
    C6_t, As2_t = t(C6), t(As2)
    T_t, TT_t = t(system["transform"]), t(system["transform"].T)
    f_t = t(system["f_full"])
    Wbarx, Wbary = _imperfection_slopes(system, Dx, Dy)
    Wbarx_t, Wbary_t = t(Wbarx), t(Wbary)
    Kf_t = t(system.get("Kf_full", np.zeros((5 * Ng, 5 * Ng))))
    thermal6_t = t(nt6)

    def residual(d_red, load_scale=1.0):
        d = T_t @ d_red
        U = d[0:Ng]; V = d[Ng:2 * Ng]; W = d[2 * Ng:3 * Ng]
        Fx = d[3 * Ng:4 * Ng]; Fy = d[4 * Ng:5 * Ng]

        Wx = Dx_t @ W
        Wy = Dy_t @ W
        Wxt = Wx + Wbarx_t
        Wyt = Wy + Wbary_t
        ex = Dx_t @ U + Wbarx_t * Wx + 0.5 * Wx * Wx
        ey = Dy_t @ V + Wbary_t * Wy + 0.5 * Wy * Wy
        gxy = Dy_t @ U + Dx_t @ V + Wbarx_t * Wy + Wbary_t * Wx + Wx * Wy
        kx = Dx_t @ Fx
        ky = Dy_t @ Fy
        kxy = Dy_t @ Fx + Dx_t @ Fy
        gxz = Wx + Fx
        gyz = Wy + Fy

        res6 = torch.stack([ex, ey, gxy, kx, ky, kxy], dim=1) @ C6_t.T - thermal6_t
        Nx_, Ny_, Nxy = res6[:, 0], res6[:, 1], res6[:, 2]
        Mx, My, Mxy = res6[:, 3], res6[:, 4], res6[:, 5]
        resS = torch.stack([gxz, gyz], dim=1) @ As2_t.T
        Qx, Qy = resS[:, 0], resS[:, 1]

        RU = DxT @ (w_t * Nx_) + DyT @ (w_t * Nxy)
        RV = DyT @ (w_t * Ny_) + DxT @ (w_t * Nxy)
        RW = (DxT @ (w_t * Nx_ * Wxt + w_t * Nxy * Wyt + w_t * Qx)
              + DyT @ (w_t * Ny_ * Wyt + w_t * Nxy * Wxt + w_t * Qy))
        RFx = DxT @ (w_t * Mx) + DyT @ (w_t * Mxy) + w_t * Qx
        RFy = DyT @ (w_t * My) + DxT @ (w_t * Mxy) + w_t * Qy

        R_full = torch.cat([RU, RV, RW, RFx, RFy]) + Kf_t @ d - load_scale * f_t
        return TT_t @ R_full

    return residual

def build_weak_vk_energy(system, device=None):
    import torch
    torch.set_default_dtype(torch.float64)
    if device is None:
        device = DEFAULT_DEVICE

    Nx, Ny, Ng = system["Nx"], system["Ny"], system["Ng"]
    x_nodes, y_nodes = system["x_nodes"], system["y_nodes"]
    D1x = dq_first_derivative_matrix(x_nodes)
    D1y = dq_first_derivative_matrix(y_nodes)
    wx = quadrature_weights_for_nodes(x_nodes)
    wy = quadrature_weights_for_nodes(y_nodes)
    Dx = np.kron(D1x, np.eye(Ny))
    Dy = np.kron(np.eye(Nx), D1y)
    w = np.kron(wx, wy)

    mat = system["material"]
    C6 = np.block([[mat["A"], mat["B"]], [mat["B"], mat["D"]]])
    As2 = mat["As"]
    nt6 = np.concatenate([mat.get("NT", np.zeros(3)), mat.get("MT", np.zeros(3))])
    if system["config"].get("NONDIM"):
        lam1 = system["lam1"]; lam2 = system["lam2"]; A110 = system["A110"]; hh = system["config"]["h"]
        Dx = (1.0 / lam1) * Dx; Dy = (lam2 / lam1) * Dy
        C6 = system["C6_nd"]; As2 = system["As2_nd"]
        nt6 = np.concatenate([mat.get("NT", np.zeros(3)) / A110, mat.get("MT", np.zeros(3)) / (hh * A110)])

    def t(M):
        return torch.as_tensor(np.asarray(M, dtype=float), device=device)

    Dx_t, Dy_t = t(Dx), t(Dy)
    w_t = t(w)
    C6_t, As2_t = t(C6), t(As2)
    T_t = t(system["transform"])
    f_t = t(system["f_full"])
    Wbarx, Wbary = _imperfection_slopes(system, Dx, Dy)
    Wbarx_t, Wbary_t = t(Wbarx), t(Wbary)
    Kf_t = t(system.get("Kf_full", np.zeros((5 * Ng, 5 * Ng))))
    thermal6_t = t(nt6)

    def energy(d_red, load_scale=1.0):
        d = T_t @ d_red
        U = d[0:Ng]; V = d[Ng:2 * Ng]; W = d[2 * Ng:3 * Ng]
        Fx = d[3 * Ng:4 * Ng]; Fy = d[4 * Ng:5 * Ng]

        Wx = Dx_t @ W
        Wy = Dy_t @ W
        Wxt = Wx + Wbarx_t
        Wyt = Wy + Wbary_t
        ex = Dx_t @ U + Wbarx_t * Wx + 0.5 * Wx * Wx
        ey = Dy_t @ V + Wbary_t * Wy + 0.5 * Wy * Wy
        gxy = Dy_t @ U + Dx_t @ V + Wbarx_t * Wy + Wbary_t * Wx + Wx * Wy
        kx = Dx_t @ Fx
        ky = Dy_t @ Fy
        kxy = Dy_t @ Fx + Dx_t @ Fy
        gxz = Wx + Fx
        gyz = Wy + Fy

        e6 = torch.stack([ex, ey, gxy, kx, ky, kxy], dim=1)
        g2 = torch.stack([gxz, gyz], dim=1)
        u_bend = 0.5 * (w_t * ((e6 @ C6_t) * e6).sum(dim=1)).sum()
        u_shear = 0.5 * (w_t * ((g2 @ As2_t) * g2).sum(dim=1)).sum()
        u_found = 0.5 * torch.dot(d, Kf_t @ d)
        u_therm = -(w_t * (e6 @ thermal6_t)).sum()
        return u_bend + u_shear + u_found + u_therm - load_scale * torch.dot(f_t, d)

    return energy
