import numpy as np

from .dq_core import (
    chebyshev_lobatto_nodes, dq_first_derivative_matrix,
    quadrature_weights_for_nodes,
)
from .material_plate import compute_A110_reference, compute_D0_reference

FIELD_NAMES = ["U", "V", "W", "phix", "phiy"]

_MEMBRANE_BENDING = {
    "ex":  [("U", True, False)],
    "ey":  [("V", False, True)],
    "gxy": [("U", False, True), ("V", True, False)],
    "kx":  [("phix", True, False)],
    "ky":  [("phiy", False, True)],
    "kxy": [("phix", False, True), ("phiy", True, False)],
}
_STRAIN_ORDER = ["ex", "ey", "gxy", "kx", "ky", "kxy"]

_SHEAR = {
    "gxz": [("W", True, False), ("phix", False, False)],
    "gyz": [("W", False, True), ("phiy", False, False)],
}
_SHEAR_ORDER = ["gxz", "gyz"]

def _edge_essential_fields(bc_type, convention, edge_normal):
    bc = str(bc_type).strip().upper()
    conv = str(convention).strip().lower()
    if edge_normal == "x":
        tangential_disp, tangential_rot = "V", "phiy"
    else:
        tangential_disp, tangential_rot = "U", "phix"
    if bc in ("FREE", "F-F"):
        return []
    if bc in ("CC", "C-C"):
        return list(FIELD_NAMES)
    if bc in ("SS", "S-S", "HH", "H-H"):
        if conv == "ss1":
            return [tangential_disp, "W", tangential_rot]
        return ["W", tangential_rot]
    raise ValueError(f"Unsupported {edge_normal} boundary condition: {bc_type}")

def _x_block(D1, W, dl, dr):
    if not dl and not dr:
        return W
    if dl and not dr:
        return D1.T @ W
    if not dl and dr:
        return W @ D1
    return D1.T @ W @ D1

def build_weak_form_plate_system(config, material):
    if config.get("NONDIM"):
        return _build_weak_form_nondim(config, material)
    Nx = config["Nx"]
    Ny = config["Ny"]
    Ng = Nx * Ny
    a = config["a"]
    b = config["b"]

    x_nodes = chebyshev_lobatto_nodes(Nx, a)
    y_nodes = chebyshev_lobatto_nodes(Ny, b)
    wx = quadrature_weights_for_nodes(x_nodes)
    wy = quadrature_weights_for_nodes(y_nodes)
    D1x = dq_first_derivative_matrix(x_nodes)
    D1y = dq_first_derivative_matrix(y_nodes)
    Wx = np.diag(wx)
    Wy = np.diag(wy)

    starts = {name: idx * Ng for idx, name in enumerate(FIELD_NAMES)}
    total = 5 * Ng

    A, B, D, As = material["A"], material["B"], material["D"], material["As"]
    C6 = np.block([[A, B], [B, D]])

    KL = np.zeros((total, total))

    def add_block(lhs, rhs, x_block, y_block):
        block = np.kron(x_block, y_block)
        ls, rs = starts[lhs], starts[rhs]
        KL[ls:ls + Ng, rs:rs + Ng] += block

    for i, si in enumerate(_STRAIN_ORDER):
        for j, sj in enumerate(_STRAIN_ORDER):
            coeff = C6[i, j]
            if abs(coeff) < 1.0e-14:
                continue
            for fl, dxl, dyl in _MEMBRANE_BENDING[si]:
                for fr, dxr, dyr in _MEMBRANE_BENDING[sj]:
                    xb = coeff * _x_block(D1x, Wx, dxl, dxr)
                    yb = _x_block(D1y, Wy, dyl, dyr)
                    add_block(fl, fr, xb, yb)

    for i, si in enumerate(_SHEAR_ORDER):
        for j, sj in enumerate(_SHEAR_ORDER):
            coeff = As[i, j]
            if abs(coeff) < 1.0e-14:
                continue
            for fl, dxl, dyl in _SHEAR[si]:
                for fr, dxr, dyr in _SHEAR[sj]:
                    xb = coeff * _x_block(D1x, Wx, dxl, dxr)
                    yb = _x_block(D1y, Wy, dyl, dyr)
                    add_block(fl, fr, xb, yb)

    KL = 0.5 * (KL + KL.T)

    I0, I1, I2 = material["I0"], material["I1"], material["I2"]
    M = np.zeros((total, total))

    def add_mass(lhs, rhs, coeff):
        block = np.kron(coeff * Wx, Wy)
        ls, rs = starts[lhs], starts[rhs]
        M[ls:ls + Ng, rs:rs + Ng] += block

    for field, coeff in (("U", I0), ("V", I0), ("W", I0), ("phix", I2), ("phiy", I2)):
        if abs(coeff) > 1.0e-14:
            add_mass(field, field, coeff)
    if abs(I1) > 1.0e-14:
        for lhs, rhs in (("U", "phix"), ("V", "phiy")):
            add_mass(lhs, rhs, I1)
            add_mass(rhs, lhs, I1)
    M = 0.5 * (M + M.T)

    nx0 = config.get("pre_stress_x", 0.0)
    ny0 = config.get("pre_stress_y", 0.0)
    nxy0 = config.get("pre_stress_xy", 0.0)
    if config.get("use_thermal_prestress", False):
        NT = material.get("NT", np.zeros(3))
        nx0 += NT[0]; ny0 += NT[1]; nxy0 += NT[2]
    KG = np.zeros((total, total))
    wW = starts["W"]
    if abs(nx0) > 1.0e-14:
        KG[wW:wW + Ng, wW:wW + Ng] += np.kron(nx0 * (D1x.T @ Wx @ D1x), Wy)
    if abs(ny0) > 1.0e-14:
        KG[wW:wW + Ng, wW:wW + Ng] += np.kron(ny0 * Wx, D1y.T @ Wy @ D1y)
    if abs(nxy0) > 1.0e-14:
        KG[wW:wW + Ng, wW:wW + Ng] += np.kron(nxy0 * (D1x.T @ Wx), Wy @ D1y)
        KG[wW:wW + Ng, wW:wW + Ng] += np.kron(nxy0 * (Wx @ D1x), D1y.T @ Wy)
    KG = 0.5 * (KG + KG.T)

    k_w = config.get("k_w", 0.0)
    k_s = config.get("k_s", 0.0)
    KF = np.zeros((total, total))
    if abs(k_w) > 1.0e-14:
        KF[wW:wW + Ng, wW:wW + Ng] += np.kron(k_w * Wx, Wy)
    if abs(k_s) > 1.0e-14:
        KF[wW:wW + Ng, wW:wW + Ng] += np.kron(k_s * (D1x.T @ Wx @ D1x), Wy)
        KF[wW:wW + Ng, wW:wW + Ng] += np.kron(k_s * Wx, D1y.T @ Wy @ D1y)
    KF = 0.5 * (KF + KF.T)
    KL = KL + KF

    f = np.zeros(total)
    q = config.get("transverse_load", 0.0)
    if abs(q) > 1.0e-14:
        f[wW:wW + Ng] = q * np.kron(wx, wy)

    x_ess = _edge_essential_fields(config["x_bc_type"], config["x_bc_convention"], "x")
    y_ess = _edge_essential_fields(config["y_bc_type"], config["y_bc_convention"], "y")
    boundary = set()
    for field in x_ess:
        for i in (0, Nx - 1):
            for j in range(Ny):
                boundary.add(starts[field] + i * Ny + j)
    for field in y_ess:
        for j in (0, Ny - 1):
            for i in range(Nx):
                boundary.add(starts[field] + i * Ny + j)

    free_dofs = np.array([d for d in range(total) if d not in boundary])
    T = np.zeros((total, free_dofs.size))
    T[free_dofs, np.arange(free_dofs.size)] = 1.0

    KL_red = T.T @ KL @ T
    M_red = T.T @ M @ T
    KG_red = T.T @ KG @ T
    f_red = T.T @ f

    return {
        "config": config,
        "material": material,
        "Nx": Nx, "Ny": Ny, "Ng": Ng, "a": a, "b": b,
        "x_nodes": x_nodes, "y_nodes": y_nodes,
        "KL_full": KL, "M_full": M, "KG_full": KG, "Kf_full": KF, "f_full": f,
        "KL_red": KL_red, "M_red": M_red, "KG_red": KG_red, "f_red": f_red,
        "transform": T,
        "free_dofs": free_dofs,
        "boundary_dofs": np.array(sorted(boundary)),
    }


def _build_weak_form_nondim(config, material):

    Nx = config["Nx"]; Ny = config["Ny"]; Ng = Nx * Ny
    a = config["a"]; b = config["b"]; h = config["h"]
    lam1 = a / h; lam2 = a / b
    A110 = compute_A110_reference(h); D0 = compute_D0_reference(h)

    x_nodes = chebyshev_lobatto_nodes(Nx, 1.0)
    y_nodes = chebyshev_lobatto_nodes(Ny, 1.0)
    wx = quadrature_weights_for_nodes(x_nodes)
    wy = quadrature_weights_for_nodes(y_nodes)
    D1x = dq_first_derivative_matrix(x_nodes)
    D1y = dq_first_derivative_matrix(y_nodes)
    Dx = np.kron(D1x, np.eye(Ny))
    Dy = np.kron(np.eye(Nx), D1y)
    w = np.kron(wx, wy); Wd = np.diag(w)
    I = np.eye(Ng)
    starts = {name: idx * Ng for idx, name in enumerate(FIELD_NAMES)}
    total = 5 * Ng

    A, B, D, As = material["A"], material["B"], material["D"], material["As"]
    C6 = np.block([[A / A110, B / (h * A110)], [B / (h * A110), D / (h ** 2 * A110)]])
    As2 = As / A110

    def Bop(blocks):
        Bm = np.zeros((Ng, total))
        for fi, op in blocks:
            Bm[:, fi * Ng:(fi + 1) * Ng] = op
        return Bm

    iU, iV, iW, iFx, iFy = 0, 1, 2, 3, 4
    B_mb = [
        Bop([(iU, (1.0 / lam1) * Dx)]),
        Bop([(iV, (lam2 / lam1) * Dy)]),
        Bop([(iU, (lam2 / lam1) * Dy), (iV, (1.0 / lam1) * Dx)]),
        Bop([(iFx, (1.0 / lam1) * Dx)]),
        Bop([(iFy, (lam2 / lam1) * Dy)]),
        Bop([(iFx, (lam2 / lam1) * Dy), (iFy, (1.0 / lam1) * Dx)]),
    ]
    B_s = [
        Bop([(iW, (1.0 / lam1) * Dx), (iFx, I)]),
        Bop([(iW, (lam2 / lam1) * Dy), (iFy, I)]),
    ]

    KL = np.zeros((total, total))
    for i in range(6):
        for j in range(6):
            c = C6[i, j]
            if abs(c) < 1.0e-14:
                continue
            KL += c * (B_mb[i].T @ Wd @ B_mb[j])
    for i in range(2):
        for j in range(2):
            c = As2[i, j]
            if abs(c) < 1.0e-14:
                continue
            KL += c * (B_s[i].T @ Wd @ B_s[j])
    KL = 0.5 * (KL + KL.T)

    k_w = config.get("k_w", 0.0); k_s = config.get("k_s", 0.0)
    cw = k_w * h ** 2 / A110; cs = k_s * h ** 2 / (A110 * a ** 2)
    KF = np.zeros((total, total)); iWs = starts["W"]
    if abs(k_w) > 1.0e-14:
        KF[iWs:iWs + Ng, iWs:iWs + Ng] += cw * Wd
    if abs(k_s) > 1.0e-14:
        KF[iWs:iWs + Ng, iWs:iWs + Ng] += cs * (Dx.T @ Wd @ Dx + lam2 ** 2 * (Dy.T @ Wd @ Dy))
    KF = 0.5 * (KF + KF.T)
    KL = KL + KF

    f = np.zeros(total)
    q = config.get("transverse_load", 0.0)
    if abs(q) > 1.0e-14:
        f[iWs:iWs + Ng] = (q * h / A110) * w

    x_ess = _edge_essential_fields(config["x_bc_type"], config["x_bc_convention"], "x")
    y_ess = _edge_essential_fields(config["y_bc_type"], config["y_bc_convention"], "y")
    boundary = set()
    for field in x_ess:
        for i in (0, Nx - 1):
            for j in range(Ny):
                boundary.add(starts[field] + i * Ny + j)
    for field in y_ess:
        for j in (0, Ny - 1):
            for i in range(Nx):
                boundary.add(starts[field] + i * Ny + j)
    free_dofs = np.array([d for d in range(total) if d not in boundary])
    T = np.zeros((total, free_dofs.size))
    T[free_dofs, np.arange(free_dofs.size)] = 1.0
    M = np.zeros((total, total)); KG = np.zeros((total, total))
    KL_red = T.T @ KL @ T; f_red = T.T @ f

    return {
        "config": config, "material": material,
        "Nx": Nx, "Ny": Ny, "Ng": Ng, "a": a, "b": b,
        "x_nodes": x_nodes, "y_nodes": y_nodes,
        "KL_full": KL, "M_full": M, "KG_full": KG, "Kf_full": KF, "f_full": f,
        "KL_red": KL_red, "M_red": T.T @ M @ T, "KG_red": T.T @ KG @ T, "f_red": f_red,
        "transform": T, "free_dofs": free_dofs,
        "boundary_dofs": np.array(sorted(boundary)),
        "nondim": True, "lam1": lam1, "lam2": lam2, "A110": A110, "D0": D0,
        "C6_nd": C6, "As2_nd": As2,
    }
