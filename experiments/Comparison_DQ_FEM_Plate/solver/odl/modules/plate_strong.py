import numpy as np

from .dq_core import chebyshev_lobatto_nodes, build_2d_operators

FIELD_NAMES = ["U", "V", "W", "phix", "phiy"]

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

def _conjugate_component(field, edge):
    if edge == "x":
        return {"U": 1, "V": 3, "W": 7, "phix": 4, "phiy": 6}[field]
    return {"U": 3, "V": 2, "W": 8, "phix": 6, "phiy": 5}[field]

def strong_boundary_map(config, Nx, Ny):

    Ng = Nx * Ny
    starts = {name: idx * Ng for idx, name in enumerate(FIELD_NAMES)}
    x_ess = _edge_essential_fields(config["x_bc_type"], config["x_bc_convention"], "x")
    y_ess = _edge_essential_fields(config["y_bc_type"], config["y_bc_convention"], "y")

    essential = set()
    natural = []
    for i in range(Nx):
        for j in range(Ny):
            on_x = (i == 0) or (i == Nx - 1)
            on_y = (j == 0) or (j == Ny - 1)
            if not on_x and not on_y:
                continue
            ess = set()
            if on_x:
                ess |= set(x_ess)
            if on_y:
                ess |= set(y_ess)
            p = i * Ny + j
            for field in FIELD_NAMES:
                dof = starts[field] + p
                if field in ess:
                    essential.add(dof)
                else:
                    edge = "x" if on_x else "y"
                    natural.append((dof, _conjugate_component(field, edge), p))
    return sorted(essential), natural

def build_strong_form_plate_system(config, material):
    Nx = config["Nx"]
    Ny = config["Ny"]
    Ng = Nx * Ny
    a = config["a"]
    b = config["b"]

    x_nodes = chebyshev_lobatto_nodes(Nx, a)
    y_nodes = chebyshev_lobatto_nodes(Ny, b)
    ops = build_2d_operators(x_nodes, y_nodes)
    Dx, Dy = ops["Dx"], ops["Dy"]
    Dxx, Dyy, Dxy = ops["Dxx"], ops["Dyy"], ops["Dxy"]

    Z = np.zeros((Ng, Ng))
    I = np.eye(Ng)

    Bmb = np.block([
        [Dx, Z,  Z,  Z,  Z],
        [Z,  Dy, Z,  Z,  Z],
        [Dy, Dx, Z,  Z,  Z],
        [Z,  Z,  Z,  Dx, Z],
        [Z,  Z,  Z,  Z,  Dy],
        [Z,  Z,  Z,  Dy, Dx],
    ])
    Bs = np.block([
        [Z, Z, Dx, I, Z],
        [Z, Z, Dy, Z, I],
    ])

    A, B, D, As = material["A"], material["B"], material["D"], material["As"]
    C6 = np.block([[A, B], [B, D]])

    R = np.kron(C6, I) @ Bmb
    Qd = np.kron(As, I) @ Bs

    Nx_op = R[0 * Ng:1 * Ng]
    Ny_op = R[1 * Ng:2 * Ng]
    Nxy_op = R[2 * Ng:3 * Ng]
    Mx_op = R[3 * Ng:4 * Ng]
    My_op = R[4 * Ng:5 * Ng]
    Mxy_op = R[5 * Ng:6 * Ng]
    Qx_op = Qd[0 * Ng:1 * Ng]
    Qy_op = Qd[1 * Ng:2 * Ng]

    EqU = Dx @ Nx_op + Dy @ Nxy_op
    EqV = Dx @ Nxy_op + Dy @ Ny_op
    EqW = Dx @ Qx_op + Dy @ Qy_op
    k_w = config.get("k_w", 0.0)
    k_s = config.get("k_s", 0.0)
    if abs(k_w) > 1.0e-14 or abs(k_s) > 1.0e-14:
        EqW[:, 2 * Ng:3 * Ng] = EqW[:, 2 * Ng:3 * Ng] - k_w * I + k_s * (Dxx + Dyy)
    EqFx = Dx @ Mx_op + Dy @ Mxy_op - Qx_op
    EqFy = Dx @ Mxy_op + Dy @ My_op - Qy_op
    KL_full = -np.vstack([EqU, EqV, EqW, EqFx, EqFy])

    iU = slice(0 * Ng, 1 * Ng)
    iV = slice(1 * Ng, 2 * Ng)
    iW = slice(2 * Ng, 3 * Ng)
    iFx = slice(3 * Ng, 4 * Ng)
    iFy = slice(4 * Ng, 5 * Ng)

    f_full = np.zeros(5 * Ng)
    f_full[iW] = config.get("transverse_load", 0.0)

    nx0 = config.get("pre_stress_x", 0.0)
    ny0 = config.get("pre_stress_y", 0.0)
    nxy0 = config.get("pre_stress_xy", 0.0)
    if config.get("use_thermal_prestress", False):
        NT = material.get("NT", np.zeros(3))
        nx0 += NT[0]; ny0 += NT[1]; nxy0 += NT[2]
    KG_full = np.zeros((5 * Ng, 5 * Ng))
    KG_full[iW, iW] = -(nx0 * Dxx + ny0 * Dyy + 2.0 * nxy0 * Dxy)

    I0, I1, I2 = material["I0"], material["I1"], material["I2"]
    M_full = np.zeros((5 * Ng, 5 * Ng))
    M_full[iU, iU] = I0 * I
    M_full[iV, iV] = I0 * I
    M_full[iW, iW] = I0 * I
    M_full[iFx, iFx] = I2 * I
    M_full[iFy, iFy] = I2 * I
    M_full[iU, iFx] = I1 * I
    M_full[iFx, iU] = I1 * I
    M_full[iV, iFy] = I1 * I
    M_full[iFy, iV] = I1 * I

    res_ops = [Nx_op, Ny_op, Nxy_op, Mx_op, My_op, Mxy_op, Qx_op, Qy_op]
    essential, natural = strong_boundary_map(config, Nx, Ny)
    for dof, comp, node in natural:
        KL_full[dof, :] = res_ops[comp - 1][node, :]
        if abs(k_s) > 1.0e-14:
            if comp == 7:
                KL_full[dof, 2 * Ng:3 * Ng] += k_s * Dx[node, :]
            elif comp == 8:
                KL_full[dof, 2 * Ng:3 * Ng] += k_s * Dy[node, :]
        M_full[dof, :] = 0.0
        M_full[:, dof] = 0.0
        KG_full[dof, :] = 0.0
        f_full[dof] = 0.0

    total = 5 * Ng
    ess_set = set(essential)
    free_dofs = np.array([d for d in range(total) if d not in ess_set])
    T = np.zeros((total, free_dofs.size))
    T[free_dofs, np.arange(free_dofs.size)] = 1.0

    KL_red = T.T @ KL_full @ T
    M_red = T.T @ M_full @ T
    KG_red = T.T @ KG_full @ T
    f_red = T.T @ f_full

    return {
        "config": config,
        "material": material,
        "Nx": Nx, "Ny": Ny, "Ng": Ng, "a": a, "b": b,
        "x_nodes": x_nodes, "y_nodes": y_nodes,
        "KL_full": KL_full, "M_full": M_full, "KG_full": KG_full, "f_full": f_full,
        "KL_red": KL_red, "M_red": M_red, "KG_red": KG_red, "f_red": f_red,
        "transform": T,
        "free_dofs": free_dofs,
        "essential_dofs": np.array(sorted(ess_set)),
        "ops": ops,
    }
