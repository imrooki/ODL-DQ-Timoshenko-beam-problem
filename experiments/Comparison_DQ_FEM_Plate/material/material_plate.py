import numpy as np

def isotropic_plane_stress_stiffness(E, nu):
    denom = 1.0 - nu ** 2
    q11 = E / denom
    q12 = nu * E / denom
    q66 = E / (2.0 * (1.0 + nu))
    return np.array([[q11, q12, 0.0],
                     [q12, q11, 0.0],
                     [0.0, 0.0, q66]])

def isotropic_shear_stiffness(E, nu):
    g = E / (2.0 * (1.0 + nu))
    return np.array([[g, 0.0],
                     [0.0, g]])

def homogeneous_laminate(h, num_layers, E, nu, rho, alpha=0.0):

    dz = h / num_layers
    z0 = -0.5 * h
    layers = []
    for k in range(num_layers):
        z_bot = z0 + k * dz
        layers.append({
            "z_bot": z_bot,
            "z_top": z_bot + dz,
            "E": E, "nu": nu, "rho": rho, "alpha": alpha,
        })
    return layers

def compute_laminate_stiffness(layers, shear_corr):
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    As = np.zeros((2, 2))
    I0 = I1 = I2 = 0.0
    for layer in layers:
        z0 = layer["z_bot"]
        z1 = layer["z_top"]
        dz = z1 - z0
        q = isotropic_plane_stress_stiffness(layer["E"], layer["nu"])
        qs = isotropic_shear_stiffness(layer["E"], layer["nu"])
        A += q * dz
        B += q * (z1 ** 2 - z0 ** 2) / 2.0
        D += q * (z1 ** 3 - z0 ** 3) / 3.0
        As += shear_corr * qs * dz
        I0 += layer["rho"] * dz
        I1 += layer["rho"] * (z1 ** 2 - z0 ** 2) / 2.0
        I2 += layer["rho"] * (z1 ** 3 - z0 ** 3) / 3.0
    return {"A": A, "B": B, "D": D, "As": As, "I0": I0, "I1": I1, "I2": I2}

def compute_thermal_resultants(layers, delta_T):
    # Thermal force/moment resultants N^th, M^th (eq.23) from the layer CTE and
    # the temperature rise delta_T above the stress-free reference. Zero when
    # delta_T = 0, so isothermal cases are unaffected.
    NT = np.zeros(3)
    MT = np.zeros(3)
    for layer in layers:
        q = isotropic_plane_stress_stiffness(layer["E"], layer["nu"])
        eps_t = np.array([layer["alpha"] * delta_T, layer["alpha"] * delta_T, 0.0])
        z0, z1 = layer["z_bot"], layer["z_top"]
        NT += q @ eps_t * (z1 - z0)
        MT += q @ eps_t * (z1 ** 2 - z0 ** 2) / 2.0
    return NT, MT

def build_homogeneous_material(h, num_layers, E, nu, rho, shear_corr, alpha=0.0, delta_T=0.0):

    layers = homogeneous_laminate(h, num_layers, E, nu, rho, alpha)
    mat = compute_laminate_stiffness(layers, shear_corr)
    mat["NT"], mat["MT"] = compute_thermal_resultants(layers, delta_T)
    return mat

GOEAM_CONSTANTS = {
    "T_0": 300.0,
    "E_Cu": 65.79e9, "nu_Cu": 0.387, "rho_Cu": 8.80e3, "alpha_Cu": 16.51e-6,
    "E_Gr": 929.57e9, "nu_Gr": 0.220, "rho_Gr": 1.80e3, "alpha_Gr": -3.98e-6,
    "l_Gr": 83.76, "t_Gr": 3.4,
}

def _clip(value, lo, hi):
    return min(max(value, lo), hi)

def graphene_volume_fraction_from_weight(W_Gr, c=GOEAM_CONSTANTS):

    if W_Gr <= 0.0:
        return 0.0
    return W_Gr / (W_Gr + (c["rho_Gr"] / c["rho_Cu"]) * (1.0 - W_Gr))

def layer_graphene_volume_fraction(layer_index, total_layers, V_base, dist):

    if total_layers == 1:
        return _clip(V_base, 0.0, 1.0)
    k, n = layer_index, total_layers
    kind = str(dist).upper()
    if kind == "X":
        v = 2.0 * V_base * abs(2 * k - n - 1) / n
    elif kind == "O":
        v = 2.0 * V_base * (1.0 - abs(2 * k - n - 1) / n)
    elif kind == "U":
        v = V_base
    else:
        raise ValueError(f"Unsupported distribution_type: {dist}")
    return _clip(v, 0.0, 1.0)

def goeam_correction_factors(V_Gr, T_ratio, H_Gr):

    if V_Gr == 0.0:
        return 1.0, 1.0, 1.0, 1.0
    V = _clip(V_Gr, 0.0, 1.0)
    H = _clip(H_Gr, 0.0, 10.0)
    Tr = _clip(T_ratio, 0.5, 3.0)
    f_E = (1.11 - 1.22 * V - 0.134 * Tr + 0.559 * V * Tr
           - 5.5 * H * V + 38.0 * H * V ** 2 - 20.6 * H ** 2 * V ** 2)
    f_nu = (1.01 - 1.43 * V + 0.165 * Tr - 16.8 * H * V
            - 1.1 * H * V * Tr + 16.0 * H ** 2 * V ** 2)
    f_alpha = 0.794 - 16.8 * V ** 2 - 0.0279 * Tr ** 2 + 0.182 * Tr * (1.0 + V)
    f_rho = 1.01 - 2.01 * V ** 2 - 0.0131 * Tr
    return f_E, f_nu, f_alpha, f_rho

def goeam_layer_properties(layer_index, total_layers, V_base, T_ratio, H_Gr, dist,
                           c=GOEAM_CONSTANTS):

    V = layer_graphene_volume_fraction(layer_index, total_layers, V_base, dist)
    V_cu = 1.0 - V
    f_E, f_nu, f_alpha, f_rho = goeam_correction_factors(V, T_ratio, H_Gr)
    ksai = 2.0 * (c["l_Gr"] / c["t_Gr"])
    ratio = c["E_Gr"] / c["E_Cu"]
    eta = (ratio - 1.0) / (ratio + ksai)
    E = (1.0 + ksai * eta * V) / (1.0 - eta * V) * c["E_Cu"] * f_E
    nu = (c["nu_Gr"] * V + c["nu_Cu"] * V_cu) * f_nu
    alpha = (c["alpha_Gr"] * V + c["alpha_Cu"] * V_cu) * f_alpha
    rho = (c["rho_Gr"] * V + c["rho_Cu"] * V_cu) * f_rho
    return E, nu, alpha, rho, V

def goeam_laminate(h, num_layers, W_Gr, H_Gr, T, dist, c=GOEAM_CONSTANTS):

    T_ratio = T / c["T_0"]
    V_base = graphene_volume_fraction_from_weight(W_Gr, c)
    dz = h / num_layers
    z0 = -0.5 * h
    layers = []
    for k in range(1, num_layers + 1):
        z_bot = z0 + (k - 1) * dz
        E, nu, alpha, rho, _ = goeam_layer_properties(
            k, num_layers, V_base, T_ratio, H_Gr, dist, c)
        layers.append({
            "z_bot": z_bot, "z_top": z_bot + dz,
            "E": E, "nu": nu, "rho": rho, "alpha": alpha,
        })
    return layers

def build_goeam_material(h, num_layers, W_Gr, H_Gr, T, dist, shear_corr):

    layers = goeam_laminate(h, num_layers, W_Gr, H_Gr, T, dist)
    mat = compute_laminate_stiffness(layers, shear_corr)
    mat["NT"], mat["MT"] = compute_thermal_resultants(layers, T - GOEAM_CONSTANTS["T_0"])
    return mat

def compute_A110_reference(h):
    """Pure-copper extensional reference stiffness  A110 = E_Cu*h/(1 - nu_Cu^2).
    The nondimensionalization reference for stiffness/load/displacement (the standard
    FGM-plate A110 convention). Uses the GOEAM copper-matrix constants."""
    E_Cu = GOEAM_CONSTANTS["E_Cu"]
    nu_Cu = GOEAM_CONSTANTS["nu_Cu"]
    return E_Cu * h / (1.0 - nu_Cu ** 2)

def compute_D0_reference(h):
    """Pure-copper bending reference rigidity  D0 = E_Cu*h^3/(12*(1 - nu_Cu^2)).
    The elastic-foundation normalization reference (literature Eq.52:
    Kw = k_w*a^4/D0, Ks = k_s*a^2/D0), pure copper -- NOT the laminate D[0,0].
    Identity used by the nondim foundation coefficients:  D0 == compute_A110_reference(h)*h**2/12."""
    E_Cu = GOEAM_CONSTANTS["E_Cu"]
    nu_Cu = GOEAM_CONSTANTS["nu_Cu"]
    return E_Cu * h ** 3 / (12.0 * (1.0 - nu_Cu ** 2))

def build_material(config, material_spec):

    kind = str(material_spec.get("kind", "homogeneous")).lower()
    mp = material_spec["params"]
    if kind == "homogeneous":
        return build_homogeneous_material(
            config["h"], config["num_layers"], mp["E"], mp["nu"], mp["rho"],
            config["shear_corr"], mp.get("alpha", 0.0), mp.get("delta_T", 0.0))
    if kind in ("goeam", "grcu", "fgm"):
        return build_goeam_material(
            config["h"], config["num_layers"], mp["W_Gr"], mp["H_Gr"], mp["T"],
            mp["distribution_type"], config["shear_corr"])
    raise ValueError(f"Unknown material kind: {kind}")
