


a = 1.0
b = 1.0
h = 0.05
num_layers = 10

MATERIAL = {
    "kind": "goeam",
    "W_Gr": 0.025,
    "H_Gr": 0.8,
    "T": 300.0,
    "distribution_type": "X",
}
shear_corr = 5.0 / 6.0

BOUNDARY_CONDITIONS = [
    {"name": "CCCC", "x_bc_type": "CC", "y_bc_type": "CC",
     "x_bc_convention": "starter", "y_bc_convention": "starter"},
    {"name": "SSSS", "x_bc_type": "SS", "y_bc_type": "SS",
     "x_bc_convention": "ss1", "y_bc_convention": "ss1"},
]

q = -2.5e7

NONDIM = True

N = 21

load_steps = 1
newton_tol = 1.0e-9
newton_max_iter = 50

FEM_MESHES = [16, 32, 64, 96]
FEM_MESH = 96

bc_weight_soft = 1.0e3

DEVICE_ODL = "cuda"
DEVICE_DQ  = "cpu"
DEVICE_FEM = "cpu"

for _dname, _dval in (("DEVICE_DQ", DEVICE_DQ), ("DEVICE_FEM", DEVICE_FEM)):
    if str(_dval).strip().lower() != "cpu":
        raise NotImplementedError(
            "%s=%r: DQ (numpy) / FEM (scipy) are CPU-only here -- no GPU port exists, and at "
            "N=21 on a consumer GPU it measures slower than CPU. Keep it 'cpu'." % (_dname, _dval))

use_cuda = (str(DEVICE_ODL).strip().lower() == "cuda")


def resolve_device():
    try:
        import torch
        if use_cuda and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

SCENARIOS = [
    {"name": "with_foundation", "Kw": 100.0, "Ks": 10.0},
    {"name": "no_foundation",   "Kw": 0.0,   "Ks": 0.0},
]

INIT_GUESS = "linear"
INIT_SWEEP = ["linear", "zero", "random", "gauss"]
INIT_SEED = 0
INIT_SCALE = 0.5
INIT_STRATEGIES = ["linear", "random"]

GN_VARIANT = "original"

LBFGS_PRECONDITION = True

LBFGS_MULTIGRID = True
LBFGS_MG_LEVELS = [2, 3, 5, 8, 13]
LBFGS_MG_STAGE_FRACTIONS = (0.15, 0.25, 0.60)
LBFGS_MG_TOL = 1.0e-5

LBFGS_MG_RESTART = True
LBFGS_MG_RESTART_SCALE = 1.0e-5
LBFGS_MG_RESTART_GATE = 1.0e-2
LBFGS_MG_RESTART_PATIENCE = 600
LBFGS_MG_RESTART_THRESHOLD = 5.0e-4
LBFGS_MG_RESTART_SEEDS = {"CCCC": 46, "SSSS": 43}


def foundation_dimensional(Kw, Ks, D0, a_len):
    k_w = Kw * D0 / a_len ** 4
    k_s = Ks * D0 / a_len ** 2
    return k_w, k_s


def make_config(scenario, N, mode, D0, bc=None):
    if bc is None:
        bc = BOUNDARY_CONDITIONS[0]
    k_w, k_s = foundation_dimensional(scenario["Kw"], scenario["Ks"], D0, a)
    return {
        "a": a, "b": b, "h": h, "num_layers": num_layers,
        "Nx": N, "Ny": N,
        "x_bc_type": bc["x_bc_type"], "y_bc_type": bc["y_bc_type"],
        "x_bc_convention": bc["x_bc_convention"], "y_bc_convention": bc["y_bc_convention"],
        "shear_corr": shear_corr,
        "problem_type": "linear_static" if mode == "linear" else "nonlinear_static",
        "transverse_load": q,
        "k_w": k_w, "k_s": k_s,
        "NONDIM": NONDIM,
    }
