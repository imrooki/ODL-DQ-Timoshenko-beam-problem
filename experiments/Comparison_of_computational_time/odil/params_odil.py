"""
ODIL parameter configuration (self-contained inside Comparison_of_computational_time/odil).

Mirrors the LBFGS/LM branches of the project root params.py but exposes
two foundation scenarios that are swept by `run_odil.py`. Two optimizer
variants are run for each scenario:

    - 'levenberg-marquardt'  : the paper's default (fast, ~30 s)
    - 'lbfgs'                : a gradient-based reference comparable to PINN

Both ODIL runs use the same DQ discretization (N=13, negative_sum) and
identical material/load parameters as the upstream PINN benchmark.

This module does NOT import the project root params.py.
"""

# ============================================================
# Geometry  (mirrors main params.py)
# ============================================================
h = 0.1
L = 20 * h
num_layers = 10

# ============================================================
# Material
# ============================================================
W_Gr = 0.025
H_Gr = 0.8
T = 300
distr_type = 'X'

# ============================================================
# Loading and boundary condition
# ============================================================
q = -0.08
bc_type = 'C-C'

# ============================================================
# Default foundation (k1, k2 are overridden per scenario at runtime)
# ============================================================
k1 = 0.01
k2 = 0.001
foundation_params = {'k1': k1, 'k2': k2}

# ============================================================
# Discretization (DQ, paper default)
# ============================================================
method = 'dq'
N = 13
dq_type = 'negative_sum'
dq_method = dq_type

# ============================================================
# Solver mode
# ============================================================
mode = 'both'  # we need both linear and nonlinear solutions for the comparison

# ============================================================
# Loss weights  (mirrors main params.py)
# ============================================================
pde_weights = (1.0, 1.0, 1.0)
reg_weight = 1e-8
bc_weight = 1000.0

# ============================================================
# Initial-value strategy (used in mode='both')
# ============================================================
use_linear_as_initial = True
initial_value_scale = 0.6
initial_value_mix_ratio = 0.7
num_solution_attempts = 3
validate_physical_solution = False  # keep quiet for sweep

# ============================================================
# Perturbation restart (LBFGS branch)
# ============================================================
perturbation_enabled = True
perturbation_threshold = 0.0005
perturbation_patience = 600
perturbation_scale = 0.005
perturbation_scale_increment = 0.25

# ============================================================
# Optimizer-specific parameters
# ============================================================
gauss_newton_tol = 1e-10
gauss_newton_damping = 0.1
lm_tol = 1e-8
lm_damping_init = 1e-3
lm_damping_factor = 10.0

# ============================================================
# Output / logging
# ============================================================
verbose = False         # silence the sweep; logs still go to per-run files
print_every = 300
save_every = 5000
print_every_epoch = True
log_mode = 'simple'
log_level = 'WARNING'
archived_logs_dir = 'archived_logs'
enable_gpu_monitor = False
gpu_monitor_interval = 99999

save_results = True
results_dir = 'results'
plot_dpi = 200
plot_format = 'png'

# ============================================================
# Compute environment
# ============================================================
use_cuda = True
dtype_str = 'float64'
seed = 42

# ============================================================
# Sweep dimensions (consumed by run_odil.py)
# ============================================================
scenarios = [
    {'name': 'with_foundation', 'k1': 0.01, 'k2': 0.001,
     'description': 'Winkler-Pasternak foundation (paper default)'},
    {'name': 'no_foundation',   'k1': 0.0,  'k2': 0.0,
     'description': 'No elastic foundation (isolated benchmark)'},
]

# Two optimizer variants per scenario.
# - 'levenberg-marquardt' : paper default, residual-vector LSQ + LM damping
# - 'lbfgs'               : gradient-based; epoch-budget comparable to PINN
optimizers = [
    {
        'name': 'levenberg-marquardt',
        'max_iter_linear': 150,
        'max_iter_nonlinear': 300,
        'lr': 1.0,           # placeholder, LM ignores
    },
    {
        'name': 'lbfgs',
        # LBFGS iteration budget: 30k linear warmup + 50k nonlinear.
        'max_iter_linear': 30000,
        'max_iter_nonlinear': 50000,
        'lr': 0.8,                   # main params.py:277
    },
    {
        # Gauss-Newton: pure 2nd-order Newton-step on the LSQ normal equations
        # without LM damping. Iteration budget mirrors main params.py:260-261.
        # tol / damping read from gauss_newton_tol / gauss_newton_damping above.
        'name': 'gauss-newton',
        'max_iter_linear': 100,
        'max_iter_nonlinear': 200,
        'lr': 1.0,                   # placeholder, GN ignores
    },
]
