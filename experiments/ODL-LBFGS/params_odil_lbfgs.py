"""
ODIL-LBFGS parameter configuration (self-contained inside experiments/ODL-LBFGS).

Configures the sweep so as to isolate the effect of the adaptive perturbation
restart on the LBFGS nonlinear solve. To that end:

    - the in-loop adaptive perturbation restart is DISABLED
      (perturbation_enabled = False),
    - only the LBFGS optimizer is swept.

The nonlinear solve still runs num_solution_attempts = 3 attempts per
scenario, over the same two foundation scenarios (with / without elastic
foundation), the same DQ discretization (N=13, negative_sum) and identical
material/load parameters as the benchmark.

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
# ------------------------------------------------------------
# DISABLED for this experiment: no adaptive perturbation restart.
# The remaining perturbation_* values are unused while disabled.
# ============================================================
perturbation_enabled = False
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

# Only the LBFGS optimizer is swept here, using the same iteration budget and
# learning rate as the benchmark's lbfgs entry. The adaptive perturbation
# restart is disabled above (perturbation_enabled = False).
optimizers = [
    {
        'name': 'lbfgs',
        'max_iter_linear': 30000,
        'max_iter_nonlinear': 50000,
        'lr': 0.8,
    },
]
