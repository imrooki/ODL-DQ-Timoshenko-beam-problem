"""
ODIL-Timoshenko Beam Solver Parameter Configuration File
=============================================

This file centrally manages all parameter configurations for the ODIL-FSDT (First-order Shear Deformation Theory) beam solver.
Based on the ODIL (Optimizing a Discrete Loss) framework, it solves the bending problem of Timoshenko beams by optimizing a discrete loss function.

## Theoretical Background
This project solves the governing equations of Timoshenko beams resting on an elastic foundation:
- Linear problem: neglects geometric nonlinear terms, suitable for small-deformation analysis
- Nonlinear problem: includes von Karman-type geometric nonlinearity, suitable for large-deflection analysis
- Material model: graphene-reinforced functionally graded composite (GPL-FGM), based on Halpin-Tsai theory

## Supported discretization methods (selected via the method parameter)
1. **DQ (Differential Quadrature)**:
   - Principle: uses Chebyshev-Gauss-Lobatto nodes and approximates derivatives via weighting coefficients
   - Advantages: spectral accuracy, high accuracy with few nodes
   - Disadvantages: condition number deteriorates at large N, sensitive to boundary conditions

2. **Taylor/Fornberg**:
   - Principle: local finite difference based on Taylor series expansion, computing weights with the Fornberg algorithm
   - Advantages: good stability, local support, easy to parallelize
   - Disadvantages: requires more nodes to reach high accuracy

3. **Spline (spline collocation method)**:
   - Principle: uses piecewise polynomial interpolation, guaranteeing the specified continuity
   - Advantages: smooth derivatives, flexible boundary handling, rigorous mathematical foundation
   - Disadvantages: may require high-order splines for large-deflection problems

## Supported optimizers (selected via the optim_name parameter)
1. **L-BFGS**: quasi-Newton method, uses limited-memory approximation of the Hessian, suitable for smooth optimization problems
2. **Adam**: first-order adaptive moment estimation, insensitive to learning rate, suitable for non-convex problems
3. **Gauss-Newton**: second-order method, exploits Jacobian information, quadratic convergence rate
4. **Levenberg-Marquardt**: Gauss-Newton with adaptive damping, improves robustness

## Parameter selection recommendations
- **High-accuracy needs**: DQ method + L-BFGS optimizer (N=11-15)
- **Engineering applications**: Taylor method + L-BFGS, balancing accuracy and efficiency (N=15-21)
- **Special problems**:
  * Standard beam problem: Cubic Spline + Natural BC
  * Requiring derivative continuity: Hermite Spline + Clamped BC
"""

# ==================================================================================
# Core Physical Parameters
# ==================================================================================

# Geometry
h = 0.1          # beam thickness (m) - total thickness of the functionally graded material beam
L = 20 * h       # beam length (m) - default length-to-thickness ratio lambda=L/h=20
num_layers = 10  # number of layers of the functionally graded material - used for layer-wise distribution of material properties

# Materials - graphene-reinforced functionally graded composite
W_Gr = 0.025     # graphene mass fraction (0.025 = 2.5%) - converted to volume fraction for material property calculation
H_Gr = 0.8       # graphene shape factor - controls how the aspect ratio of graphene platelets affects the reinforcement effect
T = 300          # temperature (K) - current temperature, reference temperature T0=300K
distr_type = 'X' # GPL distribution type: 'X' (more at both ends, less in the middle), 'U' (uniform distribution), 'O' (more in the middle, less at both ends)

# Loading & Boundary Conditions
q = -0.08        # uniformly distributed load (dimensionless) - negative value means downward, already non-dimensionalized by A11_0
bc_type = 'C-C'  # boundary condition type:
                 # 'C-C': Clamped-Clamped (both ends clamped)
                 # 'S-S': Simply Supported (both ends simply supported)
                 # 'H-H': Hinged-Hinged (both ends hinged)
                 # 'C-S': Clamped-Simply Supported (one end clamped, one end simply supported)
                 # 'C-H': Clamped-Hinged (one end clamped, one end hinged)

# ==================================================================================
# Foundation Parameters
# ==================================================================================

# Elastic foundation parameters: model the elastic support beneath the beam
# Foundation reaction: q_foundation = -k1*w + k2*d^2w/dx^2
k1 = 0.01   # Winkler foundation stiffness (dimensionless):
            # recommended range: 0-200, interval 25
            # physical meaning: vertical spring support, larger k1 means stiffer support
k2 = 0.001   # Pasternak foundation stiffness (dimensionless):
            # recommended range: 0-200, interval 25
            # physical meaning: continuous foundation accounting for shear action, larger k2 means stronger shear interaction

# Organize parameter dictionaries
foundation_params = {'k1': k1, 'k2': k2}

# ==================================================================================
# Discretization Method Configuration
# ==================================================================================

# Method selection: 'dq', 'taylor', 'spline'
method = 'spline'

# DQ method parameter configuration (when method='dq')
# The differential quadrature method uses Chebyshev-Gauss-Lobatto nodes for global spectral approximation
if method == 'dq':
    N = 13                    # number of DQ nodes (11-15 is the optimal range, more nodes may cause the Runge phenomenon)
    dq_type = 'negative_sum'  # weighting matrix computation method:
                             # 'original': direct formula method, high accuracy but poor numerical stability
                             # 'negative_sum': negative sum rule, improves stability by enforcing the constraint sum(D_ij)=0
    dq_method = dq_type      # backward-compatibility parameter

# Taylor/Fornberg method parameters (when method='taylor')
# Uses the Fornberg algorithm to compute finite difference weights, supporting high-accuracy approximation of arbitrary-order derivatives
elif method == 'taylor':
    N = 15                        # number of nodes (15-25 is the recommended range, balancing accuracy and efficiency)
    taylor_stencil_size = 9       # local stencil size (odd number, 5-11, larger means higher accuracy)
    taylor_nodes = 'cheb'         # node distribution type:
                                 # 'cheb': Chebyshev nodes, reduce the Runge phenomenon
                                 # 'uniform': uniform nodes, simple but may oscillate
    taylor_sparse_format = 'dense'  # derivative matrix storage format:
                                   # 'dense': dense matrix, efficient when there are few nodes
                                   # 'coo': COO sparse format, for large-scale problems
                                   # 'csr': CSR sparse format, optimized for matrix operations

    # Backward-compatibility parameters (keep consistent with the old version interface)
    dq_method = 'fornberg_local'
    fd_stencil_size = taylor_stencil_size
    x_nodes = taylor_nodes
    fd_build_orders = (1, 2)     # derivative orders to build
    fd_build_C_D = False         # whether to build the first-order derivative matrix
    fd_B_from_A = False          # whether to derive the B matrix from the A matrix

# Spline method parameters (when method='spline')
# Spline collocation method, using piecewise polynomials to guarantee high-order continuity
elif method == 'spline':
    N = 15                        # number of spline collocation points (11-51, more points improve accuracy but increase computation)

    # Spline type selection (affects approximation characteristics and boundary handling)
    # NOTE: currently only 'b_spline' can be selected; the other spline types are not supported for now.
    # - 'cubic': cubic spline, C2 continuous, most commonly used
    # - 'quintic': quintic spline, C4 continuous, smoother
    # - 'b_spline': B-spline, local support, numerically stable
    # - 'tension': tension spline, can control oscillation
    # - 'hermite': Hermite spline, specifies derivative values
    spline_type = 'b_spline'

    # Boundary condition type (affects behavior at the endpoints)
    # - 'natural': natural boundary (second derivative is 0)
    # - 'clamped': clamped boundary (specifies first derivative)
    # - 'not-a-knot': not-a-knot condition (third derivative continuous)
    # - 'periodic': periodic boundary
    spline_boundary = 'natural'

    spline_degree = 7 if spline_type in ['b_spline', 'bspline', 'b-spline'] else None  # B-spline degree
    spline_tension = 0.0         # tension parameter (0=Catmull-Rom spline, 1=straight-line interpolation)
    bc_type_spline = spline_boundary  # spline boundary condition

    dq_method = 'spline'  # backward-compatibility identifier

# ==================================================================================
# Solver Configuration
# ==================================================================================

# Solve mode: 'linear', 'nonlinear', 'both'
mode = 'both'

# Optimizer selection: 'lbfgs', 'adam', 'gauss-newton', 'levenberg-marquardt'
# optim_name = "lbfgs"
# optim_name = "gauss-newton"
optim_name = "levenberg-marquardt"

# Maximum iterations configuration
# Automatically adjusted based on optimizer characteristics and boundary condition type
if optim_name.lower() in ['gauss-newton', 'gauss_newton', 'newton']:
    max_iter_linear = 100      # Gauss-Newton: second-order method, fast convergence (quadratic convergence rate)
    max_iter_nonlinear = 200   # the nonlinear problem requires more iterations
elif optim_name.lower() in ['lm', 'levenberg-marquardt', 'levenberg_marquardt']:
    max_iter_linear = 150      # LM algorithm: adaptive damping, robust but slightly slower than GN
    max_iter_nonlinear = 300   # number of iterations for the nonlinear problem
else:
    # First-order methods such as L-BFGS and Adam require more iterations
    if bc_type.upper() == 'C-C':
        max_iter_linear = 15000      # C-C boundary (clamped): strong constraints, relatively fast convergence
        max_iter_nonlinear = 16000
    else:
        max_iter_linear = 100000     # other boundary conditions: may require more iterations to converge
        max_iter_nonlinear = 70000

# Learning rate configuration (only some optimizers require a learning rate)
# Note: only L-BFGS and Adam require a learning rate; GN and LM are direct methods and do not
if optim_name.lower() == 'lbfgs':
    lr = 0.8  # L-BFGS: second-order method, requires a lower learning rate
elif optim_name.lower() == 'adam':
    lr = 0.01  # Adam: adaptive learning rate optimizer
else:
    # Gauss-Newton and Levenberg-Marquardt do not use a learning rate parameter
    # They are direct methods: GN uses -(J^TJ)^(-1)J^Tr, LM uses -(J^TJ+lambda*I)^(-1)J^Tr
    lr = 1.0  # set a default value to avoid code errors, but it is not actually used by these optimizers

# Fine-tuning for boundary conditions and load (keeps the original logic, only effective for optimizers that require a learning rate)
if bc_type.upper() == 'H-H' and abs(q) < 0.05 and optim_name.lower() in ['lbfgs', 'adam']:
    lr *= 1.0  # H-H boundary, small load: keep unchanged

print(f"[Parameter Configuration] Optimizer {optim_name}: {f'learning rate {lr}' if optim_name.lower() in ['lbfgs', 'adam'] else 'does not use a learning rate parameter'}")

# ==================================================================================
# Advanced Solver Options
# ==================================================================================

# Loss function weights
pde_weights = (1.0, 1.0, 1.0)  # PDE residual weights: (R1, R2, R3)
reg_weight = 1e-10              # regularization weight
bc_weight = 1000.0               # boundary condition weight (for soft constraints)

# Initial value strategy (used when mode='both')
use_linear_as_initial = True    # whether to use the linear solution as the nonlinear initial value
initial_value_scale = 0.6        # linear solution scaling factor (0-1)
initial_value_mix_ratio = 0.7    # mixing ratio of the linear solution and random perturbation (0-1)
num_solution_attempts = 3        # number of nonlinear solve attempts
validate_physical_solution = True  # whether to validate the physical plausibility of the solution

# Random perturbation restart strategy
perturbation_enabled = True          # whether to enable the perturbation strategy
perturbation_threshold = 0.0005      # stagnation detection threshold
perturbation_patience = 600          # stagnation tolerance count
perturbation_scale = 0.005           # perturbation magnitude
perturbation_scale_increment = 0.25  # perturbation growth factor

# Optimizer-specific parameters
gauss_newton_tol = 1e-10         # Gauss-Newton convergence tolerance
gauss_newton_damping = 0.1       # Gauss-Newton damping factor
lm_tol = 1e-8                    # LM algorithm convergence tolerance
lm_damping_init = 1e-3           # LM algorithm initial damping factor
lm_damping_factor = 10.0         # LM algorithm damping update factor

# ==================================================================================
# Output & Monitoring Configuration
# ==================================================================================

# Output control
print_every = 300                # iteration output frequency
save_every = 5000               # intermediate result saving frequency
verbose = True                  # verbose output mode
print_every_epoch = True        # L-BFGS prints the loss every epoch

# Logging system
log_mode = 'full'               # log mode: 'simple' or 'full'
log_level = 'INFO'              # log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
archived_logs_dir = 'archived_logs'  # archived logs directory

# GPU monitoring
enable_gpu_monitor = True       # whether to enable GPU monitoring
gpu_monitor_interval = 500      # GPU monitoring frequency

# Result saving
save_results = True             # whether to save results
results_dir = 'results'         # result saving directory
plot_dpi = 300                 # plot DPI
plot_format = 'png'            # plot format: 'png', 'pdf', 'svg'

# ==================================================================================
# Computing Environment Configuration
# ==================================================================================

# Compute device
use_cuda = True                 # whether to use CUDA (if available)
dtype_str = 'float64'
seed = 42                       # random seed (for result reproducibility)

# Test mode
test_mode = False               # test mode switch
test_N = 13                     # number of nodes in test mode
test_max_iter = 100             # maximum iterations in test mode

if test_mode:
    N = test_N
    max_iter_linear = test_max_iter
    max_iter_nonlinear = test_max_iter
    print_every = 10
    save_every = 50
