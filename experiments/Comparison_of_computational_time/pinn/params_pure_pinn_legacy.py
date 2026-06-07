#!/usr/bin/env python3
"""
Pure PINN - Unified parameter configuration

Author: Yang
Version: 1.0

Description:
Pure PINN training configuration (no pseudo-supervision, no transfer learning)
Training is performed with the MC integrator
"""

# ============================================================================
# Physical Parameters
# ============================================================================

h = 0.1                    # Beam thickness (m)
L_factor = 20              # Beam length-to-thickness ratio (L/h)
L = L_factor * h           # Beam length (m)
num_layers = 10            # Number of material layers

W_Gr = 0.025              # Graphene mass fraction
H_Gr = 0.8                # Graphene shape factor
T = 300                   # Temperature (K)

q = -0.08                 # Dimensionless distributed load

bc_type = 'C-C'           # Boundary condition type
distribution = 'X'        # Material distribution type

k1 = 0.01                 # Elastic foundation Winkler stiffness
k2 = 0.001                # Elastic foundation Pasternak stiffness

# Boundary constraint type
lifting_basis = 'poly'    # Boundary constraint lifting basis function type ('poly', 'trig', 'none')

# ============================================================================
# Training Parameters
# ============================================================================

epochs = 100000           # Number of training epochs
lr = 8e-5                 # Learning rate
seed = 42                 # Random seed

use_adaptive_lr = False   # Disable adaptive learning rate
optimizer_type = 'Adam'

# Network architecture
network_arch = 'shared'
encoder_dims_shared = [1, 32, 64, 128]
head_dims = [128, 64, 32, 1]
input_dim = 1
bc_weight = 1000.0

# Activation function
activation_type = 'Tanh'
siren_omega_0 = 30.0
siren_omega_hidden = 30.0

# ============================================================================
# Pure PINN specific configuration - disable pseudo-supervision and transfer learning
# ============================================================================

use_pseudo_supervision = False  # Key: disable pseudo-supervision
transfer_freq = 0               # No transfer learning

# Pseudo-supervision weight parameters (disabled but kept to avoid errors)
ps_w_non_start = 1.0
ps_w_lin_start = 0.5
ps_cut_ratio = 0.8
ps_use_phi = False

# Transfer learning parameters (disabled but kept to avoid errors)
transfer_alpha = 0.3
transfer_ratio = 0.7
transfer_cut_ratio = 0.2

# Adaptive learning rate parameters (inactive when disabled, but kept to avoid errors)
lr_early_max = 1e-3
lr_early_min = 2e-4
lr_mid_max = 2e-4
lr_mid_min = 1e-4
lr_late_fixed = 1e-4
lr_patience = 500
lr_improvement_threshold = 1e-6
lr_decay_factor = 0.5
lr_verbose = True
lr_warmup_epochs = 100
lr_early_ratio = 0.6
lr_mid_ratio = 0.85
lr_min_early_epochs = 1000
lr_min_mid_epochs = 5000

# ============================================================================
# Integrator configuration - pure PINN uses MC integration
# ============================================================================

integrator = 'mc'         # Use Monte Carlo integration (standard pure PINN configuration)
N_train = 21              # Number of training sample points
sampler = 'uniform'
sampler_reuse = True

# AGQ parameters (unused with MC integration, but kept to avoid errors)
agq_rule = 'G10K21'
agq_abs_tol = 1e-6
agq_rel_tol = 1e-4
agq_max_points = 21
agq_max_depth = 0
agq_refine_every = 0
agq_fail_policy = 'use_partial'

# ============================================================================
# Output parameters
# ============================================================================

generate_plots = True
plot_interval = 1000
save_best_model = True
save_model_interval = 5000
print_every = 1000
verbose = True


def get_params() -> dict:
    """Get the complete parameter dictionary for the pure PINN method"""
    return {
        # Physical parameters
        'h': h, 'L': L, 'L_factor': L_factor, 'num_layers': num_layers,
        'W_Gr': W_Gr, 'H_Gr': H_Gr, 'T': T, 'q': q,
        'bc_type': bc_type, 'distribution': distribution,
        'k1': k1, 'k2': k2,
        'lifting_basis': lifting_basis,
        # Training parameters
        'epochs': epochs, 'lr': lr, 'seed': seed,
        'use_adaptive_lr': use_adaptive_lr, 'optimizer_type': optimizer_type,
        'network_arch': network_arch,
        'encoder_dims_shared': encoder_dims_shared,
        'head_dims': head_dims, 'input_dim': input_dim,
        'bc_weight': bc_weight,
        'activation_type': activation_type,
        'siren_omega_0': siren_omega_0,
        'siren_omega_hidden': siren_omega_hidden,
        # Pure PINN key configuration
        'use_pseudo_supervision': use_pseudo_supervision,
        'transfer_freq': transfer_freq,
        # Pseudo-supervision parameters (disabled)
        'ps_w_non_start': ps_w_non_start, 'ps_w_lin_start': ps_w_lin_start,
        'ps_cut_ratio': ps_cut_ratio, 'ps_use_phi': ps_use_phi,
        # Transfer learning parameters (disabled)
        'transfer_alpha': transfer_alpha, 'transfer_ratio': transfer_ratio,
        'transfer_cut_ratio': transfer_cut_ratio,
        # Adaptive learning rate parameters
        'lr_early_max': lr_early_max, 'lr_early_min': lr_early_min,
        'lr_mid_max': lr_mid_max, 'lr_mid_min': lr_mid_min,
        'lr_late_fixed': lr_late_fixed, 'lr_patience': lr_patience,
        'lr_improvement_threshold': lr_improvement_threshold,
        'lr_decay_factor': lr_decay_factor, 'lr_verbose': lr_verbose,
        'lr_warmup_epochs': lr_warmup_epochs, 'lr_early_ratio': lr_early_ratio,
        'lr_mid_ratio': lr_mid_ratio, 'lr_min_early_epochs': lr_min_early_epochs,
        'lr_min_mid_epochs': lr_min_mid_epochs,
        # Integrator parameters
        'integrator': integrator, 'N_train': N_train,
        'sampler': sampler, 'sampler_reuse': sampler_reuse,
        'agq_rule': agq_rule, 'agq_abs_tol': agq_abs_tol,
        'agq_rel_tol': agq_rel_tol, 'agq_max_points': agq_max_points,
        'agq_max_depth': agq_max_depth, 'agq_refine_every': agq_refine_every,
        'agq_fail_policy': agq_fail_policy,
        # Output parameters
        'generate_plots': generate_plots, 'plot_interval': plot_interval,
        'save_best_model': save_best_model, 'save_model_interval': save_model_interval,
        'print_every': print_every, 'verbose': verbose,
    }


def print_config_summary():
    """Print the configuration summary"""
    print("\n" + "=" * 60)
    print("Pure PINN - Configuration Summary")
    print("=" * 60)
    print(f"\n[Physical Parameters]")
    print(f"  Geometry: h={h}m, L/h={L_factor}")
    print(f"  Material: W_Gr={W_Gr}, H_Gr={H_Gr}, T={T}K")
    print(f"  Load: q={q}")
    print(f"  Boundary: {bc_type}, Distribution: {distribution}")
    print(f"  Foundation: k1={k1}, k2={k2}")
    print(f"\n[Training Parameters]")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Network: {encoder_dims_shared} -> {head_dims}")
    print(f"  Activation: {activation_type}")
    print(f"\n[Pure PINN Configuration]")
    print(f"  Pseudo-Supervision: DISABLED")
    print(f"  Transfer Learning: DISABLED")
    print(f"  Integrator: {integrator.upper()} ({N_train} points)")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
