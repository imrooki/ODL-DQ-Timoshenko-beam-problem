


h = 0.1
L = 20 * h
num_layers = 10


W_Gr = 0.025
H_Gr = 0.8
T = 300
distr_type = 'X'


q = -0.08
bc_type = 'C-C'


k1 = 0.01
k2 = 0.001
foundation_params = {'k1': k1, 'k2': k2}


method = 'dq'
N = 13
dq_type = 'negative_sum'
dq_method = dq_type


mode = 'both'  


pde_weights = (1.0, 1.0, 1.0)
reg_weight = 1e-8
bc_weight = 1000.0


use_linear_as_initial = True
initial_value_scale = 0.6
initial_value_mix_ratio = 0.7
num_solution_attempts = 3
validate_physical_solution = False




INIT_STRATEGIES = ["linear", "random"]


perturbation_enabled = True
perturbation_threshold = 0.0005
perturbation_patience = 600
perturbation_scale = 0.005
perturbation_scale_increment = 0.25


gauss_newton_tol = 1e-10
gauss_newton_damping = 0.1
lm_tol = 1e-8
lm_damping_init = 1e-3
lm_damping_factor = 10.0


verbose = False         
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


use_cuda = True
dtype_str = 'float64'
seed = 42


scenarios = [
    {'name': 'with_foundation', 'k1': 0.01, 'k2': 0.001,
     'description': 'Winkler-Pasternak foundation (paper default)'},
    {'name': 'no_foundation',   'k1': 0.0,  'k2': 0.0,
     'description': 'No elastic foundation (isolated benchmark)'},
]

optimizers = [
    {
        
        'name': 'levenberg-marquardt',
        'max_iter_linear': 150,
        'max_iter_nonlinear': 300,
        'lr': 1.0,           
    },
    {
        'name': 'lbfgs',
        'max_iter_linear': 30000,    
        'max_iter_nonlinear': 50000,
        'lr': 0.8,                   
    },
    {
        
        
        'name': 'gauss-newton',
        'max_iter_linear': 100,
        'max_iter_nonlinear': 200,
        'lr': 1.0,                   
    },
]
