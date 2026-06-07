"""
ODIL-FSDT beam bending solver main program

## Project overview
Based on the ODIL (Optimizing a Discrete Loss) framework, solves the bending problem of
functionally graded Timoshenko beams (FSDT - First-order Shear Deformation Theory)
considering elastic foundations.

## Supported discretization methods
1. **DQ (Differential Quadrature)**: differential quadrature method, spectral accuracy
2. **Taylor/Fornberg**: finite difference method based on Taylor series
3. **Spline**: spline collocation method (cubic, quintic, b-spline, etc.)

## Governing equations solved
### Linear problem (small deformation theory)
- 3 coupled partial differential equations: axial equilibrium, transverse equilibrium, bending moment equilibrium
- Neglects geometric nonlinearity terms

### Nonlinear problem (large deformation theory)
- Includes von Kármán type geometric nonlinearity terms
- Considers axial-transverse displacement coupling effects

## Special features
- **Elastic foundation**: Winkler (k₁) and Pasternak (k₂) foundation
- **Functionally graded material**: graphene-reinforced composite material (GPL-FGM)
- **Multiple boundary conditions**: C-C, S-S, H-H, C-S, C-H

## Output results
- Displacement field: axial u, transverse w, rotation φ
- Comparative analysis: linear vs nonlinear solutions
- Convergence history: loss function evolution
- Model saving: optimal parameter storage
"""

import os
import sys
import torch

# Resolve the conflict of duplicate OpenMP library loading on Windows systems
# This setting is needed when using multiple libraries that depend on OpenMP (such as MKL, OpenBLAS)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add the current project root directory to the Python search path to ensure modules import correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project core modules
import params
from modules.solver_odil import ODILSolver
from modules.material_properties import compute_material_params_for_solver
from utils.boundary_conditions import get_boundary_conditions, print_boundary_info
from utils.output_manager import create_output_manager, get_filename_base
from utils.common import initialize_computing_environment
from utils.nonlinear_solving import solve_nonlinear_with_strategy
from utils.solution_validation import print_solution_comparison

# Initialize the computing environment
# Set the random seed, select the computing device (CPU/GPU), specify numerical precision
device = initialize_computing_environment(
    seed=params.seed,  # Random seed, ensures reproducible results
    use_cuda=params.use_cuda,  # Whether to use CUDA GPU acceleration
    dtype=torch.float64 if params.dtype_str == 'float64' else torch.float32,  # Numerical precision: float64 (double precision) or float32 (single precision)
    verbose=False  # Avoid repeatedly printing initialization information
)

def main():
    """
    Main function - the entry point of the ODIL-Timoshenko beam solver

    Execution flow:
    1. Display program information and configuration summary
    2. Validate elastic foundation parameters
    3. Compute material parameters (based on composite laminate theory)
    4. Create ODIL solver instance
    5. Solve the problem according to the mode (linear/nonlinear/both)
    6. Save and visualize the results
    """

    print("="*60)
    print("ODIL-Timoshenko beam solver")
    print("Optimizing a Discrete Loss with Multiple Discretization Methods")
    print("Based on the ODIL framework: https://github.com/cselab/odil/")
    print("="*60)

    # Display core configuration summary
    print(f"\nCore configuration summary:")
    print(f"   Discretization method: {params.method.upper()}")
    print(f"   Optimizer: {params.optim_name.upper()}")
    print(f"   Boundary condition: {params.bc_type}")
    print(f"   Solving mode: {params.mode}")
    print("="*60)

    # First compute material parameters to obtain the lambda value (λ=L/h, length ratio)
    # This step needs to be executed in advance, because the lambda value is used to determine the output folder naming
    material_params_for_folder = compute_material_params_for_solver(
        h=params.h,
        L=params.L,
        num_layers=params.num_layers,
        W_Gr=params.W_Gr,
        H_Gr=params.H_Gr,
        T=params.T,
        distribution_type=params.distr_type,
        q=params.q
    )
    lambda_val_for_folder = material_params_for_folder.get('lambda_val', 20)

    # Create the output manager
    # Automatically organize the output folder structure according to material parameters, boundary conditions and distribution type
    # Folder structure: results/main/{bc_type}/{distribution_type}/{parameter_combination}/
    output_manager = create_output_manager(
        base_dir=params.results_dir,
        sub_dir='main',
        W_Gr=params.W_Gr,
        T=params.T,
        H_Gr=params.H_Gr,
        q=params.q,
        lambda_val=lambda_val_for_folder,
        h=params.h,
        L=params.L,
        foundation_params=params.foundation_params,
        config=params
    )
    
    # Validate parameter validity
    # Ensure elastic foundation parameters are within reasonable ranges
    print("\n[Parameter validation]")
    from utils.foundation_functions import validate_foundation_parameters
    from utils.foundation_functions import get_foundation_description

    # Validate elastic foundation parameters (k1: Winkler foundation stiffness, k2: Pasternak foundation stiffness)
    foundation_valid = validate_foundation_parameters(params.k1, params.k2)
    if foundation_valid:
        print("  [√] Elastic foundation parameter validation passed")
    else:
        print("  [×] Elastic foundation parameter validation failed")

    # Print parameter information
    print("\n[Parameter configuration]")
    print(f"  Beam thickness h = {params.h} m")
    print(f"  Beam length L = {params.L} m")
    print(f"  Graphene mass fraction W_Gr = {params.W_Gr}")
    print(f"  Graphene shape factor H_Gr = {params.H_Gr}")
    print(f"  Temperature T = {params.T} K")
    print(f"  Uniformly distributed load q = {params.q}")
    print(f"  Boundary condition: {params.bc_type}")
    print(f"  Number of nodes: N = {params.N}")

    # Print elastic foundation configuration
    print("\n[Elastic foundation configuration]")
    print(f"  Elastic foundation: {get_foundation_description(params.k1, params.k2)}")

    # Display discretization method information (highlighted)
    print("\n" + "-"*50)
    print(f"[Currently used discretization method]: {params.method.upper()}")
    print("-"*50)
    from modules.method_factory import print_method_info
    print_method_info(params.method, params)
    print(f"  Solving mode: {params.mode}")

    # Print detailed boundary condition information
    print_boundary_info(params.bc_type)

    # Use the already computed material parameters (graphene composite based on Halpin-Tsai theory)
    print("\n[Compute material parameters]")
    material_params = material_params_for_folder
    material_params.pop('q', None)  # Remove the uniformly distributed load q, pass it as an independent parameter
    
    print(f"  a11 = {material_params['a11']:.6f}")
    print(f"  b11 = {material_params['b11']:.6f}")
    print(f"  d11 = {material_params['d11']:.6f}")
    print(f"  a55 = {material_params['a55']:.6f}")
    print(f"  lambda = {material_params['lambda_val']:.2f}")
    print(f"  n_xT = {material_params['n_xT']:.6f}")
    
    # Get boundary conditions (C: Clamped, H: Hinged, S: Simply Supported)
    bcs = get_boundary_conditions(bc_type=params.bc_type, device=device)

    # Get the log directory (used to save training process logs)
    log_dir = output_manager.get_logs_dir()

    # Create the ODIL solver instance
    print("\n[Create ODIL solver]")
    # Display relevant configuration information according to the selected discretization method
    if params.method == 'dq':
        if params.dq_method == 'negative_sum':
            print("  DQ method: use the negative sum rule to compute the weighting matrix")
        else:
            print("  DQ method: use the original method (direct formula) to compute the weighting matrix")
    elif params.method == 'taylor':
        print(f"  Taylor method: stencil size={params.taylor_stencil_size}")

    solver = ODILSolver(N=params.N, material_params=material_params, device=device,
                       dq_method=params.dq_method, bc_type=params.bc_type,
                       log_dir=log_dir, config=params)
    
    # Generate the base name for output files (contains all key parameters)
    base_name = get_filename_base(W_Gr=params.W_Gr, T=params.T, H_Gr=params.H_Gr, q=params.q, lambda_val=lambda_val_for_folder)

    # Display the currently used optimizer (LBFGS, Adam, etc.)
    print("\n" + "="*60)
    print(f"[Currently used optimizer]: {params.optim_name.upper()}")
    print("="*60)

    # Execute the corresponding solving flow according to the solving mode (linear/nonlinear/both)
    if params.mode == 'linear':
        print("\n[Solve linear problem]")
        result = solver.solve(
            bcs=bcs,
            q=params.q,
            is_nonlinear=False,
            optim_name=params.optim_name,
            max_iter=params.max_iter_linear,
            lr=params.lr,
            loss_weights=params.pde_weights,
            reg=params.reg_weight,
            verbose=params.verbose,
            bc_weight=getattr(params, 'bc_weight', 10.0),
            print_every=params.print_every,
            gpu_monitor_interval=params.gpu_monitor_interval
        )
        
        # Save all computation results (including displacement data, model files, loss history)
        output_manager.save_all_results(result=result, mode='linear', base_name=base_name, dpi=params.plot_dpi)

        # Print the result summary (display maximum displacement, convergence status, etc.)
        output_manager.print_summary(linear_result=result)

    elif params.mode == 'nonlinear':
        print("\n[Solve nonlinear problem]")
        result = solver.solve(
            bcs=bcs,
            q=params.q,
            is_nonlinear=True,
            optim_name=params.optim_name,
            max_iter=params.max_iter_nonlinear,
            lr=params.lr,
            loss_weights=params.pde_weights,
            reg=params.reg_weight,
            verbose=params.verbose,
            bc_weight=getattr(params, 'bc_weight', 10.0),
            print_every=params.print_every,
            gpu_monitor_interval=params.gpu_monitor_interval
        )
        
        # Save results
        output_manager.save_all_results(result=result, mode='nonlinear', base_name=base_name, dpi=params.plot_dpi)

        # Print the result summary
        output_manager.print_summary(nonlinear_result=result)

    elif params.mode == 'both':
        print("\n[Solve both linear and nonlinear problems]")

        # Step 1: First solve the linear problem to obtain a baseline solution
        print("\n[1/2] Solving linear problem...")
        linear_result = solver.solve(
            bcs=bcs,
            q=params.q,
            is_nonlinear=False,
            optim_name=params.optim_name,
            max_iter=params.max_iter_linear,
            lr=params.lr,
            loss_weights=params.pde_weights,
            reg=params.reg_weight,
            verbose=params.verbose,
            bc_weight=getattr(params, 'bc_weight', 10.0),
            print_every=params.print_every,
            gpu_monitor_interval=params.gpu_monitor_interval
        )
        
        # Step 2: Solve the nonlinear problem (the linear solution can be used as the initial value)
        print("\n[2/2] Solving nonlinear problem...")

        # Use the intelligent nonlinear solving strategy (includes multiple initial-value attempts and adaptive adjustment)
        nonlinear_result = solve_nonlinear_with_strategy(
            solver=solver,
            bcs=bcs,
            q=params.q,
            linear_result=linear_result,
            params=params,
            verbose=params.verbose,
            verbose_attempts=True  # Display attempt information
        )

        # Optional: perform physical plausibility validation (check the influence of nonlinear effects)
        if params.validate_physical_solution:
            print_solution_comparison(linear_result=linear_result, nonlinear_result=nonlinear_result)

        # Saving and plotting strategy:
        # - Separately plot the figures of the linear and nonlinear solutions (each with its own displacement contour)
        # - Save data files in a merged manner (to avoid duplicate storage)
        output_manager.plot_single_solution(linear_result, mode='linear', base_name=base_name, plot_dpi=params.plot_dpi)
        output_manager.plot_single_solution(nonlinear_result, mode='nonlinear', base_name=base_name, plot_dpi=params.plot_dpi)

        # Save the merged data file and loss history
        # Merge the data of the linear and nonlinear solutions into one CSV file, convenient for comparative analysis
        try:
            # Save the merged displacement data (contains linear and nonlinear u, w, phi)
            output_manager.save_combined_solution_data(
                linear_result=linear_result,
                nonlinear_result=nonlinear_result,
                base_name=base_name
            )
            # Save the merged loss history
            output_manager.save_combined_loss_history(
                linear_loss_history=linear_result.get('loss_history', []),
                nonlinear_loss_history=nonlinear_result.get('loss_history', []),
                base_name=base_name
            )
            # Plot the loss comparison figure (display the convergence process of linear and nonlinear)
            output_manager.plot_combined_loss_history(
                linear_loss_history=linear_result.get('loss_history', []),
                nonlinear_loss_history=nonlinear_result.get('loss_history', []),
                base_name=base_name,
                plot_dpi=params.plot_dpi
            )
        except Exception as e:
            print(f"[Warning] Merged data saving failed: {e}")

        # Plot the comparison figure of the linear and nonlinear solutions
        # Display the displacement distributions of both solutions in the same figure, for intuitive comparison of nonlinear effects
        output_manager.plot_comparison(
            linear_result=linear_result,
            nonlinear_result=nonlinear_result,
            base_name=base_name,
            plot_dpi=params.plot_dpi
        )

        # Print the comprehensive result summary (contains key information of the linear and nonlinear solutions)
        output_manager.print_summary(linear_result=linear_result, nonlinear_result=nonlinear_result)

    print("\n" + "="*60)
    print("All computations complete!")
    print("="*60)

if __name__ == "__main__":
    main()
