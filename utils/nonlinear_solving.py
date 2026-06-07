"""
Nonlinear solving strategy module

Provides a unified nonlinear solving strategy, including:
- Using the linear solution as the initial guess
- Multiple-attempt mechanism
- Physical validation
- Selecting the best solution
"""

import torch
from typing import Dict
from utils.solution_validation import select_best_solution, validate_nonlinear_solution


def solve_nonlinear_with_strategy(
    solver,
    bcs: Dict,
    q: float,
    linear_result: Dict,
    params,
    optim_name: str = None,
    max_iter: int = None,
    lr: float = None,
    loss_weights: tuple = None,
    reg_weight: float = None,
    verbose: bool = False,
    verbose_attempts: bool = True
) -> Dict:
    """
    Solve the nonlinear problem using an improved strategy

    Parameters:
        solver: ODIL solver instance
        bcs: boundary conditions
        q: load
        linear_result: linear solution result
        params: parameter object (must include use_linear_as_initial, etc.)
        optim_name: optimizer name
        max_iter: maximum number of iterations
        lr: learning rate
        loss_weights: PDE weights
        reg_weight: regularization weight
        verbose: whether to display detailed iteration information
        verbose_attempts: whether to display attempt information

    Returns:
        the best nonlinear solution result
    """
    # Get default parameters
    if optim_name is None:
        optim_name = params.optim_name
    if max_iter is None:
        max_iter = params.max_iter_nonlinear
    if lr is None:
        lr = params.lr
    if loss_weights is None:
        loss_weights = params.pde_weights
    if reg_weight is None:
        reg_weight = params.reg_weight
    
    # Check whether to use the improved initial guess strategy
    if params.use_linear_as_initial and params.num_solution_attempts > 1:
        if verbose_attempts:
            print(f"  Using improved strategy: scaling={params.initial_value_scale}, "
                  f"mix={params.initial_value_mix_ratio}, "
                  f"attempts={params.num_solution_attempts}")

        # Prepare initial guess
        initial_guess = {
            'u_inner': linear_result['u'][1:-1].detach().clone(),
            'w_inner': linear_result['w'][1:-1].detach().clone(),
            'phi_inner': linear_result['phi'][1:-1].detach().clone()
        }
        
        # Multiple attempts
        solutions = []
        for attempt in range(params.num_solution_attempts):
            if verbose and verbose_attempts:
                print(f"\n  Attempt {attempt+1}/{params.num_solution_attempts}...")

            # Each attempt has a random perturbation (mixed inside the solver)
            result = solver.solve(
                bcs=bcs,
                q=q,
                is_nonlinear=True,
                optim_name=optim_name,
                max_iter=max_iter,
                lr=lr,
                loss_weights=loss_weights,
                reg=reg_weight,
                verbose=verbose,
                initial_guess=initial_guess
            )
            
            solutions.append(result)

            # Validate and display the result
            if params.validate_physical_solution and verbose_attempts:
                is_valid, msg = validate_nonlinear_solution(linear_result, result)
                if verbose:
                    print(f"    Loss: {result.get('final_loss', 'N/A'):.3e}")
                    print(f"    Max deflection: {torch.abs(result['w']).max().item():.6f}")
                    print(f"    Physical validation: {'passed' if is_valid else 'failed'}")

        # Select the best solution
        nonlinear_result, best_idx = select_best_solution(
            solutions, 
            linear_result,
            require_physical_validity=params.validate_physical_solution
        )
        
        if verbose_attempts:
            print(f"\n  Selecting the result of attempt {best_idx+1} as the final solution")

    elif params.use_linear_as_initial:
        # Single attempt but using the linear solution as the initial guess
        if verbose_attempts:
            print("  Using the linear solution as the initial guess (single attempt)")

        initial_guess = {
            'u_inner': linear_result['u'][1:-1].detach().clone(),
            'w_inner': linear_result['w'][1:-1].detach().clone(),
            'phi_inner': linear_result['phi'][1:-1].detach().clone()
        }
        
        nonlinear_result = solver.solve(
            bcs=bcs,
            q=q,
            is_nonlinear=True,
            optim_name=optim_name,
            max_iter=max_iter,
            lr=lr,
            loss_weights=loss_weights,
            reg=reg_weight,
            verbose=verbose,
            initial_guess=initial_guess
        )
        
        # Validate physical plausibility
        if params.validate_physical_solution and verbose_attempts:
            is_valid, msg = validate_nonlinear_solution(linear_result, nonlinear_result)
            if not is_valid:
                print(f"    Warning: {msg}")

    else:
        # Original random initialization
        if verbose_attempts:
            print("  Using random initialization")
        nonlinear_result = solver.solve(
            bcs=bcs,
            q=q,
            is_nonlinear=True,
            optim_name=optim_name,
            max_iter=max_iter,
            lr=lr,
            loss_weights=loss_weights,
            reg=reg_weight,
            verbose=verbose
        )
    
    return nonlinear_result


