"""
Solution physical plausibility validation module

Used to validate the physical plausibility of linear and nonlinear solution results
"""

import torch
import warnings
from typing import Dict, Tuple

def validate_nonlinear_solution(linear_result: Dict, nonlinear_result: Dict, 
                               tolerance: float = 1.2) -> Tuple[bool, str]:
    """
    Validate the physical plausibility of the nonlinear solution

    Parameters:
        linear_result: linear solution result
        nonlinear_result: nonlinear solution result
        tolerance: tolerance factor (the maximum allowed multiple of the linear solution for the nonlinear solution)

    Returns:
        (is_valid, message): validation result and diagnostic information
    """
    # Extract maximum deflection
    linear_max = torch.abs(linear_result['w']).max().item()
    nonlinear_max = torch.abs(nonlinear_result['w']).max().item()

    # Check loss values
    linear_loss = linear_result.get('final_loss', float('inf'))
    nonlinear_loss = nonlinear_result.get('final_loss', float('inf'))

    # Validation rules
    is_valid = True
    messages = []

    # Rule 1: the nonlinear deflection should usually be smaller than the linear one (geometric stiffening effect)
    # but it may be slightly larger under certain loading conditions, so a certain tolerance is allowed
    if nonlinear_max > linear_max * tolerance:
        is_valid = False
        messages.append(f"Nonlinear deflection ({nonlinear_max:.6f}) exceeds {tolerance} times the linear deflection ({linear_max:.6f})")
        warnings.warn(f"Physical validation failed: {messages[-1]}")

    # Rule 2: the loss should be sufficiently small
    if nonlinear_loss > 1e-1:
        is_valid = False
        messages.append(f"Nonlinear solution loss too large: {nonlinear_loss:.3e}")

    # Rule 3: the solution should not contain NaN or Inf
    if torch.isnan(nonlinear_result['w']).any() or torch.isinf(nonlinear_result['w']).any():
        is_valid = False
        messages.append("Solution contains NaN or Inf values")

    # Generate diagnostic information
    if is_valid:
        message = f"Validation passed: linear deflection={linear_max:.6f}, nonlinear deflection={nonlinear_max:.6f}, ratio={nonlinear_max/linear_max:.3f}"
    else:
        message = "Validation failed: " + "; ".join(messages)
    
    return is_valid, message

def select_best_solution(solutions: list, linear_result: Dict = None,
                        require_physical_validity: bool = True) -> Tuple[Dict, int]:
    """
    Select the best solution from multiple solution results

    Parameters:
        solutions: list of solution results
        linear_result: linear solution (used for physical validation)
        require_physical_validity: whether physical plausibility is required

    Returns:
        (best_solution, best_index): the best solution and its index
    """
    if not solutions:
        raise ValueError("Solution list is empty")

    best_solution = None
    best_index = -1
    best_score = float('inf')

    for i, solution in enumerate(solutions):
        # Compute score (primarily based on loss)
        score = solution.get('final_loss', float('inf'))

        # If physical validation is required
        if require_physical_validity and linear_result is not None:
            is_valid, _ = validate_nonlinear_solution(linear_result, solution)
            if not is_valid:
                # Add a penalty for physically implausible solutions
                score *= 10

        # Check whether this is the current best
        if score < best_score:
            best_score = score
            best_solution = solution
            best_index = i

    if best_solution is None:
        # If no solution meets the requirements, return the one with the smallest loss
        best_index = 0
        best_solution = min(solutions, key=lambda s: s.get('final_loss', float('inf')))
    
    return best_solution, best_index

def print_solution_comparison(linear_result: Dict, nonlinear_result: Dict):
    """
    Print comparison information for the linear and nonlinear solutions

    Parameters:
        linear_result: linear solution result
        nonlinear_result: nonlinear solution result
    """
    linear_max = torch.abs(linear_result['w']).max().item()
    nonlinear_max = torch.abs(nonlinear_result['w']).max().item()

    linear_loss = linear_result.get('final_loss', 'N/A')
    nonlinear_loss = nonlinear_result.get('final_loss', 'N/A')

    print("\n[Solution comparison]")
    print(f"  Linear solution:")
    print(f"    Max deflection: {linear_max:.6f}")
    print(f"    Final loss: {linear_loss}")
    print(f"  Nonlinear solution:")
    print(f"    Max deflection: {nonlinear_max:.6f}")
    print(f"    Final loss: {nonlinear_loss}")
    print(f"  Deflection ratio (nonlinear/linear): {nonlinear_max/linear_max:.3f}")

    # Physical plausibility check
    is_valid, message = validate_nonlinear_solution(linear_result, nonlinear_result)
    print(f"  Physical validation: {message}")