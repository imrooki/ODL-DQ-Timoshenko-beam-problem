

import torch
import warnings
from typing import Dict, Tuple

def validate_nonlinear_solution(linear_result: Dict, nonlinear_result: Dict, 
                               tolerance: float = 1.2) -> Tuple[bool, str]:
    
    
    linear_max = torch.abs(linear_result['w']).max().item()
    nonlinear_max = torch.abs(nonlinear_result['w']).max().item()

    
    linear_loss = linear_result.get('final_loss', float('inf'))
    nonlinear_loss = nonlinear_result.get('final_loss', float('inf'))

    
    is_valid = True
    messages = []

    
    
    if nonlinear_max > linear_max * tolerance:
        is_valid = False
        messages.append(f"Nonlinear deflection ({nonlinear_max:.6f}) exceeds {tolerance} times the linear deflection ({linear_max:.6f})")
        warnings.warn(f"Physical validation failed: {messages[-1]}")

    
    if nonlinear_loss > 1e-1:
        is_valid = False
        messages.append(f"Nonlinear solution loss too large: {nonlinear_loss:.3e}")

    
    if torch.isnan(nonlinear_result['w']).any() or torch.isinf(nonlinear_result['w']).any():
        is_valid = False
        messages.append("Solution contains NaN or Inf values")

    
    if is_valid:
        message = f"Validation passed: linear deflection={linear_max:.6f}, nonlinear deflection={nonlinear_max:.6f}, ratio={nonlinear_max/linear_max:.3f}"
    else:
        message = "Validation failed: " + "; ".join(messages)
    
    return is_valid, message

def select_best_solution(solutions: list, linear_result: Dict = None,
                        require_physical_validity: bool = True) -> Tuple[Dict, int]:
    
    if not solutions:
        raise ValueError("Solution list is empty")

    best_solution = None
    best_index = -1
    best_score = float('inf')

    for i, solution in enumerate(solutions):
        
        score = solution.get('final_loss', float('inf'))

        
        if require_physical_validity and linear_result is not None:
            is_valid, _ = validate_nonlinear_solution(linear_result, solution)
            if not is_valid:
                
                score *= 10

        
        if score < best_score:
            best_score = score
            best_solution = solution
            best_index = i

    if best_solution is None:
        
        best_index = 0
        best_solution = min(solutions, key=lambda s: s.get('final_loss', float('inf')))
    
    return best_solution, best_index

def print_solution_comparison(linear_result: Dict, nonlinear_result: Dict):
    
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

    
    is_valid, message = validate_nonlinear_solution(linear_result, nonlinear_result)
    print(f"  Physical validation: {message}")