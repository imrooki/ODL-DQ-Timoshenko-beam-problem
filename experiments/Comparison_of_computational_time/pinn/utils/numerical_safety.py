"""
Numerical Safety Utils

Provides key functionality such as division-by-zero protection, NaN/Inf
detection, and numerical stability monitoring, ensuring the numerical safety and
stability of scientific computations.
"""

import warnings
from typing import Union, Optional
import torch
import numpy as np


class NumericalSafetyError(Exception):
    """Base class for numerical safety exceptions"""
    pass


class NaNDetectedError(NumericalSafetyError):
    """Exception raised when a NaN value is detected"""
    pass


class InfDetectedError(NumericalSafetyError):
    """Exception raised when an Inf value is detected"""
    pass


class DivisionByZeroError(NumericalSafetyError):
    """Division-by-zero exception"""
    pass


class NumericalSafety:
    """Numerical safety utility class

    Provides common numerical safety protection features in scientific computing:
    - Safe division (division-by-zero protection)
    - NaN/Inf detection and handling
    - Numerical range checking
    - Gradient explosion detection
    """
    
    @staticmethod
    def safe_divide(numerator: torch.Tensor, 
                   denominator: Union[torch.Tensor, float], 
                   eps: float = 1e-10,
                   method: str = 'add_eps') -> torch.Tensor:
        """Safe division operation, avoiding division-by-zero errors

        Args:
            numerator: Numerator tensor
            denominator: Denominator tensor or scalar
            eps: Small value to prevent division by zero
            method: Protection method
                - 'add_eps': Add a small value to the denominator (recommended)
                - 'clamp': Clamp the minimum value of the denominator
                - 'sign_preserving': Sign-preserving safe division

        Returns:
            Safe division result

        Raises:
            DivisionByZeroError: When the denominator is exactly zero and cannot be fixed
        """
        if isinstance(denominator, (int, float)):
            if abs(denominator) < eps:
                if denominator == 0:
                    raise DivisionByZeroError(f"Division by exact zero: {denominator}")
                warnings.warn(f"Small denominator detected: {denominator}, using eps protection")
            denominator = torch.tensor(denominator, dtype=numerator.dtype, device=numerator.device)
        
        if method == 'add_eps':
            # Method 1: Add-small-value protection on the denominator (default method)
            safe_denom = denominator + eps * torch.sign(denominator)
            # Handle the case where the denominator is zero
            zero_mask = torch.abs(denominator) < eps
            safe_denom = torch.where(zero_mask, eps, safe_denom)

        elif method == 'clamp':
            # Method 2: Clamp the minimum absolute value of the denominator
            safe_denom = torch.clamp(torch.abs(denominator), min=eps) * torch.sign(denominator)
            safe_denom = torch.where(denominator == 0, eps, safe_denom)

        elif method == 'sign_preserving':
            # Method 3: Sign-preserving exact division protection
            abs_denom = torch.abs(denominator)
            sign_denom = torch.sign(denominator)
            safe_abs_denom = torch.maximum(abs_denom, torch.tensor(eps, device=denominator.device))
            safe_denom = safe_abs_denom * sign_denom
            
        else:
            raise ValueError(f"Unknown safe division method: {method}")
        
        result = numerator / safe_denom

        # Validate the validity of the result
        NumericalSafety.check_tensor_validity(result, f"safe_divide_result")
        
        return result
    
    @staticmethod
    def check_tensor_validity(tensor: torch.Tensor, 
                            name: str = "tensor",
                            raise_on_nan: bool = True,
                            raise_on_inf: bool = True,
                            max_abs_value: float = 1e10) -> bool:
        """Check the numerical validity of a tensor

        Args:
            tensor: Tensor to be checked
            name: Tensor name (used for error messages)
            raise_on_nan: Whether to raise an exception when NaN is found
            raise_on_inf: Whether to raise an exception when Inf is found
            max_abs_value: Maximum absolute value limit for the values

        Returns:
            Whether the tensor is valid

        Raises:
            NaNDetectedError: A NaN value is detected
            InfDetectedError: An Inf value is detected
        """
        # Check for NaN
        nan_count = torch.isnan(tensor).sum().item()
        if nan_count > 0:
            message = f"{name} contains {nan_count} NaN values out of {tensor.numel()} elements"
            if raise_on_nan:
                raise NaNDetectedError(message)
            else:
                warnings.warn(message)
                return False
        
        # Check for Inf
        inf_count = torch.isinf(tensor).sum().item()
        if inf_count > 0:
            message = f"{name} contains {inf_count} Inf values out of {tensor.numel()} elements"
            if raise_on_inf:
                raise InfDetectedError(message)
            else:
                warnings.warn(message)
                return False
        
        # Check the numerical range
        max_val = torch.abs(tensor).max().item()
        if max_val > max_abs_value:
            message = f"{name} contains very large values (max: {max_val:.2e}), potential overflow risk"
            warnings.warn(message)
        
        return True
    
    @staticmethod
    def safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Safe square-root operation

        Args:
            x: Input tensor
            eps: Small value to avoid taking the square root of a negative number

        Returns:
            Safe square-root result
        """
        safe_x = torch.clamp(x, min=eps)
        result = torch.sqrt(safe_x)
        NumericalSafety.check_tensor_validity(result, "safe_sqrt_result")
        return result
    
    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Safe logarithm operation

        Args:
            x: Input tensor
            eps: Small value to avoid taking the logarithm of zero or a negative number

        Returns:
            Safe logarithm result
        """
        safe_x = torch.clamp(x, min=eps)
        result = torch.log(safe_x)
        NumericalSafety.check_tensor_validity(result, "safe_log_result")
        return result
    
    @staticmethod
    def check_gradient_health(model: torch.nn.Module, 
                            max_grad_norm: float = 10.0,
                            name: str = "model") -> dict:
        """Check the health of the model gradients

        Args:
            model: PyTorch model
            max_grad_norm: Gradient norm threshold
            name: Model name

        Returns:
            Gradient statistics dictionary
        """
        stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'nan_grad_count': 0,
            'inf_grad_count': 0,
            'max_grad_norm': 0.0,
            'mean_grad_norm': 0.0,
            'is_healthy': True
        }
        
        grad_norms = []
        
        for param_name, param in model.named_parameters():
            stats['total_params'] += 1
            
            if param.grad is not None:
                stats['params_with_grad'] += 1
                grad = param.grad.data
                
                # Check for NaN gradients
                if torch.isnan(grad).any():
                    stats['nan_grad_count'] += 1
                    stats['is_healthy'] = False
                    warnings.warn(f"NaN gradient detected in {name}.{param_name}")

                # Check for Inf gradients
                if torch.isinf(grad).any():
                    stats['inf_grad_count'] += 1
                    stats['is_healthy'] = False
                    warnings.warn(f"Inf gradient detected in {name}.{param_name}")

                # Compute the gradient norm
                grad_norm = grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm > max_grad_norm:
                    stats['is_healthy'] = False
                    warnings.warn(f"Large gradient norm {grad_norm:.2e} in {name}.{param_name}")
        
        if grad_norms:
            stats['max_grad_norm'] = max(grad_norms)
            stats['mean_grad_norm'] = np.mean(grad_norms)
        
        return stats
    
    @staticmethod  
    def safe_tensor_operation(func, *tensors, operation_name: str = "tensor_operation"):
        """Safe tensor operation wrapper

        Args:
            func: Tensor operation function to execute
            *tensors: Input tensors
            operation_name: Operation name

        Returns:
            Operation result
        """
        # Pre-check the input tensors
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor):
                NumericalSafety.check_tensor_validity(tensor, f"{operation_name}_input_{i}")

        # Execute the operation
        try:
            result = func(*tensors)
        except Exception as e:
            raise NumericalSafetyError(f"Error in {operation_name}: {str(e)}")

        # Check the output
        if isinstance(result, torch.Tensor):
            NumericalSafety.check_tensor_validity(result, f"{operation_name}_output")
        elif isinstance(result, (tuple, list)):
            for i, out_tensor in enumerate(result):
                if isinstance(out_tensor, torch.Tensor):
                    NumericalSafety.check_tensor_validity(out_tensor, f"{operation_name}_output_{i}")
        
        return result


# Convenience functions
def check_nan_inf(tensor, name="tensor"):
    """Global NaN/Inf check function"""
    return NumericalSafety.check_tensor_validity(tensor, name)


# Test function
if __name__ == "__main__":
    print("Numerical safety utilities test...")

    # Import safe_divide from the correct module
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
    from numerics import safe_divide

    # Test safe division
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 1e-12])

    try:
        result = safe_divide(a, b)
        print(f"Safe division test passed: {result}")
    except Exception as e:
        print(f"Safe division test failed: {e}")

    # Test NaN detection
    nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
    try:
        check_nan_inf(nan_tensor, "test_tensor")
    except NaNDetectedError as e:
        print(f"NaN detection working: {e}")

    print("Numerical safety utilities module test complete!")