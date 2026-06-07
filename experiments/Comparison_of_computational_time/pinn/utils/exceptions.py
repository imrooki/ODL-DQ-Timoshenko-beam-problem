"""
Exception System

Defines the project-specific exception class hierarchy, providing rich error
information and recovery suggestions so that errors can be accurately diagnosed
and handled.
"""

import traceback
import logging
from typing import Optional, Dict, Any, List
import torch


# ============================================================================
# Exception hierarchy
# ============================================================================

class TimoshenkoError(Exception):
    """Base exception class for the Timoshenko beam PINNs project"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 suggestions: Optional[List[str]] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.suggestions = suggestions or []
        self.context = context or {}
        
    def __str__(self) -> str:
        result = f"[{self.error_code}] {self.message}"
        
        if self.context:
            result += f"\nContext: {self.context}"
            
        if self.suggestions:
            result += "\nSuggested solutions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                result += f"\n  {i}. {suggestion}"
                
        return result


class ConfigurationError(TimoshenkoError):
    """Configuration error"""
    pass


class PhysicsError(TimoshenkoError):
    """Physics modeling error"""
    pass


class NumericalError(TimoshenkoError):
    """Numerical computation error"""
    pass


class NetworkError(TimoshenkoError):
    """Neural-network-related error"""
    pass


class TrainingError(TimoshenkoError):
    """Training process error"""
    pass


class DataError(TimoshenkoError):
    """Data-related error"""
    pass


# ============================================================================
# Concrete exception classes
# ============================================================================

class InvalidParameterError(ConfigurationError):
    """Invalid parameter error"""
    
    def __init__(self, param_name: str, param_value: Any, 
                 valid_range: Optional[str] = None):
        suggestions = [
            f"Check the valid range of parameter {param_name}",
            "Refer to the parameter description in the documentation",
            "Test with default parameter values"
        ]
        
        if valid_range:
            suggestions.insert(0, f"Ensure {param_name} is within the valid range: {valid_range}")
        
        context = {
            "parameter": param_name,
            "value": param_value,
            "valid_range": valid_range
        }
        
        message = f"Parameter {param_name} has invalid value {param_value}"
        super().__init__(message, "INVALID_PARAM", suggestions, context)


class PhysicsViolationError(PhysicsError):
    """Physics constraint violation error"""
    
    def __init__(self, violation_type: str, details: Optional[str] = None):
        suggestions = [
            "Check whether material parameters are reasonable",
            "Verify boundary condition settings",
            "Confirm load magnitude and direction",
            "Check the physical meaning of geometric parameters"
        ]
        
        context = {
            "violation_type": violation_type,
            "details": details
        }
        
        message = f"Physics constraint violation: {violation_type}"
        if details:
            message += f" - {details}"
            
        super().__init__(message, "PHYSICS_VIOLATION", suggestions, context)


class NumericalInstabilityError(NumericalError):
    """Numerical instability error"""
    
    def __init__(self, instability_type: str, location: Optional[str] = None):
        suggestions = [
            "Reduce the learning rate or time step",
            "Add numerical stability checks",
            "Use a more stable numerical method",
            "Check the boundary condition implementation"
        ]
        
        if "NaN" in instability_type:
            suggestions.insert(0, "Check for division-by-zero operations and numerical overflow")
        elif "divergence" in instability_type.lower():
            suggestions.insert(0, "Reduce training parameters or increase regularization")
        
        context = {
            "type": instability_type,
            "location": location
        }
        
        message = f"Numerical instability: {instability_type}"
        if location:
            message += f" (location: {location})"
            
        super().__init__(message, "NUMERICAL_INSTABILITY", suggestions, context)


class ConvergenceError(TrainingError):
    """Convergence failure error"""
    
    def __init__(self, reason: str, epoch: Optional[int] = None, 
                 loss_value: Optional[float] = None):
        suggestions = [
            "Adjust the learning rate and optimizer parameters",
            "Check whether the network architecture is reasonable",
            "Increase training data or improve data quality",
            "Use a different initialization strategy",
            "Add a regularization term"
        ]
        
        context = {
            "reason": reason,
            "epoch": epoch,
            "loss_value": loss_value
        }
        
        message = f"Training convergence failed: {reason}"
        if epoch is not None:
            message += f" (epoch: {epoch})"
        if loss_value is not None:
            message += f" (loss: {loss_value:.2e})"
            
        super().__init__(message, "CONVERGENCE_FAILED", suggestions, context)


class GPUMemoryError(NetworkError):
    """Out-of-GPU-memory error"""
    
    def __init__(self, operation: str, required_memory: Optional[float] = None):
        suggestions = [
            "Reduce the batch size",
            "Use gradient accumulation",
            "Enable mixed precision training (AMP)",
            "Clear the GPU cache",
            "Use the CPU for computation"
        ]
        
        context = {
            "operation": operation,
            "required_memory": required_memory
        }
        
        message = f"Out of GPU memory: {operation}"
        if required_memory:
            message += f" (required: {required_memory:.2f}GB)"
            
        super().__init__(message, "GPU_MEMORY_ERROR", suggestions, context)


# ============================================================================
# Exception handling utilities
# ============================================================================

class ExceptionHandler:
    """Exception handler"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def handle_exception(self, exc: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Unified exception handling"""

        # Collect exception information
        exc_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        # If it is a project-specific exception, add extra information
        if isinstance(exc, TimoshenkoError):
            exc_info.update({
                "error_code": exc.error_code,
                "suggestions": exc.suggestions,
                "exception_context": exc.context
            })

        # Record the log
        self.logger.error(f"Exception occurred: {exc_info['type']} - {exc_info['message']}")
        
        if isinstance(exc, TimoshenkoError) and exc.suggestions:
            self.logger.info("Suggested solutions:")
            for suggestion in exc.suggestions:
                self.logger.info(f"  - {suggestion}")
        
        return exc_info
    
    def try_recovery(self, exc: Exception) -> Optional[str]:
        """Attempt automatic recovery"""
        
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return "GPU cache cleared, please try reducing the batch size"
        
        elif isinstance(exc, NumericalInstabilityError):
            return "Suggest lowering the learning rate and checking numerical stability"
        
        elif "NaN" in str(exc) or "Inf" in str(exc):
            return "Numerical anomaly detected, suggest checking input data and computation process"
        
        return None


def safe_execute(func, *args, exception_handler: Optional[ExceptionHandler] = None, **kwargs):
    """Safely execute a function, with exception handling"""
    
    handler = exception_handler or ExceptionHandler()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        exc_info = handler.handle_exception(e)
        recovery_msg = handler.try_recovery(e)
        
        if recovery_msg:
            print(f"Auto recovery attempt: {recovery_msg}")

        # Re-raise the exception, but the detailed information has now been recorded
        raise e


# ============================================================================
# Decorators
# ============================================================================

def error_handler(logger: Optional[logging.Logger] = None):
    """Error handling decorator"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = ExceptionHandler(logger)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exc_info = handler.handle_exception(e, {"function": func.__name__})
                recovery_msg = handler.try_recovery(e)
                
                if recovery_msg:
                    print(f"Function {func.__name__} exception, attempting recovery: {recovery_msg}")
                
                raise e
        return wrapper
    return decorator


def validate_parameters(**validators):
    """Parameter validation decorator"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get the function parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise InvalidParameterError(
                            param_name, value, 
                            f"Validation function: {validator.__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Validation functions
# ============================================================================

def is_positive(x):
    """Validate that the value is positive"""
    return x > 0

def is_non_negative(x):
    """Validate that the value is non-negative"""
    return x >= 0

def is_in_range(min_val, max_val):
    """Validate that the value is within the specified range"""
    def validator(x):
        return min_val <= x <= max_val
    return validator

def is_valid_tensor(x):
    """Validate tensor validity"""
    if not isinstance(x, torch.Tensor):
        return False
    return not (torch.isnan(x).any() or torch.isinf(x).any())


# ============================================================================
# Usage examples and tests
# ============================================================================

if __name__ == "__main__":
    print("Exception handling system test...")

    # Create the exception handler
    handler = ExceptionHandler()

    # Test the custom exception
    try:
        raise InvalidParameterError("learning_rate", -0.01, "must be positive")
    except TimoshenkoError as e:
        print(f"Caught custom exception:\n{e}")
        handler.handle_exception(e)

    # Test the decorator
    @validate_parameters(x=is_positive, y=is_in_range(0, 1))
    def test_function(x, y):
        return x * y

    try:
        test_function(-1, 0.5)  # Should raise an exception
    except InvalidParameterError as e:
        print(f"\nParameter validation exception:\n{e}")
    
    print("\nException handling system created!")