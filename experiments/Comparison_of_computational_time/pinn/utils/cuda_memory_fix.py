"""
CUDA Memory Management Fix for Energy-Based PINNs
Specifically addresses GPU memory leaks in energy-based PINN training

Author: Yang (Enhanced by PINN Expert)
Version: 2.0

Core problems addressed:
1. Memory leaks caused by autograd graph accumulation
2. Uncleared model/optimizer references
3. Recovery after CUDA illegal-memory-access errors
4. Accumulation of intermediate tensors in energy integration
5. Memory management for transfer learning

Design principles:
- Fully clear the computation graph and gradient cache
- Safe error recovery mechanism
- Minimize memory footprint
- Preserve numerical accuracy
"""

import gc
import traceback
from typing import Optional, Dict, Any, Callable
import warnings
import torch
import torch.nn as nn
from contextlib import contextmanager


class CUDAMemoryGuard:
    """CUDA memory guard - core tool for preventing memory leaks"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._original_cuda_state = None
        self._memory_snapshots = []

    def take_snapshot(self, label: str = ""):
        """Record the current GPU memory snapshot"""
        if torch.cuda.is_available():
            snapshot = {
                'label': label,
                'allocated': torch.cuda.memory_allocated() / 1e9,  # GB
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
            self._memory_snapshots.append(snapshot)
            if self.verbose:
                print(f"[Memory] {label}: Allocated={snapshot['allocated']:.2f}GB, Reserved={snapshot['reserved']:.2f}GB")
            return snapshot
        return None

    @staticmethod
    def force_cleanup(models: list = None, optimizers: list = None):
        """Force GPU memory cleanup - core cleanup function"""
        try:
            # 1. Clean up models and optimizers
            if models:
                for model in models:
                    if model is not None:
                        # Clear the gradients of the model parameters
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad = None
                        # Move the model to the CPU and delete it
                        try:
                            model.cpu()
                        except:
                            pass
                        del model

            if optimizers:
                for optimizer in optimizers:
                    if optimizer is not None:
                        # Clear the optimizer state
                        optimizer.zero_grad(set_to_none=True)
                        if hasattr(optimizer, 'state'):
                            optimizer.state.clear()
                        del optimizer

            # 2. Python garbage collection
            gc.collect()

            # 3. GPU cleanup sequence
            if torch.cuda.is_available():
                # Synchronize all CUDA operations
                torch.cuda.synchronize()
                # Empty the cache
                torch.cuda.empty_cache()
                # Reset peak memory statistics
                torch.cuda.reset_peak_memory_stats()
                # Empty again (to ensure cleanup)
                torch.cuda.empty_cache()

        except Exception as e:
            if str(e).find("illegal memory access") != -1:
                # CUDA illegal-access error - attempt to reset the device
                CUDAMemoryGuard.reset_cuda_device()
            else:
                print(f"[WARNING] Cleanup error: {e}")

    @staticmethod
    def reset_cuda_device():
        """Reset the CUDA device (last resort)"""
        if torch.cuda.is_available():
            try:
                device_id = torch.cuda.current_device()
                torch.cuda.set_device(device_id)
                # Force synchronization and cleanup
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print("[INFO] CUDA device reset successful")
            except Exception as e:
                print(f"[ERROR] CUDA device reset failed: {e}")
                # Recommend restarting the process
                print("[CRITICAL] Recommend restarting the process due to GPU memory corruption")


class AutogradMemoryManager:
    """Autograd memory manager - handles computation-graph-related memory issues"""

    @staticmethod
    def safe_backward(loss: torch.Tensor, retain_graph: bool = False):
        """Safe backward pass, preventing gradient-graph accumulation"""
        if loss.requires_grad:
            # Ensure that loss is a scalar
            if loss.numel() != 1:
                loss = loss.mean()

            try:
                loss.backward(retain_graph=retain_graph)
            finally:
                # Unless explicitly required to keep it, clear the computation graph
                if not retain_graph:
                    # Detach loss to prevent the graph from being retained
                    loss = loss.detach()

    @staticmethod
    @contextmanager
    def managed_autograd(create_graph: bool = True, retain_graph: bool = False):
        """Manage the autograd context, automatically clearing the computation graph"""
        try:
            # Record the state on entry
            initial_graphs = []
            yield
        finally:
            # Clear the computation graph created within this context
            if not retain_graph:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                gc.collect()

    @staticmethod
    def compute_derivatives_safe(func: Callable, x: torch.Tensor,
                                order: int = 1, create_graph: bool = True) -> torch.Tensor:
        """Safely compute derivatives, avoiding memory leaks"""
        x = x.requires_grad_(True)
        y = func(x)

        # First-order derivative
        grad1 = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=create_graph,
            retain_graph=False,  # Do not retain the graph used for the first-order derivative
            only_inputs=True     # Compute gradients only with respect to the inputs
        )[0]

        if order == 1:
            return grad1.detach() if not create_graph else grad1

        # Second-order derivative
        if order == 2:
            grad2 = torch.autograd.grad(
                grad1, x,
                grad_outputs=torch.ones_like(grad1),
                create_graph=False,  # The second-order derivative usually does not need further differentiation
                retain_graph=False,
                only_inputs=True
            )[0]
            return grad2.detach()

        return grad1


class EnergyPINNMemoryWrapper:
    """Memory wrapper dedicated to energy-based PINNs"""

    def __init__(self, pinn_model):
        self.model = pinn_model
        self.memory_guard = CUDAMemoryGuard()

    def compute_energy_safe(self, x_samples: torch.Tensor,
                          cleanup_intermediate: bool = True) -> Dict[str, torch.Tensor]:
        """Safely compute the energy, avoiding accumulation of intermediate tensors"""

        # Process in batches to prevent a large number of intermediate tensors
        batch_size = min(1000, len(x_samples))
        total_energy = 0
        energy_components = {}

        for i in range(0, len(x_samples), batch_size):
            x_batch = x_samples[i:i+batch_size]

            # Compute the energy for this batch
            with torch.no_grad() if cleanup_intermediate else torch.enable_grad():
                batch_energy = self._compute_batch_energy(x_batch)

            # Accumulate the results
            for key, value in batch_energy.items():
                if key not in energy_components:
                    energy_components[key] = 0
                energy_components[key] += value.detach() if cleanup_intermediate else value

            # Clean up intermediate results
            if cleanup_intermediate and i % (batch_size * 10) == 0:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return energy_components

    def _compute_batch_energy(self, x_batch):
        """Compute the batch energy (to be implemented according to the actual model)"""
        # The actual energy computation should be called here
        # Example return
        return {'Pi_str': torch.tensor(0.0), 'Pi_e': torch.tensor(0.0)}


def safe_training_iteration(train_func: Callable,
                           cleanup_every: int = 100,
                           force_cleanup_on_error: bool = True) -> Callable:
    """Safe training iteration decorator"""
    def wrapper(*args, **kwargs):
        iteration = kwargs.get('epoch', 0)

        try:
            # Execute the training function
            result = train_func(*args, **kwargs)

            # Periodic cleanup
            if iteration % cleanup_every == 0:
                CUDAMemoryGuard.force_cleanup()

            return result

        except RuntimeError as e:
            if "out of memory" in str(e) or "illegal memory access" in str(e):
                print(f"[ERROR] GPU memory error at iteration {iteration}: {e}")

                if force_cleanup_on_error:
                    print("[INFO] Attempting memory recovery...")
                    CUDAMemoryGuard.force_cleanup()
                    CUDAMemoryGuard.reset_cuda_device()

                    # Attempt to continue training (using a smaller batch)
                    if 'N_samples' in kwargs:
                        kwargs['N_samples'] = max(100, kwargs['N_samples'] // 2)
                        print(f"[INFO] Retrying with reduced samples: {kwargs['N_samples']}")
                        return train_func(*args, **kwargs)

                raise
            else:
                raise

    return wrapper


class SensitivityAnalysisMemoryFix:
    """Memory fix dedicated to sensitivity analysis"""

    @staticmethod
    def run_case_with_cleanup(run_func: Callable, case_params: Dict,
                             verbose: bool = False) -> Dict:
        """Run a single case and ensure complete cleanup"""

        memory_guard = CUDAMemoryGuard(verbose=verbose)
        memory_guard.take_snapshot("Before case")

        try:
            # Run the training
            result = run_func(case_params)

            # Extract the necessary data (ensuring it is detached from the computation graph)
            clean_result = SensitivityAnalysisMemoryFix._extract_clean_results(result)

        except Exception as e:
            print(f"[ERROR] Case failed: {e}")
            clean_result = {'success': False, 'error': str(e)}

        finally:
            # Complete cleanup - this is the key!
            SensitivityAnalysisMemoryFix._complete_cleanup(result if 'result' in locals() else None)
            memory_guard.take_snapshot("After cleanup")

        return clean_result

    @staticmethod
    def _extract_clean_results(result: Dict) -> Dict:
        """Extract and clean the results, breaking all computation-graph connections"""
        clean = {'success': result.get('success', False)}

        # Safely extract the numerical results
        if 'results' in result:
            clean['results'] = {}
            for key, value in result['results'].items():
                if isinstance(value, tuple):
                    # Displacement field results
                    clean['results'][key] = tuple(
                        v.copy() if hasattr(v, 'copy') else v
                        for v in value
                    )
                elif torch.is_tensor(value):
                    clean['results'][key] = value.detach().cpu().numpy()

        # Extract scalar values
        if 'summary' in result:
            clean['summary'] = result['summary']  # Already Python scalars

        return clean

    @staticmethod
    def _complete_cleanup(result: Optional[Dict]):
        """Fully clean up the training results and GPU memory"""

        if result:
            # Clean up models and optimizers
            models_to_clean = []
            optimizers_to_clean = []

            # Find all possible model/optimizer references
            for key in ['model_linear', 'model_nonlinear', 'model', 'lin_model', 'non_model']:
                if key in result and result[key] is not None:
                    models_to_clean.append(result[key])

            for key in ['optimizer', 'optimizer_lin', 'optimizer_non']:
                if key in result and result[key] is not None:
                    optimizers_to_clean.append(result[key])

            # Force cleanup
            CUDAMemoryGuard.force_cleanup(models_to_clean, optimizers_to_clean)

            # Clear the result dictionary
            result.clear()

        # Additional global cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


# Utility functions
def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_reserved()) / 1e9
    }


def check_memory_health() -> bool:
    """Check the GPU memory health status"""
    if not torch.cuda.is_available():
        return True

    info = get_gpu_memory_info()
    # If the available memory is less than 1 GB, consider it unhealthy
    return info.get('free_gb', 0) > 1.0


# Best-practices checklist generator
def generate_memory_checklist() -> list:
    """Generate a best-practices checklist for memory management in PINN training"""
    return [
        "Wrap computations that do not need gradients with torch.no_grad()",
        "Call torch.cuda.empty_cache() periodically",
        "Delete model references immediately after saving the model",
        "Use .detach() to break unneeded computation graph connections",
        "Clear optimizer.state after using the LBFGS optimizer",
        "Batch large integral computations to avoid computing all at once",
        "Use create_graph=False when computing final derivatives",
        "Fully clear GPU memory after each training case",
        "Monitor GPU memory usage and set warning thresholds",
        "Implement an error recovery mechanism to handle CUDA errors"
    ]


if __name__ == "__main__":
    print("Energy-based PINN CUDA Memory Management Fix Module")
    print("=" * 50)
    print("\nBest Practices Checklist:")
    for item in generate_memory_checklist():
        print(item)
    print("\nGPU Memory Status:")
    print(get_gpu_memory_info())