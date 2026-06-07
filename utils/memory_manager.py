"""
Memory management utility module

Provides a tensor memory pool and memory optimization tools to reduce memory allocation and fragmentation
"""

import torch
from typing import Dict, Tuple, Optional
import gc
try:
    import psutil
except ImportError:
    psutil = None


class TensorPool:
    """
    Tensor memory pool manager

    Reduces memory allocation overhead and fragmentation by reusing tensors
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float64):
        """
        Initialize the memory pool

        Parameters:
            device: computation device
            dtype: default data type
        """
        self.device = device
        self.dtype = dtype
        self.pool: Dict[Tuple, torch.Tensor] = {}
        self.in_use: set = set()
        
    def get(self, shape: Tuple, dtype: Optional[torch.dtype] = None, 
            zero_init: bool = True) -> torch.Tensor:
        """
        Get an existing tensor from the pool or create a new one

        Parameters:
            shape: tensor shape
            dtype: data type (uses the pool's default type by default)
            zero_init: whether to initialize to zero

        Returns:
            a reused or newly created tensor
        """
        if dtype is None:
            dtype = self.dtype

        key = (shape, dtype)

        # Check whether an available tensor exists in the pool
        if key in self.pool and key not in self.in_use:
            tensor = self.pool[key]
            self.in_use.add(key)
            if zero_init:
                tensor.zero_()
            return tensor

        # Create a new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        self.pool[key] = tensor
        self.in_use.add(key)
        return tensor
    
    def release(self, tensor: torch.Tensor):
        """
        Release a tensor back to the pool (mark it as reusable)

        Parameters:
            tensor: the tensor to release
        """
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        key = (shape, dtype)
        
        if key in self.in_use:
            self.in_use.remove(key)
    
    def clear(self):
        """
        Clear the memory pool, releasing all tensors
        """
        self.pool.clear()
        self.in_use.clear()

        # Trigger garbage collection
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_stats(self) -> Dict:
        """
        Get memory pool statistics

        Returns:
            a dictionary containing the statistics
        """
        total_tensors = len(self.pool)
        in_use_count = len(self.in_use)
        available_count = total_tensors - in_use_count

        # Compute memory usage
        total_memory = 0
        for tensor in self.pool.values():
            total_memory += tensor.element_size() * tensor.nelement()
        
        return {
            'total_tensors': total_tensors,
            'in_use': in_use_count,
            'available': available_count,
            'memory_bytes': total_memory,
            'memory_mb': total_memory / (1024 * 1024)
        }


class MemoryOptimizer:
    """
    Memory optimizer, providing various memory optimization techniques
    """

    @staticmethod
    def clear_cache():
        """
        Clear caches and unused memory
        """
        # Python garbage collection
        gc.collect()

        # CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
