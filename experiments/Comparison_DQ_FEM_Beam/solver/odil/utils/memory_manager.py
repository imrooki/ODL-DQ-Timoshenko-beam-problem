

import torch
from typing import Dict, Tuple, Optional
import gc
try:
    import psutil
except ImportError:
    psutil = None


class TensorPool:
    

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float64):
        
        self.device = device
        self.dtype = dtype
        self.pool: Dict[Tuple, torch.Tensor] = {}
        self.in_use: set = set()
        
    def get(self, shape: Tuple, dtype: Optional[torch.dtype] = None, 
            zero_init: bool = True) -> torch.Tensor:
        
        if dtype is None:
            dtype = self.dtype

        key = (shape, dtype)

        
        if key in self.pool and key not in self.in_use:
            tensor = self.pool[key]
            self.in_use.add(key)
            if zero_init:
                tensor.zero_()
            return tensor

        
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        self.pool[key] = tensor
        self.in_use.add(key)
        return tensor
    
    def release(self, tensor: torch.Tensor):
        
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        key = (shape, dtype)
        
        if key in self.in_use:
            self.in_use.remove(key)
    
    def clear(self):
        
        self.pool.clear()
        self.in_use.clear()

        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_stats(self) -> Dict:
        
        total_tensors = len(self.pool)
        in_use_count = len(self.in_use)
        available_count = total_tensors - in_use_count

        
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
    

    @staticmethod
    def clear_cache():
        
        
        gc.collect()

        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
