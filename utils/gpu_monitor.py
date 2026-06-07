"""
GPU monitoring module
Used to monitor performance metrics such as GPU utilization and GPU memory usage
"""

import torch
import time
from typing import Dict, Optional
import warnings

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not installed, will use PyTorch built-in GPU monitoring")


class GPUMonitor:
    """
    GPU performance monitor
    Supports monitoring metrics such as GPU utilization, GPU memory usage, and temperature
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize the GPU monitor

        Parameters:
            device_id: GPU device ID
        """
        self.device_id = device_id
        self.use_pynvml = False
        self.handle = None

        # Try to initialize pynvml
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.use_pynvml = True
                print(f"[GPU monitor] Using pynvml to monitor GPU {device_id}")
            except Exception as e:
                print(f"[GPU monitor] pynvml initialization failed: {e}")
                self.use_pynvml = False

        # Statistics data storage
        self.stats_history = []
        self.start_time = time.time()

    def get_stats(self) -> Dict:
        """
        Get the current GPU status

        Returns:
            a dictionary containing GPU status information
        """
        stats = {}
        
        if torch.cuda.is_available():
            # Basic PyTorch GPU information
            stats['cuda_available'] = True
            stats['device_name'] = torch.cuda.get_device_name(self.device_id)
            stats['device_count'] = torch.cuda.device_count()

            # GPU memory information (PyTorch built-in)
            mem_allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3    # GB
            mem_total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3

            stats['mem_allocated'] = mem_allocated
            stats['mem_reserved'] = mem_reserved
            stats['mem_total'] = mem_total
            stats['mem_used'] = mem_reserved  # Use reserved as the actual usage
            stats['mem_free'] = mem_total - mem_reserved
            stats['mem_percent'] = (mem_reserved / mem_total * 100) if mem_total > 0 else 0

            # If pynvml is available, get more detailed information
            if self.use_pynvml and self.handle:
                try:
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    stats['gpu_util'] = util.gpu
                    stats['mem_util'] = util.memory

                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                    stats['temperature'] = temp

                    # Power consumption (if supported)
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert to watts
                        stats['power'] = power
                    except:
                        stats['power'] = None

                except Exception as e:
                    # pynvml retrieval failed, use default values and record the error type
                    if verbose := False:  # Detailed logging can be enabled as needed
                        print(f"[GPU monitor] pynvml data retrieval failed: {type(e).__name__}: {str(e)}")
                    stats['gpu_util'] = -1
                    stats['mem_util'] = stats['mem_percent']
                    stats['temperature'] = -1
                    stats['power'] = None
                    stats['error'] = f"{type(e).__name__}"
            else:
                # No pynvml, use estimated values
                stats['gpu_util'] = -1  # Cannot retrieve
                stats['mem_util'] = stats['mem_percent']
                stats['temperature'] = -1
                stats['power'] = None

        else:
            stats['cuda_available'] = False
            stats['device_name'] = 'CPU'
            stats['gpu_util'] = 0
            stats['mem_used'] = 0
            stats['mem_total'] = 0

        # Add timestamp
        stats['timestamp'] = time.time() - self.start_time
        
        return stats
    
    def record_stats(self) -> Dict:
        """
        Record the current GPU status to the history

        Returns:
            the current status
        """
        stats = self.get_stats()
        self.stats_history.append(stats)
        return stats

    def get_summary(self) -> Dict:
        """
        Get the statistics summary

        Returns:
            a dictionary containing statistics
        """
        if not self.stats_history:
            return {}

        # Compute statistics
        gpu_utils = [s.get('gpu_util', 0) for s in self.stats_history if s.get('gpu_util', -1) >= 0]
        mem_useds = [s.get('mem_used', 0) for s in self.stats_history]
        temperatures = [s.get('temperature', 0) for s in self.stats_history if s.get('temperature', -1) >= 0]
        
        summary = {
            'num_records': len(self.stats_history),
            'total_time': time.time() - self.start_time,
        }
        
        if gpu_utils:
            summary['avg_gpu_util'] = sum(gpu_utils) / len(gpu_utils)
            summary['max_gpu_util'] = max(gpu_utils)
            summary['min_gpu_util'] = min(gpu_utils)
        
        if mem_useds:
            summary['avg_mem_used'] = sum(mem_useds) / len(mem_useds)
            summary['max_mem_used'] = max(mem_useds)
            summary['min_mem_used'] = min(mem_useds)
        
        if temperatures:
            summary['avg_temperature'] = sum(temperatures) / len(temperatures)
            summary['max_temperature'] = max(temperatures)
        
        return summary
    
    def print_current_stats(self, prefix: str = ""):
        """
        Print the current GPU status

        Parameters:
            prefix: output prefix
        """
        stats = self.get_stats()

        if stats['cuda_available']:
            output = f"{prefix}GPU: "

            # GPU utilization
            if stats['gpu_util'] >= 0:
                output += f"{stats['gpu_util']:.0f}%, "

            # GPU memory
            output += f"GPU memory: {stats['mem_used']:.1f}/{stats['mem_total']:.1f}GB"

            # Temperature (if available)
            if stats.get('temperature', -1) > 0:
                output += f", Temperature: {stats['temperature']}°C"

            # Power consumption (if available)
            if stats.get('power'):
                output += f", Power: {stats['power']:.0f}W"

            print(output)
        else:
            print(f"{prefix}Using CPU for computation")
    
    def print_summary(self):
        """
        Print the statistics summary
        """
        summary = self.get_summary()

        if not summary:
            print("[GPU monitor] No statistics data")
            return

        print("\n" + "="*50)
        print("GPU usage statistics")
        print("="*50)
        print(f"Total run time: {summary['total_time']:.1f} s")
        print(f"Number of records: {summary['num_records']}")

        if 'avg_gpu_util' in summary:
            print(f"GPU utilization: average {summary['avg_gpu_util']:.1f}%, "
                  f"maximum {summary['max_gpu_util']:.1f}%, "
                  f"minimum {summary['min_gpu_util']:.1f}%")

        if 'avg_mem_used' in summary:
            print(f"GPU memory usage: average {summary['avg_mem_used']:.2f}GB, "
                  f"maximum {summary['max_mem_used']:.2f}GB")

        if 'avg_temperature' in summary:
            print(f"GPU temperature: average {summary['avg_temperature']:.1f}°C, "
                  f"maximum {summary['max_temperature']:.1f}°C")

        print("="*50)

    def reset(self):
        """
        Reset the statistics data
        """
        self.stats_history = []
        self.start_time = time.time()

    def __del__(self):
        """
        Clean up resources
        """
        if self.use_pynvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# Convenience function
def create_gpu_monitor(device_id: int = 0) -> Optional[GPUMonitor]:
    """
    Create a GPU monitor

    Parameters:
        device_id: GPU device ID

    Returns:
        a GPUMonitor instance, or None if the GPU is not available
    """
    if torch.cuda.is_available():
        return GPUMonitor(device_id)
    else:
        print("[GPU monitor] CUDA not available, GPU monitoring disabled")
        return None


# Test code
if __name__ == "__main__":
    print("Testing the GPU monitoring module...")

    monitor = create_gpu_monitor()
    if monitor:
        # Show the current status
        monitor.print_current_stats()

        # Simulate some GPU operations
        print("\nRunning GPU computation test...")
        device = torch.device("cuda")
        for i in range(5):
            # Create large matrices for computation
            a = torch.randn(5000, 5000, device=device)
            b = torch.randn(5000, 5000, device=device)
            c = torch.matmul(a, b)

            # Record the status
            monitor.record_stats()
            time.sleep(1)

        # Show statistics
        monitor.print_summary()
    else:
        print("GPU not available, skipping test")