"""
GPU monitoring utility (gpu_monitor)

Responsibilities:
- Periodically sample GPU status in the background during training and save it as CSV (located in results/.../logs/).
- Prefer pynvml (if available), otherwise call nvidia-smi; otherwise fall back to the GPU memory information provided by PyTorch.

Usage:
    from utils.gpu_monitor import GPUMonitor
    mon = GPUMonitor(interval=2.0, output_csv_path=".../logs/gpu_xxx.csv", log_to_console=False)
    mon.start()
    ... training ...
    mon.stop()

Notes:
- If the environment has no NVIDIA GPU, sampling is silently skipped (safe no-op).
- No extra dependencies are installed: pynvml is used if present; otherwise nvidia-smi or torch.cuda.* information is used on a best-effort basis.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from typing import Any, Dict, List, Optional


class GPUMonitor:
    """Simple background thread for GPU monitoring."""

    def __init__(
        self,
        interval: float = 2.0,
        device_ids: Optional[List[int]] = None,
        output_csv_path: Optional[str] = None,
        log_to_console: bool = False,
    ) -> None:
        self.interval = float(interval)
        self.device_ids = device_ids
        self.output_csv_path = output_csv_path
        self.log_to_console = bool(log_to_console)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._csv_file = None
        self._csv_writer: Optional[csv.writer] = None

        # Runtime fallback strategy flags
        self._use_nvml = False
        self._use_nvsmi = False
        self._use_torch = False

        # Added: state flags to prevent double release
        self._stopped = False
        self._nvml_initialized = False

        # Lazy import to avoid errors when the environment has no torch or nvml
        try:
            import torch  # noqa: F401
            self._torch = __import__("torch")
        except Exception:
            self._torch = None

        # Initialize availability detection
        self._init_backends()

    def _init_backends(self) -> None:
        """Detect available backends: pynvml -> nvidia-smi -> torch."""

        # 1) NVML
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                self._use_nvml = True
                self._pynvml = pynvml
                self._nvml_device_count = count
                self._nvml_initialized = True  # mark NVML as initialized
            else:
                pynvml.nvmlShutdown()
        except Exception:
            self._use_nvml = False
            self._pynvml = None
            self._nvml_initialized = False

        # 2) nvidia-smi
        if not self._use_nvml:
            import shutil

            if shutil.which("nvidia-smi") is not None:
                self._use_nvsmi = True

        # 3) Torch fallback
        if not self._use_nvml and not self._use_nvsmi and self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    # At least the device name and GPU memory information can be read
                    self._use_torch = True
            except Exception:
                self._use_torch = False

    def start(self) -> None:
        """Start the monitoring thread (return immediately if no backend is available)."""

        if not (self._use_nvml or self._use_nvsmi or self._use_torch):
            return  # No GPU or not monitorable, silently skip

        # Prepare CSV writing
        if self.output_csv_path is not None:
            # Ensure an absolute path is used and create the directory safely
            abs_path = os.path.abspath(self.output_csv_path)
            dir_path = os.path.dirname(abs_path)

            # Only create the directory when the directory path is non-empty
            if dir_path and dir_path != '.':
                os.makedirs(dir_path, exist_ok=True)

            self._csv_file = open(abs_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            header = [
                "timestamp",
                "gpu_id",
                "name",
                "utilization(%)",
                "mem_used(MiB)",
                "mem_total(MiB)",
                "temperature(C)",
                "power(W)",
            ]
            self._csv_writer.writerow(header)
            self._csv_file.flush()

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="GPUMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread and release resources (idempotent operation)."""
        # Check whether already stopped, to prevent double release
        if self._stopped:
            return

        self._stopped = True  # mark as stopped

        # Set the stop flag
        self._stop.set()

        # Ensure the thread terminates correctly
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.interval * 3)  # increase the timeout
            if self._thread.is_alive():
                print("WARNING: GPU monitor thread did not stop within timeout")
            self._thread = None

        # Clean up NVML resources (add a state check to prevent double shutdown)
        if self._nvml_initialized and self._use_nvml and getattr(self, "_pynvml", None) is not None:
            try:
                self._pynvml.nvmlShutdown()
                self._nvml_initialized = False  # mark as shut down
            except Exception:
                pass
            finally:
                self._pynvml = None
                self._use_nvml = False

        # Ensure the file handle is closed correctly
        if self._csv_file is not None:
            try:
                if not self._csv_file.closed:
                    self._csv_file.flush()
                    self._csv_file.close()
            except Exception:
                pass
            finally:
                self._csv_file = None
                self._csv_writer = None

    # Used as a context manager (with GPUMonitor(...) as mon: ...)
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except Exception:
            pass
        # Do not swallow the exception
        return False

    def __del__(self):
        """Destructor, ensures resources are released"""
        try:
            self.stop()
        except Exception:
            pass  # A destructor should not raise exceptions

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                rows = self._sample_once()
                if rows and self._csv_writer is not None:
                    for r in rows:
                        self._csv_writer.writerow(r)
                    self._csv_file.flush()
                if rows and self.log_to_console:
                    for r in rows:
                        # Brief console output
                        print(
                            f"GPU{r[1]} {r[2]} util={r[3]}% mem={r[4]}/{r[5]}MiB temp={r[6]}C power={r[7]}W"
                        )
            except Exception:
                # Ignore sampling exceptions and keep trying
                pass
            # Control the sampling interval
            dt = time.time() - t0
            time.sleep(max(0.0, self.interval - dt))

    def _sample_once(self) -> List[List[Any]]:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if self._use_nvml:
            return self._sample_nvml(ts)
        if self._use_nvsmi:
            return self._sample_nvsmi(ts)
        if self._use_torch:
            return self._sample_torch(ts)
        return []

    def _selected_device_ids(self, count: int) -> List[int]:
        if self.device_ids is None:
            return list(range(count))
        return [i for i in self.device_ids if 0 <= i < count]

    # Backend 1: NVML
    def _sample_nvml(self, ts: str) -> List[List[Any]]:
        rows: List[List[Any]] = []
        nvml = self._pynvml
        count = self._nvml_device_count
        for i in self._selected_device_ids(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            try:
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW->W
            except Exception:
                power = 0.0
            rows.append(
                [ts, i, name, getattr(util, "gpu", 0), int(mem.used / 1024 ** 2), int(mem.total / 1024 ** 2), temp, power]
            )
        return rows

    # Backend 2: nvidia-smi
    def _sample_nvsmi(self, ts: str) -> List[List[Any]]:
        import subprocess

        query = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(query, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return []
        rows: List[List[Any]] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                gpu_id = int(parts[0])
                name = parts[1]
                util = int(float(parts[2]))
                mem_used = int(float(parts[3]))
                mem_total = int(float(parts[4]))
                temp = int(float(parts[5]))
                power = float(parts[6])
            except Exception:
                continue
            rows.append([ts, gpu_id, name, util, mem_used, mem_total, temp, power])
        if self.device_ids is not None:
            rows = [r for r in rows if r[1] in self.device_ids]
        return rows

    # Backend 3: PyTorch fallback
    def _sample_torch(self, ts: str) -> List[List[Any]]:
        rows: List[List[Any]] = []
        torch = self._torch
        try:
            count = torch.cuda.device_count()
            ids = self._selected_device_ids(count)
            for i in ids:
                props = torch.cuda.get_device_properties(i)
                name = props.name
                # GPU memory: prefer mem_get_info (returns bytes), otherwise estimate from allocated/reserved
                mem_used = int(torch.cuda.memory_allocated(i) / (1024 ** 2))
                mem_reserved = int(torch.cuda.memory_reserved(i) / (1024 ** 2))
                try:
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    mem_total = int(total_b / (1024 ** 2))
                    # If mem_used is smaller than reserved, use reserved as the approximation
                    mem_used = max(mem_used, mem_reserved)
                except Exception:
                    # Use reserved only as the used approximation; estimate the total from properties (MB)
                    mem_total = int(props.total_memory / (1024 ** 2))
                    mem_used = max(mem_used, mem_reserved)
                rows.append([ts, i, name, 0, mem_used, mem_total, 0, 0.0])
        except Exception:
            return []
        return rows


def get_gpu_status_string():
    """Get a concise string representation of the current GPU status"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "GPU not available"

        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3    # GB

        # Try to get the GPU utilization (requires pynvml)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()

            return (f"GPU {gpu_id} ({gpu_name}): "
                    f"Utilization {gpu_util}%, "
                    f"Memory {mem_allocated:.2f}/{mem_reserved:.2f}GB, "
                    f"Temp {temp}°C")
        except:
            # No pynvml, only show GPU memory information
            return (f"GPU {gpu_id} ({gpu_name}): "
                    f"Memory {mem_allocated:.2f}/{mem_reserved:.2f}GB")
    except Exception:
        return "Failed to get GPU status"


__all__ = ["GPUMonitor", "get_gpu_status_string"]
