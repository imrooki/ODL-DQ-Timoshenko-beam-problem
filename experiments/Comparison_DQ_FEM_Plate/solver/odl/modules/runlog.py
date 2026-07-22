import contextlib
import datetime
import os
import platform
import sys
import time


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _sync():
    try:
        import torch
        from .odl_config import DEFAULT_DEVICE
        if str(DEFAULT_DEVICE).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def device_string():
    import torch
    from .odl_config import DEFAULT_DEVICE
    if torch.cuda.is_available():
        hw = f"GPU available: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    else:
        hw = f"no GPU; CPU {platform.processor() or platform.machine()}"
    return f"solve-device={DEFAULT_DEVICE} | {hw}"


@contextlib.contextmanager
def run_logger(behavior, log_dir=None):
    import torch
    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{behavior}_{stamp}.log")
    fh = open(path, "w", encoding="utf-8")
    old = sys.stdout
    t0 = time.perf_counter()
    try:
        sys.stdout = _Tee(old, fh)
        print(f"[run] {behavior} @ {stamp}  |  torch {torch.__version__}  |  {device_string()}")
        yield
    finally:
        _sync()
        try:
            print(f"[run] total wall-time: {time.perf_counter() - t0:.2f} s")
        except Exception:
            pass
        sys.stdout = old
        fh.close()
        print(f"[log] saved {path}")


@contextlib.contextmanager
def timed(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        print(f"[time] {name}: {time.perf_counter() - t0:.2f} s")
