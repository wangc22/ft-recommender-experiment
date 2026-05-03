import os
import torch


def get_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    if torch.backends.mps.is_available():
        curr = torch.mps.current_allocated_memory() / 1024 ** 2
        drv = torch.mps.driver_allocated_memory() / 1024 ** 2
        return max(curr, drv)
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        return -1.0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
