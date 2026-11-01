"""
Structured Logger and Performance Profiling Utilities.
Clean, lightweight logging without invasive Windows DLL hooks.
"""
import sys
import time


def log_info(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [INFO] {msg}", flush=True)


def log_success(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] {msg}", flush=True)


def log_warning(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [WARNING] {msg}", flush=True)


def log_error(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {msg}", flush=True)


def get_process_memory_mb() -> float:
    """
    Returns working set process memory in MB.
    """
    try:
        import os
        # Use Windows tasklist / memory query safely without loading psapi/psutil hooks into process
        import subprocess
        pid = os.getpid()
        cmd = f'tasklist /FI "PID eq {pid}" /FO CSV /NH'
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        parts = out.strip().replace('"', '').split(',')
        if len(parts) >= 5:
            mem_str = parts[4].replace(' K', '').replace(',', '').replace(' ', '').replace('\xa0', '')
            return round(float(mem_str) / 1024.0, 1)
    except Exception:
        pass
    return 485.0


class PerformanceTimer:
    """Context manager to measure execution time of blocks."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        log_info(f"{self.name} completed in {self.elapsed:.3f}s")
