"""Benchmarks for the performance requirements, with the hardware they ran on."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numba
import numpy as np

from overfitlab.factory import run_strategies, strategy_grid
from overfitlab.synth import gbm_prices
from overfitlab_study.reports import write_table

NFR_03_TARGET_S = 5.0
NFR_03_JIT_TARGET_S = 30.0
JIT_PROBE = (
    "from overfitlab.factory import run_strategies, strategy_grid;"
    "from overfitlab.synth import gbm_prices;"
    "run_strategies(gbm_prices(300, rng=0), strategy_grid())"
)


def total_ram_gb() -> float:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("rest", ctypes.c_ulonglong * 6),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return float(status.total) / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def hardware() -> dict[str, Any]:
    return {
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "ram_gb": round(total_ram_gb(), 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numba": numba.__version__,
    }


def median_time(action: Callable[[], object], repeats: int = 5) -> float:
    action()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def jit_compile_seconds() -> float:
    """Time a fresh process with an empty Numba cache, so every kernel compiles."""
    with tempfile.TemporaryDirectory() as cache:
        env = {**os.environ, "NUMBA_CACHE_DIR": cache}
        started = time.perf_counter()
        subprocess.run([sys.executable, "-c", JIT_PROBE], env=env, check=True)
        return time.perf_counter() - started


def row(requirement: str, case: str, seconds: float, target: float) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "case": case,
        "seconds": seconds,
        "target_s": target,
        "met": seconds <= target,
        **hardware(),
    }


def factory_rows() -> list[dict[str, Any]]:
    close = gbm_prices(5000, rng=0)
    specs = strategy_grid()
    case = "run_strategies, 1,034 variants, 5,000 bars"
    return [
        row("NFR-03", case, median_time(lambda: run_strategies(close, specs)), NFR_03_TARGET_S),
        row("NFR-03", "first-call JIT compile", jit_compile_seconds(), NFR_03_JIT_TARGET_S),
    ]


def run(reports_dir: Path) -> list[dict[str, Any]]:
    rows = factory_rows()
    write_table(rows, reports_dir / "tables" / "benchmarks.csv")
    return rows
