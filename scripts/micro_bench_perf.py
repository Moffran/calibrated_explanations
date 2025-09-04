"""Micro benchmarks for perf scaffolding (manual smoke test).

Measures:
- Import time for calibrated_explanations
- sequential_map vs JoblibBackend.map on a trivial CPU-bound function

Usage (optional): run as a script to print JSON metrics. CI integration can
parse this later and compare to thresholds in benchmarks/perf_thresholds.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def measure_import_time() -> float:
    t0 = time.perf_counter()
    import calibrated_explanations  # noqa: F401

    return time.perf_counter() - t0


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def measure_map_throughput() -> dict[str, float]:
    from calibrated_explanations.perf import sequential_map, JoblibBackend

    items = [30] * 2000  # small, deterministic CPU work

    t0 = time.perf_counter()
    sequential_map(fib, items)
    seq = time.perf_counter() - t0

    jb = JoblibBackend()
    t0 = time.perf_counter()
    jb.map(fib, items, workers=None)
    par = time.perf_counter() - t0

    return {"sequential_s": seq, "joblib_s": par}


def main() -> None:
    metrics = {
        "import_time_seconds": measure_import_time(),
        "map": measure_map_throughput(),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
