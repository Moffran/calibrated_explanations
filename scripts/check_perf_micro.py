"""Compare micro benchmark metrics with a baseline using thresholds.

Exits with non-zero status if thresholds are exceeded.

Inputs:
- baseline JSON produced by scripts/micro_bench_perf.py
- current JSON produced by scripts/micro_bench_perf.py
- thresholds JSON (benchmarks/perf_thresholds.json)

Currently enforced:
- import_time_seconds: relative increase <= max_relative_increase

Other fields are logged for visibility but not enforced yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("thresholds", type=Path)
    args = p.parse_args(argv)

    baseline = load_json(args.baseline)
    current = load_json(args.current)
    thresholds = load_json(args.thresholds)

    # Enforce import time threshold if present
    thr = thresholds.get("import_time_seconds", {}).get("max_relative_increase")
    status = 0
    messages: list[str] = []

    if thr is not None:
        b = baseline.get("import_time_seconds")
        c = current.get("import_time_seconds")
        if isinstance(b, (int, float)) and isinstance(c, (int, float)) and b > 0:
            rel = (c - b) / b
            if rel > thr:
                status = 2
                messages.append(
                    f"import_time_seconds regression: baseline={b:.6f}s current={c:.6f}s rel={rel:.3f} > {thr:.3f}"
                )
            else:
                messages.append(
                    f"import_time_seconds OK: baseline={b:.6f}s current={c:.6f}s rel={rel:.3f} <= {thr:.3f}"
                )
        else:
            messages.append("import_time_seconds missing or invalid; skipping check")

    # Optional logging for map times
    if isinstance(current.get("map"), dict):
        m = current["map"]
        seq = m.get("sequential_s")
        job = m.get("joblib_s")
        if isinstance(seq, (int, float)) and isinstance(job, (int, float)):
            messages.append(f"map timings: sequential={seq:.6f}s joblib={job:.6f}s")

    for line in messages:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
