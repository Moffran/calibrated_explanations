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
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("thresholds", type=Path)
    args = p.parse_args(argv)

    baseline_raw = load_json(args.baseline)
    current_raw = load_json(args.current)
    thresholds = load_json(args.thresholds)

    status = 0
    messages: list[str] = []

    def get_nested(data: Mapping[str, Any], dotted: str) -> Any:
        node: Any = data
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return None
        return node

    for metric, config in thresholds.items():
        thr = config.get("max_relative_increase")
        if thr is None:
            continue
        b_val = get_nested(baseline_raw, metric)
        c_val = get_nested(current_raw, metric)
        if not isinstance(b_val, (int, float)) or not isinstance(c_val, (int, float)):
            status = 2
            messages.append(f"{metric} missing or invalid; failing check")
            continue
        if b_val <= 0:
            status = 2
            messages.append(f"{metric} baseline non-positive ({b_val}); failing check")
            continue
        rel = (c_val - b_val) / b_val
        if rel > thr:
            status = 2
            messages.append(
                f"{metric} regression: baseline={b_val:.6f}s current={c_val:.6f}s rel={rel:.3f} > {thr:.3f}"
            )
        else:
            messages.append(
                f"{metric} OK: baseline={b_val:.6f}s current={c_val:.6f}s rel={rel:.3f} <= {thr:.3f}"
            )

    # Optional logging for map times
    if isinstance(current_raw.get("map"), dict):
        m = current_raw["map"]
        seq = m.get("sequential_s")
        job = m.get("joblib_s")
        if isinstance(seq, (int, float)) and isinstance(job, (int, float)):
            messages.append(f"map timings: sequential={seq:.6f}s joblib={job:.6f}s")

    for line in messages:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
