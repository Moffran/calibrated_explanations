# pylint: disable=line-too-long, missing-function-docstring, too-many-branches, too-many-statements, too-many-locals
"""Check performance regressions against stored baseline and thresholds.

Usage:
  python scripts/check_perf_regression.py \
      --baseline tests/benchmarks/baseline_20250816.json \
      --thresholds tests/benchmarks/perf_thresholds.json \
      --current tests/benchmarks/new_baseline.json

If --current is omitted, the script will invoke collect_baseline internally
into a temp file.
Exit code 0 = OK, 1 = regression detected, 2 = configuration error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: Dict[str, Any], dotted: str):
    parts = dotted.split(".")
    cur: Any = d
    for p in parts:
        if p not in cur:
            return None
        cur = cur[p]
    return cur


def main():
    parser = argparse.ArgumentParser(description="Check performance regressions")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--thresholds", required=True, help="Threshold JSON path")
    parser.add_argument("--current", help="Current metrics JSON path (optional)")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Force collect current metrics even if --current provided",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    thresholds_path = Path(args.thresholds)

    if not baseline_path.exists() or not thresholds_path.exists():
        print("Missing required file(s)", file=sys.stderr)
        sys.exit(2)

    if args.current and not args.collect:
        current_path = Path(args.current)
        if not current_path.exists():
            print(f"Current metrics file not found: {current_path}", file=sys.stderr)
            sys.exit(2)
    else:
        # Collect a fresh baseline
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp_path = Path(tmp.name)
        cmd = [sys.executable, "scripts/collect_baseline.py", "--output", str(tmp_path)]
        subprocess.check_call(cmd)
        current_path = tmp_path

    baseline = load_json(baseline_path)
    current = load_json(current_path)
    thresholds = load_json(thresholds_path)

    # Flatten runtime metrics for simpler access
    if "runtime" in baseline:
        for task, metrics in baseline["runtime"].items():
            for k, v in metrics.items():
                baseline_key = f"{task}.{k}" if k != task else k
                baseline[baseline_key] = v
    if "runtime" in current:
        for task, metrics in current["runtime"].items():
            for k, v in metrics.items():
                current_key = f"{task}.{k}" if k != task else k
                current[current_key] = v

    regressions = []
    summary = []
    for key, cfg in thresholds.items():
        base_val = get_nested(baseline, key) or baseline.get(key)
        cur_val = get_nested(current, key) or current.get(key)
        if base_val is None or cur_val is None:
            summary.append(f"SKIP {key}: missing")
            continue
        rel_increase = (cur_val - base_val) / base_val if base_val else 0.0
        max_rel = cfg.get("max_relative_increase")
        status = "OK"
        if max_rel is not None and rel_increase > max_rel:
            status = "REGRESSION"
            regressions.append((key, base_val, cur_val, rel_increase, max_rel))
        summary.append(
            f"{status} {key}: base={base_val:.6g} current={cur_val:.6g} rel={rel_increase:.2%} (limit {max_rel:.0%})"
        )

    print("Performance Check Summary:")
    for line in summary:
        print("  " + line)

    if regressions:
        print("\nRegressions detected:")
        for key, base_val, cur_val, rel, limit in regressions:
            print(f"  {key}: {base_val:.6g} -> {cur_val:.6g} (+{rel:.2%} > {limit:.0%})")
        sys.exit(1)

    # API removal check (optional: simple diff on symbol count)
    base_api = set(baseline.get("public_api_symbols", []))
    cur_api = set(current.get("public_api_symbols", []))
    removed = base_api - cur_api
    if removed:
        print(f"\nAPI REMOVALS DETECTED: {sorted(removed)}")
        sys.exit(1)

    print("\nAll performance metrics within thresholds and no API removals.")


if __name__ == "__main__":
    main()
