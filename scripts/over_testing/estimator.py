"""Estimate coverage impact of removing tests without running full pipeline.

Usage examples:
  python scripts/over_testing/estimator.py \
    --per-test reports/over-testing/per_test_summary.csv \
    --baseline reports/over-testing/line_coverage_counts.csv \
    --remove-list candidates.txt

The script expects a per-test CSV with at least: `test_name,unique_lines,runtime`.
The baseline CSV should contain overall covered and total line counts or the script
will try to infer totals from the file if present.

This tool is intentionally conservative: estimated coverage after removals is
computed by subtracting `unique_lines` contributed by removed tests from the
baseline covered lines count.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple


def read_per_test_csv(path: str) -> Dict[str, Dict[str, float]]:
    tests = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("test") or row.get("test_name") or row.get("id")
            if not name:
                continue
            try:
                unique = float(row.get("unique_lines", row.get("unique", 0) or 0))
            except Exception:
                unique = 0.0
            try:
                runtime = float(row.get("runtime", row.get("time", 0) or 0))
            except Exception:
                runtime = 0.0
            try:
                flaky = float(row.get("flaky", 0) or 0)
            except Exception:
                flaky = 0.0
            tests[name] = {"unique_lines": unique, "runtime": runtime, "flaky": flaky}
    return tests


def read_baseline_counts(path: str) -> Tuple[int, int]:
    # Try to detect covered and total lines from a CSV or JSON. Support a few column names.
    covered = None
    total = None
    # If the file is a JSON with summary, load it
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
            if text.lstrip().startswith("{"):
                j = json.loads(text)
                if "covered_lines" in j and "total_lines" in j:
                    return int(j["covered_lines"]), int(j["total_lines"])
    except Exception:
        pass

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"Baseline file {path} is empty or not CSV/JSON")
        # If columns exist on first row
        first = rows[0]
        cov_key = None
        tot_key = None
        for c in ("covered_lines", "covered", "covered_count", "covered_lines_count"):
            if c in first:
                cov_key = c
                break
        for c in ("total_lines", "total", "total_count", "lines"):
            if c in first:
                tot_key = c
                break
        if cov_key and tot_key:
            covered = sum(int(r.get(cov_key, 0) or 0) for r in rows)
            total = sum(int(r.get(tot_key, 0) or 0) for r in rows)
        # Fallback: single aggregated row
        if covered is None or total is None:
            for r in rows:
                if "summary" in (r.get("file", "") or "").lower():
                    covered = int(r.get("covered_lines", r.get("covered", r.get("covered_count", 0)) or 0))
                    total = int(r.get("total_lines", r.get("total", r.get("total_count", 0)) or 0))
                    break
        # Heuristic: rows represent individual lines with a hit field
        if covered is None or total is None:
            total = 0
            covered = 0
            for r in rows:
                if "hit" in r or "hits" in r:
                    hit = int(r.get("hit", r.get("hits", 0)) or 0)
                    total += 1
                    if hit > 0:
                        covered += 1
            if total == 0:
                raise RuntimeError("Could not infer baseline covered/total lines from baseline CSV")
    return covered, total


def load_remove_list(path: str) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        raise RuntimeError(f"remove list {path} not found")
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


def estimate_after_removal(baseline_covered: int, baseline_total: int, per_test: Dict[str, Dict[str, float]], remove: List[str]) -> Tuple[float, int, int]:
    removed_unique = 0
    for t in remove:
        info = per_test.get(t)
        if info:
            removed_unique += int(info.get("unique_lines", 0))
        else:
            print(f"warning: test {t} not found in per-test CSV", file=sys.stderr)
    new_covered = max(0, baseline_covered - removed_unique)
    pct = float(new_covered) / float(baseline_total) if baseline_total else 0.0
    return pct, new_covered, baseline_total


def recommend_removals(per_test: Dict[str, Dict[str, float]], budget: int = 500) -> List[Tuple[str, float]]:
    # Recommend tests sorted by value_score = unique_lines / runtime (low is better for removal)
    rows = []
    for name, info in per_test.items():
        unique = float(info.get("unique_lines", 0))
        runtime = float(info.get("runtime", 0.0001))
        score = unique / runtime if runtime > 0 else float("inf")
        rows.append((name, score, unique, runtime))
    rows.sort(key=lambda r: (r[1], r[2]))
    return [(r[0], r[1]) for r in rows[:budget]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--per-test", help="CSV with per-test unique_lines and runtime", required=True)
    p.add_argument("--baseline", help="Baseline coverage CSV or JSON (line_coverage_counts.csv)", required=True)
    p.add_argument("--remove-list", help="File with tests to remove (one per line)")
    p.add_argument("--recommend", action="store_true", help="Output recommended low-value tests for removal")
    p.add_argument("--budget", type=int, default=500, help="Max recommendations to output")
    args = p.parse_args(argv)

    per_test = read_per_test_csv(args.per_test)
    baseline_covered, baseline_total = read_baseline_counts(args.baseline)

    print(f"baseline covered={baseline_covered} total={baseline_total} pct={baseline_covered/baseline_total:.4f}")

    remove = load_remove_list(args.remove_list) if args.remove_list else []
    if remove:
        pct, new_cov, tot = estimate_after_removal(baseline_covered, baseline_total, per_test, remove)
        print(f"after removing {len(remove)} tests -> covered={new_cov}/{tot} pct={pct:.4f}")
        if pct < 0.90:
            print("WARNING: estimated coverage below 90% — do NOT apply these removals unless you add tests to restore coverage.", file=sys.stderr)
    if args.recommend:
        recs = recommend_removals(per_test, budget=args.budget)
        print("#recommended_removals,score")
        for name, score in recs:
            print(f"{name},{score:.6f}")


if __name__ == "__main__":
    main()
