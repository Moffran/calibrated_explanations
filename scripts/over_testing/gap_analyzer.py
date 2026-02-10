"""Analyze coverage gap blocks from line-level coverage CSVs.

Detect contiguous untested blocks >= UNTESTED_BLOCK_THRESHOLD and print a simple
report suitable for feeding into minimal test generation.

Usage:
  python scripts/over_testing/gap_analyzer.py --line-csv reports/over-testing/line_coverage_counts.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_line_csv(path: str) -> Dict[str, Dict[int, int]]:
    # Returns mapping: filename -> {line_number: hit_count}
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            file = r.get("file") or r.get("filename") or r.get("path")
            if not file:
                continue
            try:
                line = int(r.get("line", r.get("lineno", r.get("line_no", 0)) or 0))
            except Exception:
                continue
            try:
                hit = int(r.get("hit", r.get("hits", r.get("count", 0)) or 0))
            except Exception:
                hit = 0
            out[file][line] = hit
    return out


def find_contiguous_zero_blocks(lines_map: Dict[int, int], threshold: int) -> List[Tuple[int, int]]:
    blocks = []
    sorted_lines = sorted(lines_map.keys())
    start = None
    prev = None
    for ln in sorted_lines:
        hit = lines_map.get(ln, 0)
        if hit == 0:
            if start is None:
                start = ln
            prev = ln
        else:
            if start is not None:
                if prev - start + 1 >= threshold:
                    blocks.append((start, prev))
                start = None
                prev = None
    if start is not None and prev is not None and prev - start + 1 >= threshold:
        blocks.append((start, prev))
    return blocks


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--line-csv", required=True, help="line_coverage_counts.csv path")
    p.add_argument("--threshold", type=int, default=20, help="UNTESTED_BLOCK_THRESHOLD")
    args = p.parse_args(argv)

    per_file = parse_line_csv(args.line_csv)
    for fpath, lines_map in per_file.items():
        blocks = find_contiguous_zero_blocks(lines_map, args.threshold)
        if blocks:
            for s, e in blocks:
                print(f"{fpath},{s},{e},{e-s+1}")


if __name__ == "__main__":
    main()
