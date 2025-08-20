# pylint: disable=line-too-long, missing-function-docstring
"""Compare exported public API symbols between two baseline JSON files.

Usage:
  python scripts/api_diff.py --old benchmarks/baseline_20250816.json --new benchmarks/baseline_20250901.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Set


def load_symbols(path: Path) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("public_api_symbols", []))


def main():
    parser = argparse.ArgumentParser(description="Diff public API symbols between baselines")
    parser.add_argument("--old", required=True, help="Old baseline JSON")
    parser.add_argument("--new", required=True, help="New baseline JSON")
    args = parser.parse_args()

    old_symbols = load_symbols(Path(args.old))
    new_symbols = load_symbols(Path(args.new))

    added = sorted(new_symbols - old_symbols)
    removed = sorted(old_symbols - new_symbols)

    print("Added symbols:" if added else "No added symbols")
    for s in added:
        print(f" + {s}")

    print("Removed symbols:" if removed else "No removed symbols")
    for s in removed:
        print(f" - {s}")

    if removed:
        print("WARNING: Public API removals detected. Consider a major/minor version bump or deprecation path.")


if __name__ == "__main__":
    main()
