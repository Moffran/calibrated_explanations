"""Compare parity snapshot JSON directories with numeric tolerance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibrated_explanations.testing import parity_compare


def _json_files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in root.rglob("*.json")
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_snapshot_dirs(
    left_dir: Path,
    right_dir: Path,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> list[dict[str, Any]]:
    """Return a list of parity diffs between two JSON snapshot directories."""
    diffs: list[dict[str, Any]] = []
    left_files = _json_files(left_dir)
    right_files = _json_files(right_dir)

    left_keys = set(left_files.keys())
    right_keys = set(right_files.keys())
    for missing in sorted(left_keys - right_keys):
        diffs.append({"type": "missing_right_file", "file": missing})
    for missing in sorted(right_keys - left_keys):
        diffs.append({"type": "missing_left_file", "file": missing})

    for rel_path in sorted(left_keys & right_keys):
        left_payload = _load_json(left_files[rel_path])
        right_payload = _load_json(right_files[rel_path])
        file_diffs = parity_compare(left_payload, right_payload, rtol=rtol, atol=atol)
        for item in file_diffs:
            diffs.append({"type": "payload_diff", "file": rel_path, "diff": item})

    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare parity snapshot directories.")
    parser.add_argument("--left", required=True, help="Left snapshot directory")
    parser.add_argument("--right", required=True, help="Right snapshot directory")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    args = parser.parse_args()

    left_dir = Path(args.left)
    right_dir = Path(args.right)
    if not left_dir.exists() or not right_dir.exists():
        print("Both --left and --right directories must exist.")
        return 2

    diffs = compare_snapshot_dirs(left_dir, right_dir, rtol=args.rtol, atol=args.atol)
    if diffs:
        print(f"Detected {len(diffs)} parity diff(s) between snapshots.")
        for diff in diffs[:50]:
            print(json.dumps(diff, indent=2, sort_keys=True))
        if len(diffs) > 50:
            print(f"... truncated {len(diffs) - 50} additional diff(s)")
        return 1

    print("Parity snapshots match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
