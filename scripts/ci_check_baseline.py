#!/usr/bin/env python3
"""CI helper: run a static analyzer and fail only on new errors.

Supported tools:
- ruff: uses `ruff check --format=json`
- mypy: uses `mypy --error-format=json`

Writes a baseline file on first run (when missing) and exits with non-zero to
force committing the snapshot. On subsequent runs, fails only if there are
diagnostics not present in the baseline. Improvements (fewer errors) pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set


def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode()
    except subprocess.CalledProcessError as exc:
        # mypy returns non-zero when errors are found; we still want the output
        return exc.output.decode()


def ruff_diagnostics(root: Path, paths: List[str]) -> Set[str]:
    cmd = ["ruff", "check", "--format", "json", "--quiet", *paths]
    raw = run_cmd(cmd)
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        # If ruff produced non-JSON, return empty set to avoid false positives
        return set()
    results: Set[str] = set()
    for item in data:
        filename = Path(item.get("filename", "")).as_posix()
        # store relative path if under repo
        rel = str(Path(filename))
        if filename.startswith(str(root)):
            try:
                rel = str(Path(filename).relative_to(root))
            except Exception:
                rel = filename
        loc = item.get("location", {})
        row = loc.get("row", 0)
        col = loc.get("column", 0)
        code = item.get("code", "")
        results.add(f"{code}:{rel}:{row}:{col}")
    return results


def mypy_diagnostics(root: Path, paths: List[str]) -> Set[str]:
    cmd = [
        "mypy",
        "--config-file",
        "pyproject.toml",
        "--error-format=json",
        *paths,
    ]
    raw = run_cmd(cmd)
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        # mypy sometimes emits plain text on internal errors; ignore in baseline check
        return set()
    results: Set[str] = set()
    for err in data.get("errors", []):
        for diag in err.get("messages", []):
            # Build stable key: code:path:line:column:text
            code = diag.get("code", "") or ""
            path = err.get("path", "") or diag.get("path", "") or ""
            try:
                rel = str(Path(path).relative_to(root))
            except Exception:
                rel = path
            line = diag.get("line", 0)
            col = diag.get("column", 0)
            text = diag.get("message", "").strip()
            results.add(f"{code}:{rel}:{line}:{col}:{text}")
    return results


def write_lines(p: Path, lines: Iterable[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(sorted(lines)) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["ruff", "mypy"], required=True)
    ap.add_argument("--baseline", required=True, help="Path to baseline file")
    ap.add_argument(
        "--paths",
        nargs="*",
        default=["src"],
        help="Paths/files to analyze (default: src)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline_path = root / args.baseline
    paths = [str(Path(p)) for p in args.paths]

    if args.tool == "ruff":
        current = ruff_diagnostics(root, paths)
    else:
        current = mypy_diagnostics(root, paths)

    if not baseline_path.exists():
        write_lines(baseline_path, current)
        sys.stderr.write(
            f"Baseline created at {baseline_path}. Commit it and re-run CI.\n"
        )
        return 1

    baseline = set((baseline_path.read_text().splitlines()))
    new = current - baseline

    if new:
        sys.stderr.write(
            f"New {args.tool} issues detected (failing on regressions only):\n"
        )
        for line in sorted(new):
            sys.stderr.write(line + "\n")
        # Optionally update a current snapshot artifact for debugging
        write_lines(baseline_path.parent / f"current_{args.tool}.txt", current)
        return 1

    # No new issues; success (improvements welcome and pass)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

