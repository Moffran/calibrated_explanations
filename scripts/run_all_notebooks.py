"""Convenience runner for executing all notebooks under notebooks/ recursively.

This is a thin wrapper around scripts.docs.run_notebooks so contributors can run
an end-to-end notebook sweep with a short command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.docs.run_notebooks import run_notebooks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for notebook execution."""
    parser = argparse.ArgumentParser(
        description="Execute all notebooks under notebooks/ and subfolders."
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path("notebooks"),
        help="Root directory to search recursively for .ipynb files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/docs/notebook_execution_report.json"),
        help="Path to write the JSON execution report.",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=30,
        help="Per-cell timeout in seconds.",
    )
    parser.add_argument(
        "--notebook-timeout",
        type=int,
        default=300,
        help="Per-notebook wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--mode",
        choices=("advisory", "blocking"),
        default="advisory",
        help="Execution mode; blocking returns exit code 1 on failures.",
    )
    return parser.parse_args()


def main() -> int:
    """Run notebook execution with CLI-provided settings."""
    args = parse_args()
    return run_notebooks(
        notebooks_dir=args.notebooks_dir,
        output_path=args.output,
        cell_timeout=args.cell_timeout,
        notebook_timeout=args.notebook_timeout,
        mode=args.mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
