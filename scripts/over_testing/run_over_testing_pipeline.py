"""Run the over-testing analysis pipeline end-to-end."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest with per-test coverage contexts, then generate over-testing reports."
        )
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=[
            "--cov=src/calibrated_explanations",
            "--cov-context=test",
            "--cov-fail-under=90",
        ],
        help="Additional arguments passed to pytest.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue report generation even if pytest fails.",
    )
    parser.add_argument(
        "--contexts-per-hotspot",
        type=int,
        default=25,
        help="Number of test contexts to include per hotspot.",
    )
    return parser.parse_args()


def _run_step(command: list[str], label: str, allow_failure: bool, env: dict[str, str] | None = None) -> int:
    print(f"\n==> {label}")
    print(" ".join(command))
    result = subprocess.run(command, check=False, env=env)
    if result.returncode != 0:
        print(f"Step failed ({label}) with exit code {result.returncode}.")
        if not allow_failure:
            return result.returncode
    return 0


def main() -> int:
    args = _parse_args()
    python = sys.executable
    coverage_file = Path(".coverage.over_testing")
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_file)

    pytest_cmd = [python, "-m", "pytest", *args.pytest_args]
    report_cmd = [
        python,
        "scripts/over_testing/over_testing_report.py",
        "--require-multiple-contexts",
        "--coverage-file",
        str(coverage_file),
    ]
    triage_cmd = [
        python,
        "scripts/over_testing/over_testing_triage.py",
        "--include-contexts",
        "--contexts-per-hotspot",
        str(args.contexts_per_hotspot),
    ]

    allow_failure = args.continue_on_failure
    step = _run_step(pytest_cmd, "Run pytest with per-test contexts", allow_failure, env=env)
    if step != 0:
        return step

    step = _run_step(report_cmd, "Generate over-testing report", allow_failure, env=env)
    if step != 0:
        return step

    step = _run_step(triage_cmd, "Generate triage report", allow_failure, env=env)
    if step != 0:
        return step

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
