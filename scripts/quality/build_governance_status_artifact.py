"""Build governance status artifact from existing quality reports.

This script aggregates CI quality check results into a single machine-readable
governance status artifact. It is a **derived CI artifact** and does not replace
runtime governance event schemas (``governance_event_schema_v1.json``,
``governance_config_event_schema_v1.json``).

Usage (local)::

    python scripts/quality/build_governance_status_artifact.py \\
        --output reports/governance/governance_status.json --validate

Usage (CI, with lint status flags)::

    python scripts/quality/build_governance_status_artifact.py \\
        --output reports/governance/governance_status.json --validate \\
        --lint-local-checks-pr passed --lint-mypy passed --lint-ruff passed

See ``docs/improvement/governance_status_artifact.md`` for full documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - optional; validation degrades gracefully
    jsonschema = None

_SCHEMA_PATH = Path("development/schemas/governance_status_schema_v1.json")
_DEFAULT_OUTPUT = Path("reports/governance/governance_status.json")

# Mapping from schema_checks key to the quality report file that provides the status.
_REPORT_SOURCES: dict[str, Path] = {
    "governance_event_schema": Path(
        "reports/quality/governance_event_schema_report.json"
    ),
    "config_manager_usage": Path("reports/config_manager_usage_report.json"),
    "logging_domains": Path("reports/quality/logging_domain_report.json"),
    "no_local_paths": Path("reports/quality/no_local_paths_report.json"),
}


def _status_from_report(path: Path) -> str:
    """Derive a ``passed``/``failed``/``unavailable`` status from a quality report file.

    Handles two report formats:

    * Standard reports: ``{"ok": true|false, ...}``
    * Config-manager report: ``{"total_violations": 0, ...}`` (no ``ok`` key)
    """
    if not path.is_file():
        return "unavailable"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unavailable"
    if "ok" in data:
        return "passed" if data["ok"] else "failed"
    if "total_violations" in data:
        return "passed" if data["total_violations"] == 0 else "failed"
    return "unavailable"


_MYPY_CANDIDATES = [
    "src/calibrated_explanations/core/exceptions.py",
    "src/calibrated_explanations/core/validation.py",
    "src/calibrated_explanations/api/params.py",
]


def _run_lint_checks() -> dict[str, str]:
    """Run ruff and mypy locally and return a lint_status dict.

    ``local_checks_pr`` is always ``"unavailable"`` from this path because the
    full PR gate requires the complete test suite; only CI can set it to
    ``"passed"``/``"failed"``.
    """

    def _check(command: list[str]) -> str:
        try:
            result = subprocess.run(command, check=False, capture_output=True)
            return "passed" if result.returncode == 0 else "failed"
        except FileNotFoundError:
            return "unavailable"

    ruff_status = _check([sys.executable, "-m", "ruff", "check", "src/"])
    mypy_targets = [p for p in _MYPY_CANDIDATES if Path(p).is_file()]
    mypy_status = (
        _check([sys.executable, "-m", "mypy", *mypy_targets, "--config-file", "pyproject.toml"])
        if mypy_targets
        else "unavailable"
    )
    return {"local_checks_pr": "unavailable", "mypy": mypy_status, "ruff": ruff_status}


def build_artifact(*, lint_status: dict[str, str] | None = None) -> dict:
    """Build the governance status artifact payload.

    Parameters
    ----------
    lint_status : dict[str, str] or None
        Mapping with keys ``local_checks_pr``, ``mypy``, ``ruff``. Each value
        must be one of ``"passed"``, ``"failed"``, ``"unavailable"``. When
        ``None``, all values default to ``"unavailable"``.

    Returns
    -------
    dict
        Artifact payload conforming to ``governance_status_schema_v1.json``.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_info: dict[str, str | None] = {
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "commit": os.environ.get("GITHUB_SHA"),
    }
    lint: dict[str, str] = lint_status or {
        "local_checks_pr": "unavailable",
        "mypy": "unavailable",
        "ruff": "unavailable",
    }
    schema_checks: dict[str, str] = {
        key: _status_from_report(path)
        for key, path in _REPORT_SOURCES.items()
    }
    return {
        "schema_version": "1.0",
        "generated_at": now,
        "run": run_info,
        "lint": lint,
        "schema_checks": schema_checks,
    }


def _validate_artifact(artifact: dict, schema_path: Path) -> list[str]:
    """Validate artifact against the governance status schema.

    Returns a list of error/warning messages. Empty list means valid (or
    validation was skipped because ``jsonschema`` is not installed).
    """
    if jsonschema is None:
        return ["jsonschema not installed; skipping schema validation."]
    if not schema_path.is_file():
        return [f"Schema file not found: {schema_path}; skipping validation."]
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"Failed to read schema file {schema_path}: {exc}"]
    try:
        jsonschema.validate(instance=artifact, schema=schema)  # type: ignore[attr-defined]
        return []
    except Exception as exc:  # noqa: BLE001
        return [f"Schema validation failed: {exc}"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the governance status artifact producer."""
    parser = argparse.ArgumentParser(
        description="Build governance status artifact from quality check reports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output path for governance_status.json artifact.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output against governance_status_schema_v1.json.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=_SCHEMA_PATH,
        help="Path to governance status schema file for validation.",
    )
    parser.add_argument(
        "--lint-local-checks-pr",
        choices=["passed", "failed", "unavailable"],
        default="unavailable",
        dest="lint_local_checks_pr",
        help="Result of the local-checks-pr / PR lint gate.",
    )
    parser.add_argument(
        "--lint-mypy",
        choices=["passed", "failed", "unavailable"],
        default="unavailable",
        dest="lint_mypy",
        help="Result of the mypy type-check step.",
    )
    parser.add_argument(
        "--lint-ruff",
        choices=["passed", "failed", "unavailable"],
        default="unavailable",
        dest="lint_ruff",
        help="Result of the ruff lint step.",
    )
    parser.add_argument(
        "--run-lint",
        action="store_true",
        dest="run_lint",
        help=(
            "Run ruff and mypy locally and use the actual exit codes for lint status. "
            "Overrides --lint-ruff and --lint-mypy. local_checks_pr stays unavailable "
            "because the full PR gate requires the test suite (only CI can set it)."
        ),
    )
    args = parser.parse_args(argv)

    if args.run_lint:
        lint_status = _run_lint_checks()
    else:
        lint_status = {
            "local_checks_pr": args.lint_local_checks_pr,
            "mypy": args.lint_mypy,
            "ruff": args.lint_ruff,
        }
    artifact = build_artifact(lint_status=lint_status)

    if args.validate:
        errors = _validate_artifact(artifact, args.schema)
        for msg in errors:
            is_hard = "Schema validation failed" in msg
            stream = sys.stderr if is_hard else sys.stdout
            prefix = "ERROR" if is_hard else "WARN"
            print(f"{prefix}: {msg}", file=stream)
        if any("Schema validation failed" in e for e in errors):
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Governance status artifact written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
