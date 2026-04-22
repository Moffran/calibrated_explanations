"""Tests for scripts/quality/build_governance_status_artifact.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/quality/build_governance_status_artifact.py")
SCHEMA = Path("docs/improvement/schemas/governance_status_schema_v1.json")


# ---------------------------------------------------------------------------
# _status_from_report
# ---------------------------------------------------------------------------


def test_should_build_valid_artifact_when_all_reports_present(tmp_path):
    """_status_from_report returns 'passed' for a report with ok=True."""
    from scripts.quality.build_governance_status_artifact import _status_from_report

    report = tmp_path / "governance_event_schema_report.json"
    report.write_text(
        json.dumps({"ok": True, "findings": [], "version": 1}),
        encoding="utf-8",
    )
    assert _status_from_report(report) == "passed"


def test_should_return_unavailable_when_report_missing(tmp_path):
    """_status_from_report returns 'unavailable' when the report file does not exist."""
    from scripts.quality.build_governance_status_artifact import _status_from_report

    assert _status_from_report(tmp_path / "nonexistent.json") == "unavailable"


def test_should_return_failed_when_report_ok_false(tmp_path):
    """_status_from_report returns 'failed' when ok is False."""
    from scripts.quality.build_governance_status_artifact import _status_from_report

    report = tmp_path / "bad_report.json"
    report.write_text(json.dumps({"ok": False}), encoding="utf-8")
    assert _status_from_report(report) == "failed"


def test_should_return_passed_for_config_manager_zero_violations(tmp_path):
    """_status_from_report maps total_violations==0 to 'passed' (config_manager format)."""
    from scripts.quality.build_governance_status_artifact import _status_from_report

    report = tmp_path / "config_manager_usage_report.json"
    report.write_text(
        json.dumps({"total_violations": 0, "version": 1}),
        encoding="utf-8",
    )
    assert _status_from_report(report) == "passed"


def test_should_return_failed_for_config_manager_nonzero_violations(tmp_path):
    """_status_from_report maps total_violations>0 to 'failed' (config_manager format)."""
    from scripts.quality.build_governance_status_artifact import _status_from_report

    report = tmp_path / "config_manager_usage_report.json"
    report.write_text(
        json.dumps({"total_violations": 3, "version": 1}),
        encoding="utf-8",
    )
    assert _status_from_report(report) == "failed"


# ---------------------------------------------------------------------------
# build_artifact
# ---------------------------------------------------------------------------


def test_should_build_artifact_with_required_keys():
    """build_artifact returns payload with all required schema keys."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    artifact = build_artifact()
    assert artifact["schema_version"] == "1.0"
    assert "generated_at" in artifact
    assert "run" in artifact
    assert "lint" in artifact
    assert "schema_checks" in artifact
    # schema_checks keys come from _REPORT_SOURCES
    for key in (
        "governance_event_schema",
        "config_manager_usage",
        "logging_domains",
        "no_local_paths",
    ):
        assert key in artifact["schema_checks"]


def test_should_build_artifact_lint_defaults_to_unavailable():
    """build_artifact defaults all lint values to 'unavailable' when no lint_status is given."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    artifact = build_artifact()
    assert artifact["lint"]["local_checks_pr"] == "unavailable"
    assert artifact["lint"]["mypy"] == "unavailable"
    assert artifact["lint"]["ruff"] == "unavailable"


def test_should_build_artifact_with_custom_lint_status():
    """build_artifact propagates supplied lint_status into the payload."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    artifact = build_artifact(
        lint_status={"local_checks_pr": "passed", "mypy": "passed", "ruff": "failed"}
    )
    assert artifact["lint"]["local_checks_pr"] == "passed"
    assert artifact["lint"]["ruff"] == "failed"


# ---------------------------------------------------------------------------
# _validate_artifact
# ---------------------------------------------------------------------------


def test_should_validate_artifact_against_schema():
    """Schema validation passes for a well-formed artifact produced by build_artifact."""
    pytest.importorskip("jsonschema")
    from scripts.quality.build_governance_status_artifact import (
        _validate_artifact,
        build_artifact,
    )

    artifact = build_artifact()
    errors = _validate_artifact(artifact, SCHEMA)
    hard_errors = [e for e in errors if "Schema validation failed" in e]
    assert not hard_errors, hard_errors


def test_should_fail_closed_on_schema_violation():
    """_validate_artifact reports hard errors when the payload violates the schema."""
    pytest.importorskip("jsonschema")
    from scripts.quality.build_governance_status_artifact import _validate_artifact

    bad_artifact = {
        "schema_version": "9.9",
        "generated_at": "bad",
        "extra_field": True,
    }
    errors = _validate_artifact(bad_artifact, SCHEMA)
    assert any(errors), "Expected validation errors for a malformed artifact"


# ---------------------------------------------------------------------------
# CLI / file I/O
# ---------------------------------------------------------------------------


def test_should_write_output_to_specified_path(tmp_path):
    """The script writes governance_status.json to the specified output path."""
    output = tmp_path / "governance_status.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert "schema_checks" in data


def test_should_write_artifact_with_validate_flag(tmp_path):
    """The script succeeds and writes the file when --validate is passed."""
    output = tmp_path / "gs.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--validate"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()


def test_should_create_output_directory_if_missing(tmp_path):
    """The script creates missing parent directories for the output path."""
    output = tmp_path / "deep" / "nested" / "governance_status.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()


def test_should_pass_lint_flags_to_artifact(tmp_path):
    """Lint status CLI flags are reflected in the artifact payload."""
    output = tmp_path / "governance_status.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--lint-local-checks-pr",
            "passed",
            "--lint-mypy",
            "failed",
            "--lint-ruff",
            "unavailable",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["lint"]["local_checks_pr"] == "passed"
    assert data["lint"]["mypy"] == "failed"
    assert data["lint"]["ruff"] == "unavailable"


# ---------------------------------------------------------------------------
# Placement invariant
# ---------------------------------------------------------------------------


def test_should_assert_governance_status_schema_not_in_runtime_package():
    """Governance status schema must not be placed in the runtime package (src/).

    It is a derived CI artifact and must live under docs/improvement/schemas/.
    """
    runtime_schema_dir = Path("src/calibrated_explanations/schemas")
    if runtime_schema_dir.is_dir():
        found = list(runtime_schema_dir.glob("governance_status*.json"))
        assert not found, (
            "governance_status schema must not be placed in the runtime package "
            f"(found: {found}). It belongs in docs/improvement/schemas/."
        )
