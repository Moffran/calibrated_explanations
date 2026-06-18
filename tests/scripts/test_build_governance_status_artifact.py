"""Tests for scripts/quality/build_governance_status_artifact.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/quality/build_governance_status_artifact.py")
SCHEMA = Path("development/schemas/governance_status_schema_v1.json")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# build_artifact status derivation
# ---------------------------------------------------------------------------


def test_should_build_valid_artifact_when_all_reports_present(tmp_path, monkeypatch):
    """build_artifact maps report files to passed statuses when reports are healthy."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    monkeypatch.chdir(tmp_path)
    write_json(
        tmp_path / "reports/quality/governance_event_schema_report.json",
        {"ok": True, "findings": [], "version": 1},
    )
    write_json(tmp_path / "reports/config_manager_usage_report.json", {"total_violations": 0})
    write_json(tmp_path / "reports/quality/logging_domain_report.json", {"ok": True})
    write_json(tmp_path / "reports/quality/no_local_paths_report.json", {"ok": True})

    artifact = build_artifact()
    assert artifact["schema_checks"]["governance_event_schema"] == "passed"
    assert artifact["schema_checks"]["config_manager_usage"] == "passed"
    assert artifact["schema_checks"]["logging_domains"] == "passed"
    assert artifact["schema_checks"]["no_local_paths"] == "passed"


def test_should_return_unavailable_when_report_missing(tmp_path, monkeypatch):
    """build_artifact returns unavailable statuses when reports are missing."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    monkeypatch.chdir(tmp_path)
    artifact = build_artifact()
    assert artifact["schema_checks"]["governance_event_schema"] == "unavailable"
    assert artifact["schema_checks"]["config_manager_usage"] == "unavailable"
    assert artifact["schema_checks"]["logging_domains"] == "unavailable"
    assert artifact["schema_checks"]["no_local_paths"] == "unavailable"


def test_should_return_failed_when_report_ok_false(tmp_path, monkeypatch):
    """build_artifact maps ok=False reports to failed status."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "reports/quality/governance_event_schema_report.json", {"ok": False})
    artifact = build_artifact()
    assert artifact["schema_checks"]["governance_event_schema"] == "failed"


def test_should_return_passed_for_config_manager_zero_violations(tmp_path, monkeypatch):
    """build_artifact maps config manager total_violations==0 to passed."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "reports/config_manager_usage_report.json", {"total_violations": 0})
    artifact = build_artifact()
    assert artifact["schema_checks"]["config_manager_usage"] == "passed"


def test_should_return_failed_for_config_manager_nonzero_violations(tmp_path, monkeypatch):
    """build_artifact maps config manager total_violations>0 to failed."""
    from scripts.quality.build_governance_status_artifact import build_artifact

    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "reports/config_manager_usage_report.json", {"total_violations": 3})
    artifact = build_artifact()
    assert artifact["schema_checks"]["config_manager_usage"] == "failed"


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
# schema validation
# ---------------------------------------------------------------------------


def test_should_validate_artifact_against_schema():
    """Schema validation passes for a well-formed artifact produced by build_artifact."""
    jsonschema = pytest.importorskip("jsonschema")
    from scripts.quality.build_governance_status_artifact import build_artifact

    artifact = build_artifact()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(instance=artifact, schema=schema)
    assert artifact["schema_version"] == schema["properties"]["schema_version"]["const"]


def test_should_fail_closed_on_schema_violation():
    """JSON schema validation fails for malformed artifact payloads."""
    jsonschema = pytest.importorskip("jsonschema")

    bad_artifact = {
        "schema_version": "9.9",
        "generated_at": "bad",
        "extra_field": True,
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=bad_artifact, schema=schema)


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

    It is a derived CI artifact and must live under development/schemas/.
    """
    runtime_schema_dir = Path("src/calibrated_explanations/schemas")
    if runtime_schema_dir.is_dir():
        found = list(runtime_schema_dir.glob("governance_status*.json"))
        assert not found, (
            "governance_status schema must not be placed in the runtime package "
            f"(found: {found}). It belongs in development/schemas/."
        )
