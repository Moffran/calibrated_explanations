from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/quality/check_governance_event_schema.py")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def minimal_valid_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "schema_version",
            "event_id",
            "event_name",
            "decision",
            "identifier",
            "source",
            "trusted",
            "actor",
            "timestamp",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": "1.0"},
            "event_id": {"type": "string"},
            "event_name": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": [
                    "accepted_registration",
                    "skipped_untrusted",
                    "skipped_denied",
                    "checksum_failure",
                    "denied_registration",
                ],
            },
            "identifier": {"type": "string"},
            "source": {"type": "string"},
            "trusted": {"type": "boolean"},
            "actor": {"type": "string"},
            "timestamp": {"type": "string"},
        },
    }


def test_should_pass_when_schema_contains_required_fields_and_decisions(tmp_path: Path) -> None:
    schema = minimal_valid_schema()
    schema_path = tmp_path / "governance_schema.json"
    report_path = tmp_path / "report.json"
    write_json(schema_path, schema)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--schema",
            str(schema_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Governance event schema check passed" in result.stdout


def test_should_fail_when_required_decision_is_missing(tmp_path: Path) -> None:
    schema = minimal_valid_schema()
    schema["properties"]["decision"]["enum"] = [  # type: ignore[index]
        "accepted_registration",
        "skipped_untrusted",
    ]
    schema_path = tmp_path / "governance_schema.json"
    report_path = tmp_path / "report.json"
    write_json(schema_path, schema)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--schema",
            str(schema_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Schema missing required decision values" in result.stdout
