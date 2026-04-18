"""Validate governance event schema contracts for plugin and config events."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - enforced in CI/local checks
    jsonschema = None

REQUIRED_DECISIONS = {
    "accepted_registration",
    "skipped_untrusted",
    "skipped_denied",
    "checksum_failure",
    "denied_registration",
}
REQUIRED_FIELDS = {
    "schema_version",
    "event_id",
    "event_name",
    "decision",
    "identifier",
    "source",
    "trusted",
    "actor",
    "timestamp",
}
REQUIRED_CONFIG_EVENT_TYPES = {
    "resolve",
    "export",
    "validation_failure",
}
REQUIRED_CONFIG_FIELDS = {
    "schema_version",
    "event_id",
    "event_name",
    "event_type",
    "profile_id",
    "config_schema_version",
    "strict",
    "source_count",
    "validation_issue_count",
    "timestamp",
}

PLUGIN_EVENT_FIXTURE = {
    "schema_version": "1.0",
    "event_id": "25a35f7b-d612-4b6f-8f6e-f6d695f0f8fe",
    "event_name": "plugin.registration.decision",
    "decision": "accepted_registration",
    "identifier": "calibrated.fast",
    "source": "builtin",
    "trusted": True,
    "actor": "PluginManager.register",
    "timestamp": "2026-04-08T12:00:00+00:00",
}

CONFIG_EVENT_FIXTURES = {
    "resolve": {
        "schema_version": "1.0",
        "event_id": "0baf9660-41b7-4b88-a3a1-665f0894f2dc",
        "event_name": "config.lifecycle",
        "event_type": "resolve",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": True,
        "source_count": 2,
        "validation_issue_count": 0,
        "timestamp": "2026-04-08T12:00:00+00:00",
        "details": None,
    },
    "export": {
        "schema_version": "1.0",
        "event_id": "6c7c8c3f-d8f4-4f9a-9d4f-7f5d2f28f1a5",
        "event_name": "config.lifecycle",
        "event_type": "export",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": True,
        "source_count": 2,
        "validation_issue_count": 0,
        "timestamp": "2026-04-08T12:00:01+00:00",
        "details": {"diagnostic_only": True},
    },
    "validation_failure": {
        "schema_version": "1.0",
        "event_id": "b4cb8b5f-c1f8-4f7b-bafd-62a3fda9dd32",
        "event_name": "config.lifecycle",
        "event_type": "validation_failure",
        "profile_id": "default",
        "config_schema_version": "1",
        "strict": False,
        "source_count": 1,
        "validation_issue_count": 1,
        "timestamp": "2026-04-08T12:00:02+00:00",
        "details": {"location": "pyproject.plugins", "issue_count": 1},
    },
}


@dataclass(frozen=True)
class Finding:
    """Validation finding entry."""

    level: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_report(path: Path, findings: list[Finding]) -> None:
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "ok": not any(item.level == "error" for item in findings),
        "findings": [item.as_dict() for item in findings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_fixture(
    *,
    schema: dict,
    payload: dict,
    label: str,
    findings: list[Finding],
) -> None:
    if jsonschema is None:
        findings.append(
            Finding(
                "warning",
                f"jsonschema package not installed; skipping fixture validation for {label}.",
            )
        )
        return
    try:
        jsonschema.validate(instance=payload, schema=schema)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - exceptional path
        findings.append(Finding("error", f"Fixture validation failed for {label}: {exc}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate governance_event_schema_v1.json structural contract.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("src/calibrated_explanations/schemas/governance_event_schema_v1.json"),
        help="Path to plugin governance event schema JSON file.",
    )
    parser.add_argument(
        "--config-schema",
        type=Path,
        default=Path("src/calibrated_explanations/schemas/governance_config_event_schema_v1.json"),
        help="Path to config governance event schema JSON file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quality/governance_event_schema_report.json"),
        help="Output report path.",
    )
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    if jsonschema is None:
        findings.append(
            Finding(
                "warning",
                "jsonschema package not installed; skipping JSON-Schema meta-validation.",
            )
        )

    if not args.schema.is_file():
        findings.append(Finding("error", f"Plugin schema file not found: {args.schema}"))
        _write_report(args.report, findings)
        print(f"ERROR: Plugin schema file not found: {args.schema}")
        return 1
    if not args.config_schema.is_file():
        findings.append(Finding("error", f"Config schema file not found: {args.config_schema}"))
        _write_report(args.report, findings)
        print(f"ERROR: Config schema file not found: {args.config_schema}")
        return 1

    schema = _load_json(args.schema)
    config_schema = _load_json(args.config_schema)

    if jsonschema is not None:
        try:
            validator_cls = jsonschema.validators.validator_for(schema)  # type: ignore[attr-defined]
            validator_cls.check_schema(schema)
            config_validator_cls = jsonschema.validators.validator_for(config_schema)  # type: ignore[attr-defined]
            config_validator_cls.check_schema(config_schema)
        except Exception as exc:  # pragma: no cover - exceptional path
            findings.append(Finding("error", f"Schema is not a valid JSON Schema document: {exc}"))

    required = set(schema.get("required", []))
    missing_required = sorted(REQUIRED_FIELDS - required)
    if missing_required:
        findings.append(
            Finding(
                "error",
                "Schema missing required governance fields: " + ", ".join(missing_required),
            )
        )

    decision_enum = set(
        schema.get("properties", {}).get("decision", {}).get("enum", [])
    )
    missing_decisions = sorted(REQUIRED_DECISIONS - decision_enum)
    if missing_decisions:
        findings.append(
            Finding(
                "error",
                "Schema missing required decision values: " + ", ".join(missing_decisions),
            )
        )

    config_required = set(config_schema.get("required", []))
    config_missing_required = sorted(REQUIRED_CONFIG_FIELDS - config_required)
    if config_missing_required:
        findings.append(
            Finding(
                "error",
                "Config schema missing required fields: " + ", ".join(config_missing_required),
            )
        )

    event_type_enum = set(
        config_schema.get("properties", {}).get("event_type", {}).get("enum", [])
    )
    missing_event_types = sorted(REQUIRED_CONFIG_EVENT_TYPES - event_type_enum)
    if missing_event_types:
        findings.append(
            Finding(
                "error",
                "Config schema missing required event_type values: " + ", ".join(missing_event_types),
            )
        )

    if config_schema.get("properties", {}).get("event_name", {}).get("const") != "config.lifecycle":
        findings.append(
            Finding("error", "Config schema event_name const must be 'config.lifecycle'.")
        )

    if config_schema.get("additionalProperties") is not False:
        findings.append(
            Finding(
                "error",
                "Config schema must set additionalProperties to false for governance hardening.",
            )
        )

    _validate_fixture(
        schema=schema,
        payload=PLUGIN_EVENT_FIXTURE,
        label="plugin governance event fixture",
        findings=findings,
    )
    for event_type, payload in CONFIG_EVENT_FIXTURES.items():
        _validate_fixture(
            schema=config_schema,
            payload=payload,
            label=f"config governance event fixture ({event_type})",
            findings=findings,
        )

    if not findings:
        findings.append(Finding("info", "Governance event schema validation passed."))
        findings.append(Finding("info", "Config governance event schema validation passed."))
        findings.append(Finding("info", "Governance event fixture validation passed."))
        _write_report(args.report, findings)
        print("Governance event schema checks passed.")
        return 0

    _write_report(args.report, findings)
    for item in findings:
        prefix = "ERROR" if item.level == "error" else item.level.upper()
        print(f"{prefix}: {item.message}")
    return 1 if any(item.level == "error" for item in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
