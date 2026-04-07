"""Validate governance event schema contract for plugin decision events."""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate governance_event_schema_v1.json structural contract.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("src/calibrated_explanations/schemas/governance_event_schema_v1.json"),
        help="Path to governance event schema JSON file.",
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
        findings.append(Finding("error", f"Schema file not found: {args.schema}"))
        _write_report(args.report, findings)
        print(f"ERROR: Schema file not found: {args.schema}")
        return 1

    schema = _load_json(args.schema)

    if jsonschema is not None:
        try:
            validator_cls = jsonschema.validators.validator_for(schema)  # type: ignore[attr-defined]
            validator_cls.check_schema(schema)
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

    if not findings:
        findings.append(Finding("info", "Governance event schema validation passed."))
        _write_report(args.report, findings)
        print("Governance event schema check passed.")
        return 0

    _write_report(args.report, findings)
    for item in findings:
        prefix = "ERROR" if item.level == "error" else item.level.upper()
        print(f"{prefix}: {item.message}")
    return 1 if any(item.level == "error" for item in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
