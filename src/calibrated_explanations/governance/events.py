"""Governance event envelope and emission helpers.

This module defines a machine-checkable event contract for governance/audit
events emitted from runtime plugin decision paths.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Mapping
from uuid import uuid4

from ..logging import ensure_logging_context_filter, get_logging_context, logging_context
from ..utils.exceptions import ValidationError

try:  # Optional dependency
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    jsonschema = None

PLUGIN_GOVERNANCE_DECISIONS: tuple[str, ...] = (
    "accepted_registration",
    "skipped_untrusted",
    "skipped_denied",
    "checksum_failure",
    "denied_registration",
)
CONFIG_GOVERNANCE_EVENT_TYPES: tuple[str, ...] = (
    "resolve",
    "export",
    "validation_failure",
)


def _schema_json() -> dict[str, Any]:  # pragma: no cover - tiny IO helper
    with (
        resources.files("calibrated_explanations.schemas")
        .joinpath("governance_event_schema_v1.json")
        .open("r", encoding="utf-8") as f
    ):
        return json.load(f)


def _config_schema_json() -> dict[str, Any]:  # pragma: no cover - tiny IO helper
    with (
        resources.files("calibrated_explanations.schemas")
        .joinpath("governance_config_event_schema_v1.json")
        .open("r", encoding="utf-8") as f
    ):
        return json.load(f)


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_governance_event(payload: Mapping[str, Any]) -> None:
    """Validate governance event payload shape.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Governance event payload.
    """
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=dict(payload), schema=_schema_json())  # type: ignore[attr-defined]
        except jsonschema.exceptions.ValidationError as exc:  # type: ignore[union-attr]
            raise ValidationError(
                "governance event payload failed schema validation",
                details={"validator": "jsonschema"},
            ) from exc
        return

    required_keys = (
        "schema_version",
        "event_id",
        "event_name",
        "decision",
        "identifier",
        "source",
        "trusted",
        "actor",
        "timestamp",
    )
    for key in required_keys:
        if key not in payload:
            raise ValidationError(
                f"governance event missing required field '{key}'",
                details={"field": key},
            )
    if payload.get("schema_version") != "1.0":
        raise ValidationError(
            "governance event schema_version must be '1.0'",
            details={"schema_version": payload.get("schema_version")},
        )
    decision = payload.get("decision")
    if decision not in PLUGIN_GOVERNANCE_DECISIONS:
        raise ValidationError(
            "governance event decision is invalid",
            details={"decision": decision, "allowed": list(PLUGIN_GOVERNANCE_DECISIONS)},
        )
    if not isinstance(payload.get("trusted"), bool):
        raise ValidationError(
            "governance event trusted must be a bool", details={"field": "trusted"}
        )
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValidationError(
            "governance event timestamp must be an ISO-8601 string",
            details={"field": "timestamp"},
        )
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(
            "governance event timestamp must be valid ISO-8601",
            details={"field": "timestamp", "value": timestamp},
        ) from exc


def validate_config_governance_event(payload: Mapping[str, Any]) -> None:
    """Validate config-governance lifecycle event payload shape."""
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=dict(payload), schema=_config_schema_json())  # type: ignore[attr-defined]
        except jsonschema.exceptions.ValidationError as exc:  # type: ignore[union-attr]
            raise ValidationError(
                "config governance event payload failed schema validation",
                details={"validator": "jsonschema"},
            ) from exc
        return

    required_keys = (
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
    )
    for key in required_keys:
        if key not in payload:
            raise ValidationError(
                f"config governance event missing required field '{key}'",
                details={"field": key},
            )
    if payload.get("schema_version") != "1.0":
        raise ValidationError(
            "config governance event schema_version must be '1.0'",
            details={"schema_version": payload.get("schema_version")},
        )
    if payload.get("event_name") != "config.lifecycle":
        raise ValidationError(
            "config governance event_name must be 'config.lifecycle'",
            details={"event_name": payload.get("event_name")},
        )
    event_type = payload.get("event_type")
    if event_type not in CONFIG_GOVERNANCE_EVENT_TYPES:
        raise ValidationError(
            "config governance event_type is invalid",
            details={"event_type": event_type, "allowed": list(CONFIG_GOVERNANCE_EVENT_TYPES)},
        )
    if not isinstance(payload.get("strict"), bool):
        raise ValidationError(
            "config governance strict must be a bool", details={"field": "strict"}
        )
    for numeric_field in ("source_count", "validation_issue_count"):
        value = payload.get(numeric_field)
        if not isinstance(value, int) or value < 0:
            raise ValidationError(
                f"config governance {numeric_field} must be a non-negative integer",
                details={"field": numeric_field, "value": value},
            )
    details = payload.get("details")
    if event_type == "resolve":
        if details is not None:
            raise ValidationError(
                "config governance resolve details must be null",
                details={"field": "details", "event_type": event_type},
            )
    elif event_type == "export":
        if details != {"diagnostic_only": True}:
            raise ValidationError(
                "config governance export details must be {'diagnostic_only': True}",
                details={"field": "details", "event_type": event_type},
            )
    elif event_type == "validation_failure":
        if not isinstance(details, Mapping):
            raise ValidationError(
                "config governance validation_failure details must be an object",
                details={"field": "details", "event_type": event_type},
            )
        if set(details.keys()) != {"location", "issue_count"}:
            raise ValidationError(
                "config governance validation_failure details keys must be {'location', 'issue_count'}",
                details={"field": "details", "event_type": event_type},
            )
        issue_count = details.get("issue_count")
        if not isinstance(issue_count, int) or issue_count < 0:
            raise ValidationError(
                "config governance validation_failure details.issue_count must be a non-negative integer",
                details={"field": "details.issue_count", "value": issue_count},
            )
        location = details.get("location")
        if location is not None and not isinstance(location, str):
            raise ValidationError(
                "config governance validation_failure details.location must be a string or null",
                details={"field": "details.location", "value": location},
            )
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValidationError(
            "config governance event timestamp must be an ISO-8601 string",
            details={"field": "timestamp"},
        )
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(
            "config governance event timestamp must be valid ISO-8601",
            details={"field": "timestamp", "value": timestamp},
        ) from exc


def _normalize_config_event_details(
    event_type: str,
    details: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if event_type == "resolve":
        return None
    if event_type == "export":
        return {"diagnostic_only": True}
    if event_type == "validation_failure":
        detail_location = None
        issue_count = 0
        if details:
            detail_location = details.get("location")
            issue_count = details.get("issue_count", 0)
        return {"location": detail_location, "issue_count": int(issue_count)}
    return dict(details) if details else None


def build_plugin_governance_event(
    *,
    decision: str,
    identifier: str,
    provider: str | None,
    source: str,
    trusted: bool,
    actor: str,
    reason_code: str | None = None,
    reason: str | None = None,
    invocation_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct and validate a plugin-governance event envelope."""
    if decision not in PLUGIN_GOVERNANCE_DECISIONS:
        raise ValidationError(
            "unsupported governance decision",
            details={"decision": decision, "allowed": list(PLUGIN_GOVERNANCE_DECISIONS)},
        )

    context = get_logging_context()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "event_name": "plugin.registration.decision",
        "decision": decision,
        "identifier": identifier,
        "provider": provider,
        "source": source,
        "trusted": bool(trusted),
        "actor": actor,
        "reason_code": reason_code,
        "reason": reason,
        "timestamp": _iso_timestamp(),
        "invocation_id": invocation_id,
        "request_id": context.get("request_id"),
        "tenant_id": context.get("tenant_id"),
        "plugin_identifier": context.get("plugin_identifier") or identifier,
        "details": dict(details) if details else None,
    }
    validate_governance_event(payload)
    return payload


def build_config_governance_event(
    *,
    event_type: str,
    profile_id: str,
    config_schema_version: str,
    strict: bool,
    source_count: int,
    validation_issue_count: int,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct and validate a config-governance lifecycle event envelope."""
    if event_type not in CONFIG_GOVERNANCE_EVENT_TYPES:
        raise ValidationError(
            "unsupported config governance event type",
            details={"event_type": event_type, "allowed": list(CONFIG_GOVERNANCE_EVENT_TYPES)},
        )

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "event_name": "config.lifecycle",
        "event_type": event_type,
        "profile_id": profile_id,
        "config_schema_version": config_schema_version,
        "strict": strict,
        "source_count": source_count,
        "validation_issue_count": validation_issue_count,
        "timestamp": _iso_timestamp(),
        "details": _normalize_config_event_details(event_type, details),
    }
    validate_config_governance_event(payload)
    return payload


def emit_plugin_governance_event(
    *,
    decision: str,
    identifier: str,
    provider: str | None,
    source: str,
    trusted: bool,
    actor: str,
    reason_code: str | None = None,
    reason: str | None = None,
    invocation_id: str | None = None,
    details: Mapping[str, Any] | None = None,
    logger_name: str = "calibrated_explanations.governance.plugins",
) -> dict[str, Any]:
    """Emit structured governance event for a plugin decision path."""
    payload = build_plugin_governance_event(
        decision=decision,
        identifier=identifier,
        provider=provider,
        source=source,
        trusted=trusted,
        actor=actor,
        reason_code=reason_code,
        reason=reason,
        invocation_id=invocation_id,
        details=details,
    )
    logger = logging.getLogger(logger_name)
    ensure_logging_context_filter(logger_name)
    with logging_context(plugin_identifier=identifier):
        logger.info(
            "Plugin governance decision event emitted",
            extra=payload,
        )
    return payload


def emit_config_governance_event(
    *,
    event_type: str,
    profile_id: str,
    config_schema_version: str,
    strict: bool,
    source_count: int,
    validation_issue_count: int,
    details: Mapping[str, Any] | None = None,
    logger_name: str = "calibrated_explanations.governance.config",
) -> dict[str, Any]:
    """Emit structured governance event for a config lifecycle path."""
    payload = build_config_governance_event(
        event_type=event_type,
        profile_id=profile_id,
        config_schema_version=config_schema_version,
        strict=strict,
        source_count=source_count,
        validation_issue_count=validation_issue_count,
        details=details,
    )
    logger = logging.getLogger(logger_name)
    ensure_logging_context_filter(logger_name)
    logger.info(
        "Config governance lifecycle event emitted",
        extra=payload,
    )
    return payload


__all__ = [
    "CONFIG_GOVERNANCE_EVENT_TYPES",
    "PLUGIN_GOVERNANCE_DECISIONS",
    "build_config_governance_event",
    "build_plugin_governance_event",
    "emit_config_governance_event",
    "emit_plugin_governance_event",
    "validate_config_governance_event",
    "validate_governance_event",
]
