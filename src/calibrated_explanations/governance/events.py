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


def _schema_json() -> dict[str, Any]:  # pragma: no cover - tiny IO helper
    with (
        resources.files("calibrated_explanations.schemas")
        .joinpath("governance_event_schema_v1.json")
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
        jsonschema.validate(instance=dict(payload), schema=_schema_json())  # type: ignore[attr-defined]
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


__all__ = [
    "PLUGIN_GOVERNANCE_DECISIONS",
    "build_plugin_governance_event",
    "emit_plugin_governance_event",
    "validate_governance_event",
]
