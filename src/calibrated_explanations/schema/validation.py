"""Schema validation and loading helpers.

This module provides utilities for loading and validating explanation payloads
against the ADR-005 explanation schema v1. Schema validation is intentionally
kept focused on structural (JSON Schema) checks and is optional to avoid a hard
dependency on ``jsonschema``. Semantic invariants (for example, the
``low <= predict <= high`` interval invariant) are enforced at
serialization-time by the library and are not performed by this helper.

Part of ADR-001: Core Decomposition Boundaries (Stage 1c).
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Mapping

from ..utils.exceptions import ValidationError

try:  # optional validator
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None


def _schema_json() -> dict[str, Any]:  # pragma: no cover - tiny IO
    """Load the bundled explanation schema as a Python dictionary."""
    with (
        resources.files("calibrated_explanations.schemas")
        .joinpath("explanation_schema_v1.json")
        .open("r", encoding="utf-8") as f
    ):
        import json

        return json.load(f)


# Public alias for testing
schema_json = _schema_json


def validate_payload(obj: Mapping[str, Any]) -> None:
    """Validate a JSON payload against schema v1 if validator is available.

    Parameters
    ----------
    obj : Mapping[str, Any]
        The JSON payload to validate.

    Raises
    ------
    jsonschema.ValidationError
        If the payload does not conform to the schema and jsonschema is installed.
    """
    # If jsonschema is available, prefer full JSON Schema validation.
    if jsonschema is not None:
        schema = _schema_json()
        jsonschema.validate(instance=obj, schema=schema)  # type: ignore[attr-defined]
        return

    # Minimal built-in structural validation when jsonschema is not installed.
    # This enforces required keys and basic types to avoid silently accepting
    # malformed payloads in core-only installs.
    required = ["task", "index", "explanation_type", "prediction", "rules"]
    for key in required:
        if key not in obj:
            raise ValidationError(f"Missing required payload key: {key}", details={"key": key})

    if not isinstance(obj.get("task"), str):
        raise ValidationError("Field 'task' must be a string", details={"field": "task"})
    if not isinstance(obj.get("index"), int):
        raise ValidationError("Field 'index' must be an integer", details={"field": "index"})

    # explanation_type must be a string
    if not isinstance(obj.get("explanation_type"), str):
        raise ValidationError(
            "Field 'explanation_type' must be a string",
            details={"field": "explanation_type"},
        )

    # prediction must be an object with at least predict/low/high keys
    pred = obj.get("prediction")
    if not isinstance(pred, Mapping):
        raise ValidationError(
            "Field 'prediction' must be an object", details={"field": "prediction"}
        )
    for sub in ("predict", "low", "high"):
        if sub not in pred:
            raise ValidationError(
                f"prediction missing required key: {sub}",
                details={"field": "prediction", "missing_key": sub},
            )

    # rules must be a list of objects with required fields
    rules = obj.get("rules")
    if not isinstance(rules, list):
        raise ValidationError("Field 'rules' must be an array", details={"field": "rules"})
    for i, r in enumerate(rules):
        if not isinstance(r, Mapping):
            raise ValidationError(f"Rule {i} must be an object", details={"rule_index": i})
        for rk in ("feature", "rule", "rule_weight", "rule_prediction"):
            if rk not in r:
                raise ValidationError(
                    f"Rule {i} missing required key: {rk}",
                    details={"rule_index": i, "missing_key": rk},
                )
        # feature may be int or array of ints
        feat = r.get("feature")
        if not isinstance(feat, (int, list)):
            raise ValidationError(
                f"Rule {i} feature must be integer or list of integers",
                details={"rule_index": i, "field": "feature"},
            )
        if isinstance(feat, list) and not all(isinstance(x, int) for x in feat):
            raise ValidationError(
                f"Rule {i} feature list must contain only integers",
                details={"rule_index": i, "field": "feature"},
            )


__all__ = ["validate_payload"]
