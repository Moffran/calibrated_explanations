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
    if jsonschema is None:  # pragma: no cover
        return
    schema = _schema_json()
    jsonschema.validate(instance=obj, schema=schema)  # type: ignore[attr-defined]


__all__ = ["validate_payload"]
