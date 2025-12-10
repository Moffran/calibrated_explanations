"""Serialization helpers for Explanation schema v1 (internal).

Round-trip domain model <-> JSON using a stable envelope aligned to ADR-005.
Schema validation is optional to avoid hard dependency; when `jsonschema`
is installed, `validate=True` will verify the payload against the v1 schema.

Part of ADR-001: Core Decomposition Boundaries (Stage 1c).
Note: Schema validation has been moved to calibrated_explanations.schema.
"""

from __future__ import annotations

from typing import Any, Mapping

from .core.exceptions import ValidationError
from .explanations import Explanation, FeatureRule
from .schema import (
    validate_payload as _schema_validate_payload,  # noqa: F401 - re-exported under alias
)


def to_json(exp: Explanation, *, include_version: bool = True) -> dict[str, Any]:
    """Serialize a domain model Explanation to JSON payload (dict).

    The result follows schema v1 fields; additional metadata/provenance are passed through.
    """
    payload: dict[str, Any] = {
        "task": exp.task,
        "index": exp.index,
        "explanation_type": exp.explanation_type,
        "prediction": dict(exp.prediction),
        "rules": [
            {
                "feature": int(r.feature),
                "rule": r.rule,
                "rule_weight": dict(r.rule_weight),
                "rule_prediction": dict(r.rule_prediction),
                "instance_prediction": (
                    dict(r.instance_prediction) if r.instance_prediction else None
                ),
                "feature_value": r.feature_value,
                "is_conjunctive": bool(r.is_conjunctive),
                "value_str": r.value_str,
                "bin_index": r.bin_index,
            }
            for r in exp.rules
        ],
        "provenance": (dict(exp.provenance) if exp.provenance else None),
        "metadata": (dict(exp.metadata) if exp.metadata else None),
    }
    if include_version:
        payload["schema_version"] = "1.0.0"

    _validate_invariants(payload)
    return payload


def _validate_invariants(payload: dict[str, Any]) -> None:
    """Enforce low <= predict <= high invariant on exported payload."""

    def check(d: dict[str, Any] | None, context: str) -> None:
        if not d:
            return
        predict = d.get("predict")
        low = d.get("low")
        high = d.get("high")
        if predict is None or low is None or high is None:
            return

        try:
            # Handle scalar values (common case)
            if isinstance(predict, (int, float)) and isinstance(low, (int, float)) and isinstance(high, (int, float)):
                if not low <= high:
                    raise ValidationError(
                        f"{context}: interval invariant violated (low > high)",
                        details={"low": low, "high": high},
                    )
                # Use small epsilon for float comparison if needed, but strict for now
                if not (low <= predict <= high):
                    # Allow small floating point tolerance
                    epsilon = 1e-9
                    if not (low - epsilon <= predict <= high + epsilon):
                        raise ValidationError(
                            f"{context}: prediction invariant violated (predict not in [low, high])",
                            details={"predict": predict, "low": low, "high": high},
                        )
        except (TypeError, ValueError):
            # Skip validation for non-numeric types (e.g. class labels in predict)
            pass

    check(payload.get("prediction"), "Global prediction")
    for i, rule in enumerate(payload.get("rules", []) or []):
        check(rule.get("rule_prediction"), f"Rule {i} prediction")
        check(rule.get("instance_prediction"), f"Rule {i} instance prediction")



def from_json(obj: Mapping[str, Any]) -> Explanation:
    """Deserialize JSON payload (dict) to domain model Explanation."""
    rules = [
        FeatureRule(
            feature=int(r.get("feature", i)),
            rule=str(r.get("rule", "")),
            rule_weight=dict(r.get("rule_weight", {})),
            rule_prediction=dict(r.get("rule_prediction", {})),
            instance_prediction=r.get("instance_prediction"),
            feature_value=r.get("feature_value"),
            is_conjunctive=bool(r.get("is_conjunctive", False)),
            value_str=r.get("value_str"),
            bin_index=r.get("bin_index"),
        )
        for i, r in enumerate(obj.get("rules", []))
    ]
    return Explanation(
        task=str(obj.get("task", "unknown")),
        index=int(obj.get("index", 0)),
        explanation_type=str(obj.get("explanation_type", "factual")),
        prediction=dict(obj.get("prediction", {})),
        rules=rules,
        provenance=obj.get("provenance"),
        metadata=obj.get("metadata"),
    )


def validate_payload(obj: Mapping[str, Any]) -> None:
    """Validate a JSON payload against schema v1 if validator is available.

    DEPRECATED: Use calibrated_explanations.schema.validate_payload instead.
    """
    # Delegate to the schema module's validator (kept as a compatibility wrapper)
    return _schema_validate_payload(obj)


__all__ = ["to_json", "from_json", "validate_payload"]
