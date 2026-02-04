"""Serialization helpers for Explanation schema v1 (internal).

Round-trip domain model <-> JSON using a stable envelope aligned to ADR-005.
Schema validation is optional to avoid hard dependency; when `jsonschema`
is installed, `validate=True` will verify the payload against the v1 schema.

Part of ADR-001: Core Decomposition Boundaries (Stage 1c).
Note: Schema validation has been moved to calibrated_explanations.schema.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping

from .explanations import Explanation, FeatureRule
from .schema import (
    validate_payload as _schema_validate_payload,  # noqa: F401 - re-exported under alias
)
from .utils.exceptions import ValidationError


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
                "feature": [int(x) for x in r.feature]
                if isinstance(r.feature, (list, tuple))
                else int(r.feature),
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

    # Ensure schema-required keys are present on exported prediction objects.
    # The schema requires `predict`, `low`, and `high` keys; when the domain
    # model provides only `predict` (common for classification), mirror the
    # `predict` value to both `low` and `high` so the payload satisfies the
    # structural validator and the invariant checker.
    pred = payload.get("prediction")
    if isinstance(pred, dict) and "predict" in pred:
        p = pred.get("predict")
        if "low" not in pred:
            pred["low"] = list(p) if isinstance(p, (list, tuple)) else p
        if "high" not in pred:
            pred["high"] = list(p) if isinstance(p, (list, tuple)) else p

    _validate_invariants(payload)
    try:
        validate_payload(payload)
    except Exception as exc:  # adr002_allow
        from .utils.exceptions import ValidationError

        raise ValidationError(
            "Serialization failed schema validation",
            details={"error": str(exc)},
        ) from exc
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

        with contextlib.suppress(TypeError, ValueError):
            # Handle scalar numeric values
            if (
                isinstance(predict, (int, float))
                and isinstance(low, (int, float))
                and isinstance(high, (int, float))
            ):
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

            # Handle vector-valued predictions (lists/tuples)
            if isinstance(predict, (list, tuple)):
                # Expect low and high to be lists/tuples of same length
                if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)):
                    raise ValidationError(
                        f"{context}: vector prediction requires vector low/high arrays",
                        details={"predict": predict, "low": low, "high": high},
                    )
                if not (len(predict) == len(low) == len(high)):
                    raise ValidationError(
                        f"{context}: vector prediction/interval length mismatch",
                        details={
                            "len_predict": len(predict),
                            "len_low": len(low),
                            "len_high": len(high),
                        },
                    )
                for j, (p, low_v, high_v) in enumerate(zip(predict, low, high, strict=False)):
                    if not (
                        isinstance(p, (int, float))
                        and isinstance(low_v, (int, float))
                        and isinstance(high_v, (int, float))
                    ):
                        raise ValidationError(
                            f"{context}[{j}]: entries must be numeric",
                            details={"p": p, "low": low_v, "high": high_v},
                        )
                    if not low_v <= high_v:
                        raise ValidationError(
                            f"{context}[{j}]: interval invariant violated (low > high)",
                            details={"low": low_v, "high": high_v, "index": j},
                        )
                    epsilon = 1e-9
                    if not (low_v - epsilon <= p <= high_v + epsilon):
                        raise ValidationError(
                            f"{context}[{j}]: prediction invariant violated (predict not in [low, high])",
                            details={"predict": p, "low": low_v, "high": high_v, "index": j},
                        )

    check(payload.get("prediction"), "Global prediction")
    for i, rule in enumerate(payload.get("rules", []) or []):
        check(rule.get("rule_prediction"), f"Rule {i} prediction")
        check(rule.get("instance_prediction"), f"Rule {i} instance prediction")


def from_json(obj: Mapping[str, Any]) -> Explanation:
    """Deserialize JSON payload (dict) to domain model Explanation."""
    rules = [
        FeatureRule(
            feature=(
                [int(x) for x in r.get("feature")]
                if isinstance(r.get("feature"), (list, tuple))
                else int(r.get("feature", i))
            ),
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
