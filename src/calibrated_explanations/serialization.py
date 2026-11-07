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
from .explanations.models import CalibrationDescriptor, ModelDescriptor
from .schema import (
    validate_payload as _schema_validate_payload,
)
from .utils.exceptions import ValidationError


def to_json(exp: Explanation, *, include_version: bool = True) -> dict[str, Any]:
    """Serialize a domain model Explanation to JSON payload (dict).

    The result follows schema v1 fields; additional metadata/provenance are passed through.
    """
    # Build metadata output: merge calibration_metadata and model_metadata into the dict
    _meta_out: dict[str, Any] = dict(exp.metadata) if isinstance(exp.metadata, Mapping) else {}
    if exp.calibration_metadata is not None:
        _meta_out["calibration_metadata"] = {"method": exp.calibration_metadata.method}
    if exp.model_metadata is not None:
        _meta_out["model_metadata"] = {"type": exp.model_metadata.type}

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
        "metadata": (_meta_out if _meta_out else None),
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
        _schema_validate_payload(payload)
    except Exception as exc:  # adr002_allow
        from .utils.exceptions import ValidationError

        raise ValidationError(
            "Serialization failed schema validation",
            details={"error": str(exc)},
        ) from exc
    from .explanations.explanations import jsonify_value

    return jsonify_value(payload)


def _validate_invariants(payload: dict[str, Any]) -> None:
    """Enforce low <= predict <= high invariant on exported payload."""
    from .utils import median_containment_sides

    metadata = payload.get("metadata")
    context = metadata.get("narrative_context", {}) if isinstance(metadata, Mapping) else {}
    sides = (True, True)
    if (
        payload.get("task") == "regression"
        and isinstance(context, Mapping)
        and context.get("problem_type") == "regression"
        and context.get("low_high_percentiles") is not None
    ):
        sides = median_containment_sides(context["low_high_percentiles"])

    def numeric(value: Any) -> Any:
        if isinstance(value, str) and value in ("-inf", "inf", "+inf"):
            return float(value)
        return value

    def check(d: dict[str, Any] | None, label: str) -> None:
        if not d:
            return
        predict = numeric(d.get("predict"))
        low = numeric(d.get("low"))
        high = numeric(d.get("high"))
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
                        f"{label}: interval invariant violated (low > high)",
                        details={"low": low, "high": high},
                    )
                # Use small epsilon for float comparison if needed, but strict for now
                epsilon = 1e-9
                above_low = not sides[0] or low - epsilon <= predict
                below_high = not sides[1] or predict <= high + epsilon
                if not (above_low and below_high):
                    raise ValidationError(
                        f"{label}: prediction invariant violated (predict not in [low, high])",
                        details={"predict": predict, "low": low, "high": high},
                    )

            # Handle vector-valued predictions (lists/tuples)
            if isinstance(predict, (list, tuple)):
                # Expect low and high to be lists/tuples of same length
                if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)):
                    raise ValidationError(
                        f"{label}: vector prediction requires vector low/high arrays",
                        details={"predict": predict, "low": low, "high": high},
                    )
                if not (len(predict) == len(low) == len(high)):
                    raise ValidationError(
                        f"{label}: vector prediction/interval length mismatch",
                        details={
                            "len_predict": len(predict),
                            "len_low": len(low),
                            "len_high": len(high),
                        },
                    )
                for j, (p, low_v, high_v) in enumerate(zip(predict, low, high, strict=False)):
                    p, low_v, high_v = numeric(p), numeric(low_v), numeric(high_v)
                    if not (
                        isinstance(p, (int, float))
                        and isinstance(low_v, (int, float))
                        and isinstance(high_v, (int, float))
                    ):
                        raise ValidationError(
                            f"{label}[{j}]: entries must be numeric",
                            details={"p": p, "low": low_v, "high": high_v},
                        )
                    if not low_v <= high_v:
                        raise ValidationError(
                            f"{label}[{j}]: interval invariant violated (low > high)",
                            details={"low": low_v, "high": high_v, "index": j},
                        )
                    epsilon = 1e-9
                    above_low = not sides[0] or low_v - epsilon <= p
                    below_high = not sides[1] or p <= high_v + epsilon
                    if not (above_low and below_high):
                        raise ValidationError(
                            f"{label}[{j}]: prediction invariant violated (predict not in [low, high])",
                            details={"predict": p, "low": low_v, "high": high_v, "index": j},
                        )

    check(payload.get("prediction"), "Global prediction")
    for i, rule in enumerate(payload.get("rules", []) or []):
        check(rule.get("rule_prediction"), f"Rule {i} prediction")
        check(rule.get("instance_prediction"), f"Rule {i} instance prediction")


def from_json(obj: Mapping[str, Any]) -> Explanation:
    """Deserialize JSON payload (dict) to domain model Explanation."""

    def restore_interval(raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw

        def restore(value: Any) -> Any:
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, str) and value in ("-inf", "inf", "+inf"):
                return float(value)
            return value

        return {
            key: restore(value) if key in ("predict", "low", "high") else value
            for key, value in raw.items()
        }

    rules = [
        FeatureRule(
            feature=(
                [int(x) for x in r.get("feature")]
                if isinstance(r.get("feature"), (list, tuple))
                else int(r.get("feature", i))
            ),
            rule=str(r.get("rule", "")),
            rule_weight=restore_interval(r.get("rule_weight", {})),
            rule_prediction=restore_interval(r.get("rule_prediction", {})),
            instance_prediction=restore_interval(r.get("instance_prediction")),
            feature_value=r.get("feature_value"),
            is_conjunctive=bool(r.get("is_conjunctive", False)),
            value_str=r.get("value_str"),
            bin_index=r.get("bin_index"),
        )
        for i, r in enumerate(obj.get("rules", []))
    ]
    raw_meta = obj.get("metadata")
    if isinstance(raw_meta, Mapping):
        raw_meta = dict(raw_meta)
        context = raw_meta.get("narrative_context")
        if isinstance(context, Mapping):
            context = dict(context)
            percentiles = context.get("low_high_percentiles")
            if isinstance(percentiles, (list, tuple)):
                context["low_high_percentiles"] = [
                    float(value)
                    if isinstance(value, str) and value in ("-inf", "inf", "+inf")
                    else value
                    for value in percentiles
                ]
            raw_meta["narrative_context"] = context
    cal_descriptor = None
    mod_descriptor = None
    if isinstance(raw_meta, Mapping):
        raw_cal = raw_meta.get("calibration_metadata")
        if isinstance(raw_cal, dict):
            try:
                cal_descriptor = CalibrationDescriptor(**raw_cal)
            except TypeError:
                cal_descriptor = None
        raw_mod = raw_meta.get("model_metadata")
        if isinstance(raw_mod, dict):
            try:
                mod_descriptor = ModelDescriptor(**raw_mod)
            except TypeError:
                mod_descriptor = None
    return Explanation(
        task=str(obj.get("task", "unknown")),
        index=int(obj.get("index", 0)),
        explanation_type=str(obj.get("explanation_type", "factual")),
        prediction=restore_interval(obj.get("prediction", {})),
        rules=rules,
        provenance=obj.get("provenance"),
        metadata=raw_meta,
        calibration_metadata=cal_descriptor,
        model_metadata=mod_descriptor,
    )


__all__ = ["to_json", "from_json"]
