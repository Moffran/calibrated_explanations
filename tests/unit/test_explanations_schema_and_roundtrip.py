"""Tests for schema acceptance, validator fallback, conjunctive round-trip, and vector invariants.
"""
from __future__ import annotations

import types

import pytest

from calibrated_explanations import serialization
from calibrated_explanations.explanations import Explanation, FeatureRule
from calibrated_explanations.utils.exceptions import ValidationError


def test_minimal_validator_rejects_missing_predict_when_jsonschema_absent(monkeypatch):
    # Ensure the minimal built-in validator runs when jsonschema is not installed
    from calibrated_explanations import schema

    monkeypatch.setattr(schema.validation, "jsonschema", None)

    payload = {
        "task": "classification",
        "index": 1,
        "explanation_type": "factual",
        "prediction": {"low": 0.0, "high": 1.0},  # missing 'predict'
        "rules": [],
    }

    with pytest.raises(ValueError):
        serialization.validate_payload(payload)


def test_schema_accepts_fast_when_jsonschema_present(monkeypatch):
    # Fake jsonschema validator that asserts schema contains 'fast' in enum
    captured = {}

    def fake_validate(*, instance, schema):
        # Assert the loader returned a schema that accepts 'fast'
        assert (
            schema.get("properties", {}).get("explanation_type", {}).get("enum")
            and "fast" in schema["properties"]["explanation_type"]["enum"]
        )
        captured["validated_instance"] = instance

    fake_validator = types.SimpleNamespace(validate=fake_validate)

    from calibrated_explanations import schema

    monkeypatch.setattr(schema.validation, "jsonschema", fake_validator)

    def fake_schema_loader():
        return {
            "type": "object",
            "properties": {
                "explanation_type": {"enum": ["factual", "alternative", "fast"]},
                "prediction": {"type": "object"},
                "rules": {"type": "array"},
            },
        }

    monkeypatch.setattr(schema.validation, "_schema_json", fake_schema_loader)

    payload = {
        "task": "classification",
        "index": 2,
        "explanation_type": "fast",
        "prediction": {"predict": 0.5, "low": 0.4, "high": 0.6},
        "rules": [],
    }

    # Should not raise and should call our fake validator
    serialization.validate_payload(payload)
    assert captured.get("validated_instance") is payload


def test_conjunctive_feature_roundtrip():
    # Build an Explanation with a conjunctive rule (feature list)
    fr = FeatureRule(
        feature=[1, 2],
        rule="x1 > 0 and x2 < 0.5",
        rule_weight={"w": 0.1},
        rule_prediction={"score": 0.9},
        instance_prediction={"score": 0.85},
        is_conjunctive=True,
    )
    exp = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.9, "low": 0.8, "high": 1.0},
        rules=[fr],
    )

    payload = serialization.to_json(exp, include_version=False)
    assert isinstance(payload["rules"][0]["feature"], list)
    assert payload["rules"][0]["feature"] == [1, 2]

    restored = serialization.from_json(payload)
    assert isinstance(restored.rules[0].feature, list)
    assert restored.rules[0].feature == [1, 2]
    assert restored.rules[0].is_conjunctive is True


def test_vector_invariant_violation_raises():
    # Global prediction with vector where predict[0] < low[0] should fail
    fr = FeatureRule(
        feature=0,
        rule="x0 > 0",
        rule_weight={},
        rule_prediction={"score": [0.5, 0.5]},
        instance_prediction=None,
    )
    exp = Explanation(
        task="regression",
        index=1,
        explanation_type="factual",
        prediction={"predict": [0.5, 0.5], "low": [0.6, 0.4], "high": [1.0, 1.0]},
        rules=[fr],
    )

    with pytest.raises(ValidationError):
        serialization.to_json(exp, include_version=False)
