"""Unit tests for serialization module (explanation round-trip).

Tests cover:
- JSON serialization with/without schema version
- Deserialization with default population
- Validation against schema when jsonschema is available
"""

from __future__ import annotations

import types

import pytest

from calibrated_explanations import serialization
from calibrated_explanations.explanations.models import Explanation, FeatureRule


# Shared fixtures and test data
@pytest.fixture
def factual_rule_data() -> FeatureRule:
    """Fully populated FeatureRule for factual explanations."""
    return FeatureRule(
        feature=5,
        rule="x > 0.2",
        rule_weight={"weight": 1.2},
        rule_prediction={"score": 0.7},
        instance_prediction={"score": 0.6},
        feature_value=0.25,
        is_conjunctive=True,
        value_str="> 0.2",
        bin_index=3,
    )


@pytest.fixture
def alternative_rule_data() -> FeatureRule:
    """Fully populated FeatureRule for alternative explanations."""
    return FeatureRule(
        feature=4,
        rule="x4 > 0.5",
        rule_weight={"coef": 0.2},
        rule_prediction={"label": "cat", "score": 0.91},
        instance_prediction={"label": "cat", "score": 0.88},
        feature_value=0.7,
        is_conjunctive=False,
        value_str="0.7",
        bin_index=1,
    )


@pytest.fixture
def factual_explanation(factual_rule_data: FeatureRule) -> Explanation:
    """Fully populated factual explanation fixture."""
    return Explanation(
        task="classification",
        index=4,
        explanation_type="factual",
        prediction={"label": "yes"},
        rules=[factual_rule_data],
        provenance={"source": "unit"},
        metadata={"note": "demo"},
    )


@pytest.fixture
def alternative_explanation(alternative_rule_data: FeatureRule) -> Explanation:
    """Fully populated alternative explanation fixture."""
    return Explanation(
        task="classification",
        index=3,
        explanation_type="alternative",
        prediction={"label": "cat", "score": 0.87},
        rules=[alternative_rule_data],
        provenance={"source": "unit-test"},
        metadata={"explanation_id": "abc-123"},
    )


# Serialization tests
def test_should_exclude_schema_version_when_disabled(
    factual_explanation: Explanation,
) -> None:
    """Verify to_json excludes schema_version when include_version=False."""
    # Arrange: use factual_explanation fixture

    # Act
    payload = serialization.to_json(factual_explanation, include_version=False)

    # Assert
    assert "schema_version" not in payload
    assert payload["rules"][0]["feature"] == 5
    assert payload["rules"][0]["is_conjunctive"] is True


def test_should_include_schema_version_by_default(
    factual_explanation: Explanation,
) -> None:
    """Verify to_json includes schema_version by default."""
    # Arrange: use factual_explanation fixture

    # Act
    payload = serialization.to_json(factual_explanation)

    # Assert
    assert "schema_version" in payload


def test_should_preserve_explanation_fields_when_serializing(
    alternative_explanation: Explanation,
) -> None:
    """Verify to_json preserves all explanation fields."""
    # Arrange: use alternative_explanation fixture

    # Act
    payload = serialization.to_json(alternative_explanation, include_version=False)

    # Assert
    assert payload["task"] == "classification"
    assert payload["index"] == 3
    assert payload["explanation_type"] == "alternative"
    assert payload["prediction"] == {"label": "cat", "score": 0.87}
    assert payload["provenance"] == {"source": "unit-test"}
    assert payload["metadata"] == {"explanation_id": "abc-123"}


def test_should_preserve_rule_fields_when_serializing(
    factual_explanation: Explanation,
) -> None:
    """Verify to_json preserves all rule fields with correct types."""
    # Arrange: use factual_explanation fixture

    # Act
    payload = serialization.to_json(factual_explanation, include_version=False)

    # Assert
    rule = payload["rules"][0]
    assert rule["feature"] == 5
    assert rule["rule"] == "x > 0.2"
    assert rule["rule_weight"] == {"weight": 1.2}
    assert rule["rule_prediction"] == {"score": 0.7}
    assert rule["instance_prediction"] == {"score": 0.6}
    assert rule["feature_value"] == 0.25
    assert rule["is_conjunctive"] is True
    assert rule["value_str"] == "> 0.2"
    assert rule["bin_index"] == 3


# Deserialization tests
def test_should_populate_defaults_for_missing_optional_fields() -> None:
    """Verify from_json applies defaults for omitted optional fields."""
    # Arrange
    payload = {
        "task": "classification",
        "index": 9,
        "prediction": {"p": 0.4},
        "rules": [
            {
                # intentionally missing most optional fields to exercise defaults
                "rule": "value > 0",
            }
        ],
    }

    # Act
    explanation = serialization.from_json(payload)

    # Assert
    assert explanation.explanation_type == "factual"
    assert explanation.rules[0].feature == 0  # defaulted to enumerate index
    assert explanation.rules[0].rule_weight == {}
    assert explanation.rules[0].instance_prediction is None
    assert explanation.rules[0].value_str is None


def test_should_populate_defaults_for_minimal_payload() -> None:
    """Verify from_json handles minimal payload with mostly defaults."""
    # Arrange
    payload = {
        "task": "classification",
        "rules": [
            {
                "rule": "x0 > 0",
                "rule_weight": {"coef": 0.1},
                "rule_prediction": {"label": "dog"},
            }
        ],
    }

    # Act
    exp = serialization.from_json(payload)

    # Assert
    assert exp.index == 0
    assert exp.explanation_type == "factual"
    assert exp.prediction == {}
    assert exp.rules[0].feature == 0
    assert exp.rules[0].is_conjunctive is False


def test_should_round_trip_explanation_when_serialized_then_deserialized(
    factual_explanation: Explanation,
) -> None:
    """Verify round-trip preserves data: Explanation -> JSON -> Explanation."""
    # Arrange: use factual_explanation fixture

    # Act
    payload = serialization.to_json(factual_explanation, include_version=False)
    restored = serialization.from_json(payload)

    # Assert
    assert restored.task == factual_explanation.task
    assert restored.index == factual_explanation.index
    assert restored.explanation_type == factual_explanation.explanation_type
    assert restored.prediction == factual_explanation.prediction
    assert len(restored.rules) == len(factual_explanation.rules)
    assert restored.rules[0].feature == factual_explanation.rules[0].feature
    assert (
        restored.rules[0].is_conjunctive
        == factual_explanation.rules[0].is_conjunctive
    )


# Validation tests
def test_should_use_jsonschema_when_available_for_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify validate_payload calls jsonschema.validate when available."""
    # Arrange
    calls: dict[str, object] = {}

    def fake_validate(*, instance, schema):
        calls["instance"] = instance
        calls["schema"] = schema

    class FakeJSONSchema:
        """Mock jsonschema module."""

        @staticmethod
        def validate(*, instance, schema):
            """Mock validate method."""
            fake_validate(instance=instance, schema=schema)

    monkeypatch.setattr(serialization, "jsonschema", FakeJSONSchema)
    monkeypatch.setattr(serialization, "_schema_json", lambda: {"title": "fake"})

    payload = {
        "task": "classification",
        "index": 1,
        "explanation_type": "factual",
        "prediction": {"label": "cat"},
        "rules": [],
    }

    # Act
    serialization.validate_payload(payload)

    # Assert
    assert calls["instance"] == payload
    assert calls["schema"] == {"title": "fake"}


def test_should_invoke_schema_loader_when_validating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify validate_payload loads schema via _schema_json."""
    # Arrange
    captured = {}

    def fake_schema_loader():
        captured["schema_loaded"] = True
        return {"type": "object"}

    def fake_validate(*, instance, schema):
        captured["instance"] = instance
        captured["schema"] = schema

    validator = types.SimpleNamespace(validate=fake_validate)

    monkeypatch.setattr(serialization, "jsonschema", validator)
    monkeypatch.setattr(serialization, "_schema_json", fake_schema_loader)

    payload = {"task": "regression"}

    # Act
    serialization.validate_payload(payload)

    # Assert
    assert captured == {
        "schema_loaded": True,
        "instance": payload,
        "schema": {"type": "object"},
    }
