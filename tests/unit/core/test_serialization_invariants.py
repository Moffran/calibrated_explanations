"""Unit tests for serialization invariant enforcement.

Tests cover:
- Validation of low <= predict <= high invariant in to_json
- Validation of low <= high invariant in to_json
"""

from __future__ import annotations

import pytest
from calibrated_explanations import serialization
from calibrated_explanations.core.exceptions import ValidationError
from calibrated_explanations.explanations import Explanation, FeatureRule

@pytest.fixture
def base_explanation() -> Explanation:
    """Minimal valid explanation fixture."""
    return Explanation(
        task="regression",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.5, "low": 0.4, "high": 0.6},
        rules=[],
        provenance=None,
        metadata=None,
    )

def test_should_raise_validation_error_when_global_prediction_violates_low_high(
    base_explanation: Explanation,
) -> None:
    """Verify to_json raises ValidationError when global prediction has low > high."""
    # Arrange
    base_explanation.prediction = {"predict": 0.5, "low": 0.7, "high": 0.6}

    # Act & Assert
    with pytest.raises(ValidationError, match="Global prediction: interval invariant violated"):
        serialization.to_json(base_explanation)

def test_should_raise_validation_error_when_global_prediction_outside_interval(
    base_explanation: Explanation,
) -> None:
    """Verify to_json raises ValidationError when global prediction is outside [low, high]."""
    # Arrange
    base_explanation.prediction = {"predict": 0.3, "low": 0.4, "high": 0.6}

    # Act & Assert
    with pytest.raises(ValidationError, match="Global prediction: prediction invariant violated"):
        serialization.to_json(base_explanation)

def test_should_raise_validation_error_when_rule_prediction_violates_invariant(
    base_explanation: Explanation,
) -> None:
    """Verify to_json raises ValidationError when rule prediction violates invariant."""
    # Arrange
    rule = FeatureRule(
        feature=1,
        rule="x > 0",
        rule_weight={},
        rule_prediction={"predict": 0.5, "low": 0.7, "high": 0.6}, # Invalid
        instance_prediction=None,
        feature_value=1.0,
        is_conjunctive=False,
        value_str="1.0",
        bin_index=0,
    )
    base_explanation.rules = [rule]

    # Act & Assert
    with pytest.raises(ValidationError, match="Rule 0 prediction: interval invariant violated"):
        serialization.to_json(base_explanation)

def test_should_pass_when_invariants_are_satisfied(
    base_explanation: Explanation,
) -> None:
    """Verify to_json passes when all invariants are satisfied."""
    # Arrange
    # base_explanation is already valid

    # Act
    payload = serialization.to_json(base_explanation)

    # Assert
    assert payload["prediction"]["predict"] == 0.5
