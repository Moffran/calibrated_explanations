import pytest

from calibrated_explanations import serialization
from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.explanations import Explanation, FeatureRule


def make_exp(prediction=None, rules=None):
    return Explanation(
        task="t",
        index=0,
        explanation_type="factual",
        prediction=prediction or {},
        rules=rules or [],
    )


def test_scalar_low_gt_high_raises():
    exp = make_exp({"predict": 1, "low": 2, "high": 0})
    with pytest.raises(ValidationError):
        serialization.to_json(exp)


def test_vector_requires_vector_low_high():
    exp = make_exp({"predict": [1, 2], "low": 0, "high": [0, 2]})
    with pytest.raises(ValidationError):
        serialization.to_json(exp)


def test_vector_length_mismatch_raises():
    exp = make_exp({"predict": [1, 2], "low": [0], "high": [2, 3]})
    with pytest.raises(ValidationError):
        serialization.to_json(exp)


def test_vector_non_numeric_entries_raises():
    exp = make_exp({"predict": [1, "x"], "low": [0, 0], "high": [2, 2]})
    with pytest.raises(ValidationError):
        serialization.to_json(exp)


def test_vector_low_gt_high_at_index_raises():
    exp = make_exp({"predict": [1, 2], "low": [0, 3], "high": [2, 4]})
    with pytest.raises(ValidationError):
        serialization.to_json(exp)


def test_scalar_predict_outside_tolerance_raises():
    exp = make_exp({"predict": 1.1, "low": 0.0, "high": 1.0})
    with pytest.raises(ValidationError):
        serialization.to_json(exp)


def test_rule_prediction_with_missing_bounds_is_tolerated():
    rule = FeatureRule(
        feature=0,
        rule="x <= 1",
        rule_weight={"predict": 0.5},
        rule_prediction={"predict": 1.0},
        instance_prediction=None,
        feature_value=0.5,
    )
    exp = make_exp({"predict": 1.0, "low": 0.0, "high": 2.0}, [rule])
    payload = serialization.to_json(exp)
    assert payload["rules"][0]["rule_prediction"]["predict"] == 1.0


def test_valid_vector_prediction_interval_serializes():
    exp = make_exp({"predict": [1.0, 2.0], "low": [0.0, 1.5], "high": [1.5, 2.5]})
    payload = serialization.to_json(exp)
    assert payload["prediction"]["predict"] == [1.0, 2.0]
