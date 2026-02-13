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




def test_rule_and_instance_prediction_checks():
    # Attach problematic predictions to a rule and its instance_prediction
    fr = FeatureRule(
        feature=0,
        rule="r",
        rule_weight={},
        rule_prediction={"predict": 5, "low": 0, "high": 1},
        instance_prediction={"predict": [1, 2], "low": [0], "high": [1, 2]},
    )
    exp = make_exp(prediction={}, rules=[fr])
    with pytest.raises(ValidationError):
        serialization.to_json(exp)
