import pytest

from calibrated_explanations.explanations.models import Explanation, FeatureRule
from calibrated_explanations.serialization import to_json, from_json
from calibrated_explanations.utils.exceptions import ValidationError


def test_to_json_with_valid_scalar_interval():
    exp = Explanation(
        task="t",
        index=0,
        explanation_type="factual",
        prediction={"predict": 1.0, "low": 0.0, "high": 2.0},
        rules=[],
    )
    payload = to_json(exp)
    assert payload["prediction"]["predict"] == 1.0


def test_to_json_raises_when_low_greater_than_high():
    exp = Explanation(
        task="t",
        index=0,
        explanation_type="factual",
        prediction={"predict": 1.0, "low": 2.0, "high": 1.0},
        rules=[],
    )
    with pytest.raises(ValidationError):
        to_json(exp)


def test_to_json_vector_length_mismatch_raises():
    fr = FeatureRule(feature=0, rule="r", rule_weight={}, rule_prediction={})
    exp = Explanation(
        task="t",
        index=1,
        explanation_type="factual",
        prediction={"predict": [1, 2], "low": [0], "high": [2, 3]},
        rules=[fr],
    )
    with pytest.raises(ValidationError):
        to_json(exp)


def test_from_json_roundtrip_preserves_basic_fields():
    payload = {
        "task": "t",
        "index": 2,
        "explanation_type": "factual",
        "prediction": {"predict": [1, 2], "low": [0, 1], "high": [1, 2]},
        "rules": [],
    }
    exp = from_json(payload)
    back = to_json(exp)
    assert back["task"] == "t"
