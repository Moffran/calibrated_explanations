from calibrated_explanations.explanations.models import Explanation, FeatureRule
from calibrated_explanations.serialization import to_json, from_json


def test_serialization_roundtrip_scalar_prediction():
    fr = FeatureRule(
        feature=0, rule="x>0", rule_weight={"predict": 1}, rule_prediction={"predict": 1}
    )
    exp = Explanation(
        task="t", index=0, explanation_type="factual", prediction={"predict": 1}, rules=[fr]
    )
    payload = to_json(exp, include_version=True)
    assert payload["task"] == "t"
    # from_json should accept the payload-like mapping
    back = from_json(payload)
    assert back.task == "t"
