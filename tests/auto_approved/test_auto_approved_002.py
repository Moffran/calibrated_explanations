from calibrated_explanations.explanations.models import Explanation, FeatureRule
from calibrated_explanations.serialization import to_json


def test_serialization_vector_prediction_handles_lists():
    # Vector-valued prediction without explicit low/high should be mirrored
    fr = FeatureRule(
        feature=[0, 1],
        rule="a",
        rule_weight={"predict": [0.1, 0.2]},
        rule_prediction={"predict": [0.1, 0.2]},
    )
    exp = Explanation(
        task="vec",
        index=0,
        explanation_type="factual",
        prediction={"predict": [0.1, 0.2]},
        rules=[fr],
    )
    payload = to_json(exp, include_version=False)
    # ensure low/high were added for vector predictions
    pred = payload.get("prediction")
    assert isinstance(pred.get("low"), list)
    assert isinstance(pred.get("high"), list)
