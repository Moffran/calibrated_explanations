from calibrated_explanations.explanations.explanation import FactualExplanation


def test_factual_build_rules_payload_minimal():
    class DummyExplainer:
        mode = "regression"
        class_labels = None

        def is_multiclass(self):
            return False

        feature_names = ["f0"]
        num_features = 1
        categorical_features = []

    inst = object.__new__(FactualExplanation)
    inst.get_explainer = lambda: DummyExplainer()
    inst.is_probabilistic = lambda: False
    inst.normalize_threshold_value = lambda: None

    inst.prediction = {"predict": 0.6, "low": 0.55, "high": 0.65, "classes": 0}

    rules = {
        "rule": ["f0 < 1"],
        "feature": [0],
        "weight": [0.1],
        "weight_low": [0.05],
        "weight_high": [0.15],
        "predict": [0.6],
        "predict_low": [0.55],
        "predict_high": [0.65],
        "value": ["1.23"],
        "feature_value": [1.23],
        "base_predict": [0.3],
        "base_predict_low": [0.25],
        "base_predict_high": [0.35],
    }

    class DummyContainer:
        low_high_percentiles = (5, 95)

    inst.calibrated_explanations = DummyContainer()
    inst.y_threshold = None
    inst.get_rules = lambda: rules
    payload = inst.build_rules_payload()
    assert payload["core"]["kind"] == "factual"
    assert "feature_rules" in payload["core"]
    assert len(payload["metadata"]["feature_rules"]) == 1
