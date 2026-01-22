# tests/unit/explanations/test_alternative_explanation.py
import pytest
import numpy as np
from unittest.mock import Mock
from calibrated_explanations.explanations.explanation import AlternativeExplanation


class TestAlternativeExplanation:
    @pytest.fixture
    def explainer(self):
        explainer = Mock()
        explainer.categorical_features = []
        explainer.features = [0, 1]
        explainer.feature_names = ["f0", "f1"]
        explainer.categorical_labels = {0: ["0"], 1: ["1"]}
        explainer.num_classes = 2
        explainer.classes = [0, 1]
        explainer.discretizer = None
        explainer.mode = "classification"
        # Fix for get_explainer iterating on Mock
        explainer.get_explainer.return_value = explainer
        # Mock is_multiclass to return False for simpler logic
        explainer.is_multiclass = Mock(return_value=False)
        # Mock discretize to return a list containing one element (the discretized instance)
        # This fixes 'Mock object is not subscriptable' when [0] is accessed
        explainer.discretize = Mock(return_value=[[1, 2]])
        # Mock x_cal for numerical feature handling
        explainer.x_cal = np.array([[1, 2], [3, 4]])
        # Mock rule_boundaries
        explainer.rule_boundaries = Mock(return_value=[[0.5, 1.5], [0.5, 1.5]])
        return explainer

    @pytest.fixture
    def explanation(self, explainer):
        index = 0
        x = np.array([1, 2])
        # binned should be subscriptable by index
        binned = {
            "f0": [1],
            "f1": [2],
            "predict": [[[0.5, 0.5], [0.5, 0.5]]],
            "low": [[[0.4, 0.4], [0.4, 0.4]]],
            "high": [[[0.6, 0.6], [0.6, 0.6]]],
            "rule_values": [[[[[1]], [[2]]], [[[3]], [[4]]]]],
        }
        # features arguments must be subscriptable by index [index]
        feature_weights = {
            "predict": [[0.1]],
            "predict_low": [[0.0]],
            "predict_high": [[0.2]],
            "weight": [[0.1]],
            "weight_low": [[0.0]],
            "weight_high": [[0.2]],
            "value": [[1]],
            "rule": [["r1"]],
            "feature": [[0]],
            "feature_value": [[None]],
            "sampled_values": [[[]]],
            "is_conjunctive": [[False]],
        }
        feature_predict = feature_weights.copy()

        prediction = {"predict": [0.5], "high": [0.6], "low": [0.4], "classes": [0, 1]}

        instance = AlternativeExplanation(
            calibrated_explanations=explainer,
            index=index,
            x=x,
            binned=binned,
            feature_weights=feature_weights,
            feature_predict=feature_predict,
            prediction=prediction,
        )

        # Mock methods to avoid SideEffects
        instance.is_regression = Mock(return_value=False)
        instance.is_probabilistic = Mock(return_value=True)
        instance.has_conjunctive_rules = False

        return instance

    def setup_test_rules(self, instance, rules_data):
        # Helper to set mocked rules
        # Add required base keys if not present
        if "base_predict_low" not in rules_data:
            rules_data["base_predict_low"] = instance.prediction["low"]
        if "base_predict_high" not in rules_data:
            rules_data["base_predict_high"] = instance.prediction["high"]
        if "classes" not in rules_data:
            rules_data["classes"] = instance.calibrated_explanations.classes

        instance.get_rules = Mock(return_value=rules_data)

    def test_super_explanations(self, explanation):
        explanation.prediction = {"predict": 0.3, "low": 0.2, "high": 0.4, "classes": [0, 1]}

        rules = {
            "predict": [0.1, 0.4, 0.2],
            "predict_low": [0.05, 0.35, 0.15],
            "predict_high": [0.15, 0.45, 0.25],
            "weight": [0, 0, 0],
            "weight_low": [0, 0, 0],
            "weight_high": [0, 0, 0],
            "value": [1, 2, 3],
            "rule": ["r1", "r2", "r3"],
            "feature": [0, 1, 0],
            "sampled_values": [[], [1], []],
            "feature_value": [None, None, None],
            "is_conjunctive": [False, False, False],
            "classes": [0, 1],
            "base_predict_low": 0.2,
            "base_predict_high": 0.4,
        }
        self.setup_test_rules(explanation, rules)

        explanation.super_explanations()

        filtered = explanation.rules
        assert filtered["predict"] == [0.1, 0.2]

    def test_counter_explanations(self, explanation):
        explanation.prediction = {"predict": 0.3, "low": 0.2, "high": 0.4, "classes": [0, 1]}

        rules = {
            "predict": [0.2, 0.4, 0.6],
            "predict_low": [0.1, 0.3, 0.5],
            "predict_high": [0.3, 0.5, 0.7],
            "weight": [0, 0, 0],
            "weight_low": [0, 0, 0],
            "weight_high": [0, 0, 0],
            "value": [1, 2, 3],
            "rule": ["r1", "r2", "r3"],
            "feature": [0, 1, 0],
            "sampled_values": [[], [1], []],
            "feature_value": [None, None, None],
            "is_conjunctive": [False, False, False],
            "classes": [0, 1],
            "base_predict_low": 0.2,
            "base_predict_high": 0.4,
        }
        self.setup_test_rules(explanation, rules)

        explanation.counter_explanations()

        filtered = explanation.rules
        assert filtered["predict"] == [0.6]

    def test_semi_explanations(self, explanation):
        explanation.prediction = {"predict": 0.3, "low": 0.2, "high": 0.4, "classes": [0, 1]}

        rules = {
            "predict": [0.1, 0.4, 0.6],
            "predict_low": [0.05, 0.35, 0.55],
            "predict_high": [0.15, 0.45, 0.65],
            "weight": [0, 0, 0],
            "weight_low": [0, 0, 0],
            "weight_high": [0, 0, 0],
            "value": [1, 2, 3],
            "rule": ["r1", "r2", "r3"],
            "feature": [0, 1, 0],
            "sampled_values": [[], [1], []],
            "feature_value": [None, None, None],
            "is_conjunctive": [False, False, False],
            "classes": [0, 1],
            "base_predict_low": 0.2,
            "base_predict_high": 0.4,
        }
        self.setup_test_rules(explanation, rules)

        explanation.semi_explanations()

        filtered = explanation.rules
        assert filtered["predict"] == [0.4]

    def test_ensured_explanations(self, explanation):
        explanation.prediction = {"predict": 0.5, "low": 0.4, "high": 0.6, "classes": [0, 1]}

        rules = {
            "predict": [0.4, 0.5],
            "predict_low": [0.35, 0.3],
            "predict_high": [0.45, 0.7],
            "weight": [0, 0],
            "weight_low": [0, 0],
            "weight_high": [0, 0],
            "value": [1, 2],
            "rule": ["r1", "r2"],
            "feature": [0, 1],
            "sampled_values": [[], [1]],
            "feature_value": [None, None],
            "is_conjunctive": [False, False],
            "classes": [0, 1],
            "base_predict_low": 0.4,
            "base_predict_high": 0.6,
        }
        self.setup_test_rules(explanation, rules)

        explanation.ensured_explanations()

        filtered = explanation.rules
        assert filtered["predict_low"] == [0.35]
        assert filtered["predict_high"] == [0.45]
