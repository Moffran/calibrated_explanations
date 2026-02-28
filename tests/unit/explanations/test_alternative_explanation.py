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

        explanation = explanation.super_explanations()

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

        explanation = explanation.counter_explanations()

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

        explanation = explanation.semi_explanations()

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

        explanation = explanation.ensured_explanations()

        filtered = explanation.rules
        assert filtered["predict_low"] == [0.35]
        assert filtered["predict_high"] == [0.45]

    def test_pareto_explanations_removes_higher_uncertainty_for_equal_output(self, explanation):
        explanation.prediction = {"predict": 0.45, "low": 0.4, "high": 0.5, "classes": [0, 1]}

        rules = {
            "predict": [0.2, 0.2, 0.5, 0.5, 0.8],
            "predict_low": [0.0, 0.15, 0.35, 0.30, 0.65],
            "predict_high": [0.4, 0.25, 0.65, 0.70, 0.95],
            "weight": [0, 0, 0, 0, 0],
            "weight_low": [0, 0, 0, 0, 0],
            "weight_high": [0, 0, 0, 0, 0],
            "value": [1, 2, 3, 4, 5],
            "rule": ["r1", "r2", "r3", "r4", "r5"],
            "feature": [0, 0, 1, 1, 0],
            "sampled_values": [[], [], [], [], []],
            "feature_value": [None, None, None, None, None],
            "is_conjunctive": [False, False, False, False, False],
            "classes": [0, 1],
            "base_predict_low": 0.4,
            "base_predict_high": 0.5,
        }
        self.setup_test_rules(explanation, rules)

        pareto = explanation.pareto_explanations()

        filtered = pareto.rules
        assert filtered["predict"] == [0.2, 0.5, 0.8]
        assert filtered["predict_low"] == [0.15, 0.35, 0.65]
        assert filtered["predict_high"] == [0.25, 0.65, 0.95]

    def test_pareto_explanations_preserves_output_span_extremes(self, explanation):
        explanation.prediction = {"predict": 0.45, "low": 0.4, "high": 0.5, "classes": [0, 1]}

        rules = {
            "predict": [0.1, 0.4, 0.7, 0.9],
            "predict_low": [-0.15, 0.30, 0.62, 0.65],
            "predict_high": [0.35, 0.50, 0.78, 1.15],
            "weight": [0, 0, 0, 0],
            "weight_low": [0, 0, 0, 0],
            "weight_high": [0, 0, 0, 0],
            "value": [1, 2, 3, 4],
            "rule": ["r1", "r2", "r3", "r4"],
            "feature": [0, 1, 0, 1],
            "sampled_values": [[], [], [], []],
            "feature_value": [None, None, None, None],
            "is_conjunctive": [False, False, False, False],
            "classes": [0, 1],
            "base_predict_low": 0.4,
            "base_predict_high": 0.5,
        }
        self.setup_test_rules(explanation, rules)

        pareto = explanation.pareto_explanations()

        filtered = pareto.rules
        assert filtered["predict"][0] == 0.1
        assert filtered["predict"][-1] == 0.9


class TestAlternativeExplanationRegression:
    """Tests for super/semi/counter/ensured filtering on plain regression (no threshold)."""

    @pytest.fixture
    def explainer(self):
        explainer = Mock()
        explainer.categorical_features = []
        explainer.features = [0, 1]
        explainer.feature_names = ["f0", "f1"]
        explainer.categorical_labels = {0: ["0"], 1: ["1"]}
        explainer.num_classes = 2
        explainer.classes = None
        explainer.discretizer = None
        explainer.mode = "regression"
        explainer.get_explainer.return_value = explainer
        explainer.is_multiclass = Mock(return_value=False)
        explainer.discretize = Mock(return_value=[[1, 2]])
        explainer.x_cal = np.array([[1, 2], [3, 4]])
        explainer.rule_boundaries = Mock(return_value=[[0.5, 1.5], [0.5, 1.5]])
        return explainer

    @pytest.fixture
    def regression_explanation(self, explainer):
        index = 0
        x = np.array([1, 2])
        binned = {
            "f0": [1],
            "f1": [2],
            "predict": [[[150.0, 150.0], [150.0, 150.0]]],
            "low": [[[140.0, 140.0], [140.0, 140.0]]],
            "high": [[[160.0, 160.0], [160.0, 160.0]]],
            "rule_values": [[[[[1]], [[2]]], [[[3]], [[4]]]]],
        }
        feature_weights = {
            "predict": [[10.0]],
            "predict_low": [[5.0]],
            "predict_high": [[15.0]],
            "weight": [[10.0]],
            "weight_low": [[5.0]],
            "weight_high": [[15.0]],
            "value": [[1]],
            "rule": [["r1"]],
            "feature": [[0]],
            "feature_value": [[None]],
            "sampled_values": [[[]]],
            "is_conjunctive": [[False]],
        }
        feature_predict = feature_weights.copy()
        prediction = {"predict": [150.0], "high": [160.0], "low": [140.0], "classes": [None]}

        instance = AlternativeExplanation(
            calibrated_explanations=explainer,
            index=index,
            x=x,
            binned=binned,
            feature_weights=feature_weights,
            feature_predict=feature_predict,
            prediction=prediction,
        )
        instance.is_regression = Mock(return_value=True)
        instance.is_probabilistic = Mock(return_value=False)
        instance.has_conjunctive_rules = False
        return instance

    def setup_test_rules(self, instance, rules_data):
        if "base_predict_low" not in rules_data:
            rules_data["base_predict_low"] = instance.prediction["low"]
        if "base_predict_high" not in rules_data:
            rules_data["base_predict_high"] = instance.prediction["high"]
        if "classes" not in rules_data:
            rules_data["classes"] = None
        instance.get_rules = Mock(return_value=rules_data)

    def make_regression_rules(self, predicts, lows, highs, base_low=140.0, base_high=160.0):
        """Build a rules dict from prediction values."""
        n = len(predicts)
        return {
            "predict": predicts,
            "predict_low": lows,
            "predict_high": highs,
            "weight": [0] * n,
            "weight_low": [0] * n,
            "weight_high": [0] * n,
            "value": list(range(1, n + 1)),
            "rule": [f"r{i + 1}" for i in range(n)],
            "feature": [i % 2 for i in range(n)],
            "sampled_values": [[]] * n,
            "feature_value": [None] * n,
            "is_conjunctive": [False] * n,
            "classes": None,
            "base_predict_low": base_low,
            "base_predict_high": base_high,
        }

    def test_super_keeps_higher_predictions(self, regression_explanation):
        regression_explanation.prediction = {
            "predict": 150.0,
            "low": 140.0,
            "high": 160.0,
            "classes": None,
        }
        # r1=130 (lower), r2=170 (higher), r3=200 (higher)
        rules = self.make_regression_rules(
            [130.0, 170.0, 200.0],
            [120.0, 160.0, 190.0],
            [140.0, 180.0, 210.0],
        )
        # base interval differs from all rules, so no "same prediction" filter
        rules["base_predict_low"] = 140.0
        rules["base_predict_high"] = 160.0
        self.setup_test_rules(regression_explanation, rules)

        result = regression_explanation.super_explanations()
        assert result.rules["predict"] == [170.0, 200.0]

    def test_semi_keeps_lower_predictions(self, regression_explanation):
        regression_explanation.prediction = {
            "predict": 150.0,
            "low": 140.0,
            "high": 160.0,
            "classes": None,
        }
        # New 'semi' semantics for plain regression: keep alternatives where
        # intervals mutually include the other's mean. Construct two rules
        # that satisfy mutual inclusion and one that does not.
        # r1=145: interval [140,150] includes base_mean=150 and base interval
        # includes r1.mean
        # r2=155: interval [150,160] includes base_mean=150 and base interval
        # includes r2.mean
        # r3=170: does not include base_mean
        rules = self.make_regression_rules(
            [145.0, 155.0, 170.0],
            [140.0, 150.0, 160.0],
            [150.0, 160.0, 180.0],
        )
        rules["base_predict_low"] = 140.0
        rules["base_predict_high"] = 160.0
        self.setup_test_rules(regression_explanation, rules)

        result = regression_explanation.semi_explanations()
        assert result.rules["predict"] == [145.0, 155.0]

    def test_counter_keeps_lower_predictions(self, regression_explanation):
        """Counter has identical semantics to semi for plain regression."""
        regression_explanation.prediction = {
            "predict": 150.0,
            "low": 140.0,
            "high": 160.0,
            "classes": None,
        }
        rules = self.make_regression_rules(
            [130.0, 100.0, 170.0],
            [120.0, 90.0, 160.0],
            [140.0, 110.0, 180.0],
        )
        rules["base_predict_low"] = 140.0
        rules["base_predict_high"] = 160.0
        self.setup_test_rules(regression_explanation, rules)

        result = regression_explanation.counter_explanations()
        assert result.rules["predict"] == [130.0, 100.0]

    def test_potential_filters_rules_covering_original(self, regression_explanation):
        """With include_potential=False, rules whose interval covers the original are excluded."""
        regression_explanation.prediction = {
            "predict": 150.0,
            "low": 140.0,
            "high": 160.0,
            "classes": None,
        }
        # r1: interval [120, 140] does NOT cover 150 → kept
        # r2: interval [145, 155] DOES cover 150 → filtered as potential
        # r3: interval [160, 180] does NOT cover 150 → kept
        rules = self.make_regression_rules(
            [130.0, 150.0, 170.0],
            [120.0, 145.0, 160.0],
            [140.0, 155.0, 180.0],
        )
        rules["base_predict_low"] = 140.0
        rules["base_predict_high"] = 160.0
        self.setup_test_rules(regression_explanation, rules)

        # ensured_explanations passes include_potential=True by default,
        # but __filter_rules default is False, so test via super with include_potential=False
        result = regression_explanation.super_explanations(include_potential=False)
        # r3 (170) is super (higher), r2 is potential and excluded
        assert result.rules["predict"] == [170.0]

    def test_ensured_filters_by_uncertainty_width(self, regression_explanation):
        regression_explanation.prediction = {
            "predict": 150.0,
            "low": 140.0,
            "high": 160.0,
            "classes": None,
        }
        # initial_uncertainty = |160 - 140| = 20
        # r1: width=10 < 20 → kept
        # r2: width=30 > 20 → filtered
        rules = self.make_regression_rules(
            [145.0, 155.0],
            [140.0, 140.0],
            [150.0, 170.0],
        )
        rules["base_predict_low"] = 140.0
        rules["base_predict_high"] = 160.0
        self.setup_test_rules(regression_explanation, rules)

        result = regression_explanation.ensured_explanations()
        assert result.rules["predict"] == [145.0]
