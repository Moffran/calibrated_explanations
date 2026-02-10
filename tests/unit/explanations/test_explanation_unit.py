"""Unit tests for Explanation class methods."""

import pytest
from unittest.mock import Mock
from calibrated_explanations.explanations.explanation import CalibratedExplanation
import numpy as np
import pandas as pd

# Alias for convenience
Explanation = CalibratedExplanation


class ConcreteExplanation(Explanation):
    """Concrete implementation that calls super().__init__."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize rules with expected structure for add_new_rule_condition
        self.rules = {
            "rule": [],
            "predict": [],
            "predict_low": [],
            "predict_high": [],
            "weight": [],
            "weight_low": [],
            "weight_high": [],
            "value": [],
            "feature": [],
            "sampled_values": [],
            "feature_value": [],
            "is_conjunctive": [],
        }

    def build_rules_payload(self):
        return {}

    def __repr__(self):
        return "ConcreteExplanation()"

    def plot(self, *args, **kwargs):
        pass

    def _check_preconditions(self, *args, **kwargs):
        pass

    def _is_lesser(self, rule_boundary, instance_value):
        # Implementation for testing: True if value is strictly less
        return instance_value < rule_boundary

    def add_conjunctions(self, *args, **kwargs):
        pass

    def get_rules(self):
        return self.rules

    def _get_rule_str(self, is_lesser, feature, rule_boundary):
        fname = self.get_explainer().feature_names[feature]
        op = "<" if is_lesser else ">"
        return f"{fname} {op} {rule_boundary:.2f}"


class TestExplanationUnit:
    def setup_method(self):
        self.container = Mock(spec=[])
        # Default explainer mock setup
        self.explainer = Mock()
        self.explainer.y_cal = np.array([0, 1])
        # Avoid Categorical check failure by default
        self.explainer.mode = "regression"
        self.explainer.num_features = 2
        self.explainer.feature_names = ["f0", "f1"]
        self.explainer.categorical_features = [1]  # f1 is categorical
        self.explainer.x_cal = np.array([[10.0, 20.0], [15.0, 25.0], [12.0, 22.0], [5.0, 20.0]])
        self.explainer.sample_percentiles = [25, 75]

        # Prediction is now sourced via the explainer's prediction orchestrator.
        self.explainer.prediction_orchestrator = Mock()

        # Mock internal prediction to return (predict, low, high, classes)
        def predict_side_effect(x, **kwargs):
            n = len(x)
            return (np.zeros(n) + 0.5, np.zeros(n) + 0.4, np.zeros(n) + 0.6, None)

        self.explainer.prediction_orchestrator.predict_internal = Mock(
            side_effect=predict_side_effect
        )

        self.container.get_explainer = Mock(return_value=self.explainer)
        self.container.low_high_percentiles = (0.05, 0.95)

        # Valid default data
        self.index = 0
        self.x = np.array([10.0, 20.0])

        # Nested structure: [ Instance0_Data, ... ]
        # Instance0_Data: [ F0_Data, F1_Data ]
        # F0_Data: [ (val,), ... ]
        self.binned = {"rule_values": [[[(10.0,)], [(20.0,)]]], "f0": [1], "f1": [1]}
        self.weights = {"f0": [0.1], "f1": [0.2]}
        self.preds = {"f0": [0.5], "f1": [0.5]}
        # Values must be indexable (lists/arrays)
        self.prediction = {"predict": [0.5], "low": [0.4], "high": [0.6], "classes": [0]}

    def create_expl(self, **kwargs):
        # Merge defaults with kwargs
        params = {
            "calibrated_explanations": self.container,
            "index": self.index,
            "x": self.x,
            "binned": self.binned,
            "feature_weights": self.weights,
            "feature_predict": self.preds,
            "prediction": self.prediction,
        }
        params.update(kwargs)
        expl = ConcreteExplanation(**params)
        return expl

    def test_init_basic(self):
        """Test successful initialization and attribute mapping."""
        expl = self.create_expl()
        assert expl.index == 0
        assert np.array_equal(expl.x_test, self.x)
        assert expl.prediction["predict"] == 0.5
        assert expl.y_minmax == [0, 1]

    def test_init_full_probabilities(self):
        """Test __full_probabilities__ special handling."""
        prediction_with_full = self.prediction.copy()
        prediction_with_full["__full_probabilities__"] = [[0.1, 0.9]]
        expl = self.create_expl(prediction=prediction_with_full)
        assert expl.prediction["__full_probabilities__"] == [[0.1, 0.9]]

    def test_init_y_threshold_array(self):
        """Test array-like y_threshold indexing."""
        expl = self.create_expl(y_threshold=np.array([0.7]))
        assert expl.y_threshold == 0.7

        expl_tuple = self.create_expl(y_threshold=(0.5, 0.6))
        assert expl_tuple.y_threshold == (0.5, 0.6)

    def test_init_categorical_y_cal(self):
        """Test y_minmax when y_cal is categorical."""
        # Must be classification to default to [0, 1] usually
        self.explainer.mode = "classification"
        self.explainer.y_cal = pd.Categorical([0, 1, 0], categories=[0, 1], ordered=True)
        expl = self.create_expl()
        # Code hardcodes [0, 0] for Categorical
        assert expl.y_minmax == [0, 0]

    # --- Tests for add_new_rule_condition ---

    def test_add_new_rule_condition_invalid_feature(self):
        expl = self.create_expl()
        with pytest.warns(UserWarning, match="Feature invalid not found"):
            expl.add_new_rule_condition("invalid", 10.0)

    def test_add_new_rule_condition_categorical_feature(self):
        expl = self.create_expl()
        # f1 (index 1) is categorical in setup
        with pytest.warns(
            UserWarning, match="Alternatives for all categorical features are already included"
        ):
            expl.add_new_rule_condition(1, 10.0)

    def test_add_new_rule_condition_existing_rule(self):
        expl = self.create_expl()
        # Manually add a rule to simulate it exists
        expl.rules["rule"].append("f0 < 12.00")

        with pytest.warns(UserWarning, match="Rule already included"):
            expl.add_new_rule_condition(0, 12.0)

    def test_add_new_rule_condition_success_lesser(self):
        expl = self.create_expl()

        # Ensure predict returns DIFFERENT values so it doesn't trigger "identical explanation" warning
        def predict_diff_effect(x, **kwargs):
            n = len(x)
            # Default low=0.4, high=0.6. We return 0.3, 0.7.
            return (np.zeros(n) + 0.5, np.zeros(n) + 0.3, np.zeros(n) + 0.7, None)

        self.explainer.prediction_orchestrator.predict_internal = Mock(
            side_effect=predict_diff_effect
        )

        # f0 is continuous. x[0]=10.0. Boundary 12.0 -> is_lesser=True.
        # x_cal for f0: [10, 15, 12, 5]. Values < 12: [10, 5].
        # Rule string: "f0 < 12.00"

        expl.add_new_rule_condition(0, 12.0)

        assert len(expl.rules["rule"]) == 1
        assert expl.rules["rule"][0] == "f0 < 12.00"
        assert expl.rules["feature"][0] == 0

    def test_add_new_rule_condition_success_greater(self):
        expl = self.create_expl()

        # Ensure predict returns DIFFERENT values
        def predict_diff_effect(x, **kwargs):
            n = len(x)
            return (np.zeros(n) + 0.5, np.zeros(n) + 0.3, np.zeros(n) + 0.7, None)

        self.explainer.prediction_orchestrator.predict_internal = Mock(
            side_effect=predict_diff_effect
        )

        # f0 continuous. x[0]=10.0. Boundary 8.0 -> is_lesser=False

        expl.add_new_rule_condition(0, 8.0)

        assert len(expl.rules["rule"]) == 1
        assert expl.rules["rule"][0] == "f0 > 8.00"

    def test_add_new_rule_condition_identical_prediction(self):
        expl = self.create_expl()

        # Setup specific return values to trigger "identical explanation" warning
        # prediction is 0.5, low 0.4, high 0.6
        def predict_same_effect(x, **kwargs):
            n = len(x)
            return (np.zeros(n) + 0.5, np.zeros(n) + 0.4, np.zeros(n) + 0.6, None)

        self.explainer.prediction_orchestrator.predict_internal = Mock(
            side_effect=predict_same_effect
        )

        with pytest.warns(
            UserWarning,
            match="The alternative explanation is identical to the original explanation",
        ):
            expl.add_new_rule_condition(0, 12.0)

    def test_add_new_rule_condition_boundary_out_of_bounds_low(self):
        expl = self.create_expl()
        # Boundary way below min of x_cal (min is 5.0)

        # To test the "Lowest feature value..." warning, we need is_lesser=True and empty.
        # boundary = 4.0. x_test = 3.0 (update x).
        expl.x_test = np.array([3.0, 20.0])  # Update local x_test

        # x_cal min is 5.0. No values < 4.0.
        with pytest.warns(UserWarning, match="Lowest feature value for feature 0 is 5.0"):
            expl.add_new_rule_condition(0, 4.0)

    def test_add_new_rule_condition_boundary_out_of_bounds_high(self):
        expl = self.create_expl()
        # x_cal max is 15.0.
        # We need is_lesser=False. x_test > boundary.
        # boundary = 16.0. x_test = 17.0.
        expl.x_test = np.array([17.0, 20.0])

        with pytest.warns(UserWarning, match="Highest feature value for feature 0 is 15.0"):
            expl.add_new_rule_condition(0, 16.0)

    # --- Tests for utils/reset/remove ---

    def test_remove_conjunctions(self):
        expl = self.create_expl()
        expl.has_conjunctive_rules = True
        expl.remove_conjunctions()
        assert expl.has_conjunctive_rules is False

    def test_reset(self):
        expl = self.create_expl()
        expl.has_rules = True
        expl.reset()
        assert expl.has_rules is False
        # It calls get_rules too, which is mocked to return self.rules

    def test_to_python_number(self):
        """Test static telemetry helper."""
        # Scalars
        assert Explanation.to_python_number(np.int64(5)) == 5
        assert isinstance(Explanation.to_python_number(np.int64(5)), int)

        val = Explanation.to_python_number(np.float64(3.5))
        assert val == 3.5
        assert isinstance(val, float)

        # Arrays
        arr = np.array([1, 2])
        res = Explanation.to_python_number(arr)
        assert res == [1, 2]

        # NaN
        assert Explanation.to_python_number(float("nan")) is None

        # Boolean
        assert Explanation.to_python_number(np.bool_(True)) is True

        # None
        assert Explanation.to_python_number(None) is None


class TestRuleWithImpact:
    """Tests for RuleWithImpact dataclass and related functionality."""

    def test_rule_with_impact_creation(self):
        """Test RuleWithImpact dataclass can be created with valid data."""
        from calibrated_explanations.explanations.explanation import RuleWithImpact

        rule = RuleWithImpact(
            rule_id="test_rule_1",
            feature="Feature_0",
            text="Feature_0 > 0.5",
            impact=1.2,
            direction="positive",
            base_predict=0.8,
            predict=2.0,
            value="> 0.5",
            uncertainty_low=1.0,
            uncertainty_high=1.4,
            predict_low=1.8,
            predict_high=2.2,
        )

        assert rule.rule_id == "test_rule_1"
        assert rule.feature == "Feature_0"
        assert rule.text == "Feature_0 > 0.5"
        assert rule.impact == 1.2
        assert rule.direction == "positive"
        assert rule.base_predict == 0.8
        assert rule.predict == 2.0
        assert rule.value == "> 0.5"
        assert rule.uncertainty_low == 1.0
        assert rule.uncertainty_high == 1.4
        assert rule.predict_low == 1.8
        assert rule.predict_high == 2.2

    def test_rule_with_impact_optional_fields(self):
        """Test RuleWithImpact with optional fields as None."""
        from calibrated_explanations.explanations.explanation import RuleWithImpact

        rule = RuleWithImpact(
            rule_id="test_rule_2",
            feature="Feature_1",
            text="Feature_1 <= 0.3",
            impact=-0.5,
            direction="negative",
            base_predict=1.0,
            predict=0.5,
            value="<= 0.3",
        )

        assert rule.uncertainty_low is None
        assert rule.uncertainty_high is None
        assert rule.predict_low is None
        assert rule.predict_high is None


def test_rule_with_impact_dataclass():
    """Test RuleWithImpact dataclass creation and attributes."""
    from calibrated_explanations.explanations.explanation import RuleWithImpact

    rule = RuleWithImpact(
        rule_id="test_rule",
        feature="Feature_0",
        text="Feature_0 > 0.5",
        impact=1.2,
        direction="positive",
        base_predict=1.0,
        predict=2.0,
        value="> 0.5",
        uncertainty_low=1.0,
        uncertainty_high=1.4,
        predict_low=1.8,
        predict_high=2.2,
    )

    assert rule.rule_id == "test_rule"
    assert rule.feature == "Feature_0"
    assert rule.text == "Feature_0 > 0.5"
    assert rule.impact == 1.2
    assert rule.direction == "positive"
    assert rule.base_predict == 1.0
    assert rule.predict == 2.0
    assert rule.value == "> 0.5"
    assert rule.uncertainty_low == 1.0
    assert rule.uncertainty_high == 1.4
    assert rule.predict_low == 1.8
    assert rule.predict_high == 2.2
