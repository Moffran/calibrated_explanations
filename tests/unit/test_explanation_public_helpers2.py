import math
import numpy as np

from calibrated_explanations.explanations.explanation import CalibratedExplanation


def test_to_python_number_and_nan():
    assert CalibratedExplanation.to_python_number(np.int32(3)) == 3
    assert CalibratedExplanation.to_python_number(np.float64(1.5)) == 1.5
    assert CalibratedExplanation.to_python_number(np.array([1, 2])) == [1, 2]
    assert CalibratedExplanation.to_python_number(np.nan) is None


def test_normalize_percentile_value_and_compute_confidence():
    assert CalibratedExplanation.normalize_percentile_value(None) is None
    assert CalibratedExplanation.normalize_percentile_value(50) == 0.5
    assert math.isinf(CalibratedExplanation.normalize_percentile_value(math.inf))
    percentiles = (0.1, 0.9)
    assert CalibratedExplanation.compute_confidence_level(percentiles) == 0.8
    assert CalibratedExplanation.compute_confidence_level(None) is None
    # -inf low returns high when high finite
    assert CalibratedExplanation.compute_confidence_level((float("-inf"), 0.2)) == 0.2


def test_build_interval_and_uncertainty_payload():
    interval = CalibratedExplanation.build_interval(0.1, 0.9)
    assert interval == {"lower": 0.1, "upper": 0.9}
    payload = CalibratedExplanation.build_uncertainty_payload(
        None,
        value=0.5,
        low=0.4,
        high=0.6,
        representation="percentile",
        percentiles=(0.1, 0.9),
        threshold=None,
        include_percentiles=True,
    )
    assert payload["legacy_interval"] == [0.4, 0.6]
    assert payload["raw_percentiles"] == [0.1, 0.9]
    assert payload["confidence_level"] == 0.8


def test_convert_and_parse_condition():
    assert CalibratedExplanation.convert_condition_value(None, 3) == 3
    assert CalibratedExplanation.convert_condition_value("-inf", 0) == float("-inf")
    assert CalibratedExplanation.convert_condition_value("3.5", 0) == 3.5
    # parse_condition tested later on a real instance


def test_normalize_threshold_value_and_instance_uncertainty():
    class DummyExplainer:
        mode = "regression"
        class_labels = None

        def is_multiclass(self):
            return False

        y_cal = np.array([0.0, 1.0])
        feature_names = ["a", "b"]
        num_features = 2
        categorical_features = None
        categorical_labels = None
        discretizer = None
        sample_percentiles = [25, 75]
        x_cal = np.array([[1.0, 2.0], [3.0, 4.0]])

    class DummyContainer:
        def get_explainer(self):
            return DummyExplainer()

        features_to_ignore = None
        feature_filter_per_instance_ignore = None
        low_high_percentiles = (5, 95)
        sample_percentiles = [25, 75]
        calibrated_explainer = None

    class E(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return ""

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return None

        def _is_lesser(self, rule_boundary, instance_value):
            return instance_value < rule_boundary

    container = DummyContainer()
    binned = {"rule_values": [[[0.0], [0.0]], [[0.0], [0.0]]]}
    feature_weights = {"predict": [[0.1, 0.2]]}
    feature_predict = {"predict": [[0.2, 0.3]]}
    prediction = {"predict": [0.25], "low": [0.2], "high": [0.3], "classes": [0]}
    inst = E(
        container,
        0,
        np.array([1.0, 2.0]),
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=(0.1, 0.9),
        instance_bin=0,
    )
    # normalize_threshold_value returns list for tuple thresholds
    nt = inst.normalize_threshold_value()
    assert isinstance(nt, list)
    # build_instance_uncertainty should return payload dict
    iu = inst.build_instance_uncertainty()
    assert "representation" in iu
    parsed = inst.parse_condition("age", "age >= 10")
    assert parsed[0] == ">=" and parsed[1] == "10"
