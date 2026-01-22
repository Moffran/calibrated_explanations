import math
import numpy as np
import pytest

from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.utils.exceptions import ValidationError


def test_to_python_number_conversions_and_normalize():
    assert CalibratedExplanation.to_python_number(None) is None
    assert CalibratedExplanation.to_python_number(np.int32(3)) == 3
    assert CalibratedExplanation.to_python_number(np.int64(3)) == 3
    assert CalibratedExplanation.to_python_number(np.float64(2.5)) == 2.5
    assert CalibratedExplanation.to_python_number(np.bool_(True)) is True
    assert CalibratedExplanation.to_python_number(np.array([1, 2])) == [1, 2]
    assert CalibratedExplanation.to_python_number(float("nan")) is None


def test_normalize_percentile_and_confidence():
    assert CalibratedExplanation.normalize_percentile_value(None) is None
    assert CalibratedExplanation.normalize_percentile_value(95) == 0.95
    assert CalibratedExplanation.normalize_percentile_value(0.5) == 0.5
    inf = float("inf")
    assert CalibratedExplanation.normalize_percentile_value(inf) == inf

    # compute confidence
    confidence = CalibratedExplanation.compute_confidence_level((0.05, 0.95))
    assert math.isclose(confidence, 0.9)
    assert CalibratedExplanation.compute_confidence_level(None) is None
    assert CalibratedExplanation.compute_confidence_level((None, 0.9)) is None


class DummyExplainer:
    def __init__(self):
        self.y_cal = np.array([0.0, 1.0])
        self.mode = "regression"
        self.feature_names = ["age", "income"]

    def is_multiclass(self):
        return False


class DummyCollection:
    def __init__(self):
        self.explainer = DummyExplainer()
        self.features_to_ignore = None
        self.feature_filter_per_instance_ignore = None
        self.low_high_percentiles = (5, 95)


class DummyExplanation(CalibratedExplanation):
    def build_rules_payload(self):
        return {"rule": []}

    def __repr__(self):
        return "Dummy"

    def plot(self, filter_top=None, **kwargs):
        return None

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):
        return self

    def _check_preconditions(self):
        return True

    def _is_lesser(self, a, b):
        return a < b


def make_instance():
    collection = DummyCollection()
    binned = {"f": [[0]]}
    feature_weights = {"f": [[1.0, 2.0]]}
    feature_predict = {"f": [[0.1, 0.2]]}
    prediction = {"predict": [0.5], "low": [0.1], "high": [0.9]}
    return DummyExplanation(
        collection,
        0,
        x=[0],
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
    )


def test_build_interval_and_uncertainty_payload_and_ranking():
    inst = make_instance()
    interval = CalibratedExplanation.build_interval(np.float64(0.1), np.float64(0.9))
    assert interval["lower"] == 0.1 and interval["upper"] == 0.9

    payload = inst.build_instance_uncertainty()
    assert payload["representation"] in {"percentile", "venn_abers", "threshold"}
    assert payload["lower_bound"] == 0.1
    assert payload["upper_bound"] == 0.9

    # ranking by width
    idxs = inst.rank_features(feature_weights=None, width=[0.1, 0.5, 0.2], num_to_show=2)
    assert set(idxs) == {1, 2}


def test_condition_parsing_and_payload_public_api():
    inst = make_instance()
    op, val = inst.parse_condition("age", "age >= 30")
    assert op == ">=" and val == "30"

    assert CalibratedExplanation.convert_condition_value(None, 5) == 5
    assert math.isinf(CalibratedExplanation.convert_condition_value("-inf", 0))
    assert CalibratedExplanation.convert_condition_value("3.14", 0) == 3.14

    # safe_feature_name is public
    assert inst.safe_feature_name(0) == "age"


def test_ignored_and_rank_features_errors():
    col = DummyCollection()
    col.features_to_ignore = (1,)
    col.feature_filter_per_instance_ignore = [[2]]
    binned = {"f": [[0]]}
    feature_weights = {"f": [[1.0]]}
    feature_predict = {"f": [[0.1]]}
    prediction = {"predict": [0.5], "low": [0.1], "high": [0.9]}
    inst = DummyExplanation(
        col,
        0,
        x=[0],
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
    )
    ignored = inst.ignored_features_for_instance()
    assert 1 in ignored and 2 in ignored

    with pytest.raises(ValidationError):
        inst.rank_features()


def test_normalize_threshold_value_variants():
    col = DummyCollection()
    binned = {"f": [[0]]}
    feature_weights = {"f": [[1.0]]}
    feature_predict = {"f": [[0.1]]}
    prediction = {"predict": [0.5], "low": [0.1], "high": [0.9]}
    inst1 = DummyExplanation(
        col,
        0,
        x=[0],
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
        y_threshold=0.5,
    )
    assert inst1.normalize_threshold_value() == 0.5

    inst2 = DummyExplanation(
        col,
        0,
        x=[0],
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
        y_threshold=(0.2, 0.8),
    )
    assert inst2.normalize_threshold_value() == [0.2, 0.8]
