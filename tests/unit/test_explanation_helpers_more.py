import math
import numpy as np
import pytest

from calibrated_explanations.explanations.explanation import CalibratedExplanation
from calibrated_explanations.utils.exceptions import ValidationError


def test_to_python_number_and_build_interval():
    # numpy scalar -> python scalar
    assert CalibratedExplanation.to_python_number(np.int32(3)) == 3
    assert CalibratedExplanation.to_python_number(np.float64(1.5)) == 1.5
    # nan -> None
    assert CalibratedExplanation.to_python_number(np.float64("nan")) is None
    # numpy bool
    assert CalibratedExplanation.to_python_number(np.bool_(True)) is True
    # ndarray -> list
    assert CalibratedExplanation.to_python_number(np.array([1, 2])) == [1, 2]

    interval = CalibratedExplanation.build_interval(0.1, 0.9)
    assert interval == {"lower": 0.1, "upper": 0.9}


def test_convert_condition_value_and_parse():
    # None -> fallback
    assert CalibratedExplanation.convert_condition_value(None, 5) == 5
    assert math.isinf(CalibratedExplanation.convert_condition_value("-inf", 0))
    assert math.isinf(CalibratedExplanation.convert_condition_value("inf", 0))
    assert CalibratedExplanation.convert_condition_value("3.14", 0) == 3.14
    # non-numeric string returns string
    assert CalibratedExplanation.convert_condition_value("abc", 0) == "abc"

    # parse_condition
    # Use a concrete dummy class to exercise the public parse_condition API
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)

    class DummyExplainer:
        feature_names = ["f0"]

    class DummyContainer:
        def __init__(self):
            self.expl = DummyExplainer()

        def get_explainer(self):
            return self.expl

    inst.calibrated_explanations = DummyContainer()
    op, val = inst.parse_condition("f0", "f0 <= 3")
    assert op == "<=" and val == "3"
    op, val = inst.parse_condition("f0", "f0 = 2")
    assert op == "==" and val == "2"
    op, val = inst.parse_condition("f0", "unexpected")
    assert op == "raw"


def test_build_condition_payload_and_threshold_normalize():
    # Create a concrete dummy subclass to allow instantiation
    class DummyExplanation(CalibratedExplanation):
        def build_rules_payload(self):
            return {"core": {}, "metadata": {}}

        def __repr__(self):
            return "Dummy"

        def plot(self, filter_top=None, **kwargs):
            return None

        def add_conjunctions(self, n_top_features=5, max_rule_size=2):
            return self

        def _check_preconditions(self):
            return True

        def _is_lesser(self):
            return False

    inst = object.__new__(DummyExplanation)

    # minimal explainer with feature_names
    class DummyExplainer2:
        feature_names = ["age"]

    class DummyContainer2:
        def __init__(self):
            self.expl = DummyExplainer2()

        def get_explainer(self):
            return self.expl

    inst.calibrated_explanations = DummyContainer2()
    inst.get_explainer = lambda: inst.calibrated_explanations.get_explainer()

    payload = inst.build_condition_payload(0, "age = 30", 30, "30")
    assert payload["operator"] == "=="
    assert payload["feature"] == "age"

    # normalize_threshold_value: None, scalar, list/tuple
    inst.y_threshold = None
    assert inst.normalize_threshold_value() is None
    inst.y_threshold = 0.4
    assert inst.normalize_threshold_value() == 0.4
    inst.y_threshold = [0.2, 0.8]
    assert inst.normalize_threshold_value() == [0.2, 0.8]


# moved imports to top


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
    # inputs are mappings from keys -> sequences; CalibratedExplanation picks index 0
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


def test_to_python_number_conversions():
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


def test_build_interval_and_uncertainty_payload():
    inst = make_instance()
    interval = CalibratedExplanation.build_interval(np.float64(0.1), np.float64(0.9))
    assert interval["lower"] == 0.1 and interval["upper"] == 0.9

    payload = inst.build_instance_uncertainty()
    # percentiles (5,95) -> confidence level 0.9 should be present
    assert payload["representation"] in {"percentile", "venn_abers", "threshold"}
    assert payload["lower_bound"] == 0.1
    assert payload["upper_bound"] == 0.9
    # confidence_level may be set because percentiles present
    assert payload.get("confidence_level") is not None


def test_condition_parsing_and_payload():
    inst = make_instance()
    # parse simple expression using the feature name 'age'
    op, val = inst.parse_condition("age", "age >= 30")
    assert op == ">=" and val == "30"

    # convert textual values
    assert CalibratedExplanation.convert_condition_value(None, 5) == 5
    assert math.isinf(CalibratedExplanation.convert_condition_value("-inf", 0))
    assert CalibratedExplanation.convert_condition_value("3.14", 0) == 3.14

    # use public API only: safe_feature_name to resolve feature index
    assert inst.safe_feature_name(0) == "age"


def test_ignored_features_and_rank_features():
    col = DummyCollection()
    col.features_to_ignore = (1,)
    col.feature_filter_per_instance_ignore = [[2]]
    binned = {"f": [[0]]}
    feature_weights = {"f": [[1.0, 2.0]]}
    feature_predict = {"f": [[0.1, 0.2]]}
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

    # rank by width
    idxs = inst.rank_features(feature_weights=None, width=[0.1, 0.5, 0.2], num_to_show=2)
    assert set(idxs) == {1, 2}

    # error when neither provided
    with pytest.raises(ValidationError):
        inst.rank_features()


def test_normalize_threshold_value_variants():
    # scalar threshold
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

    # list/tuple threshold
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


def test_build_uncertainty_payload_and_ranking_both_args():
    inst = make_instance()
    # call the public alias which maps to the internal builder
    payload = inst.build_uncertainty_payload(
        value=0.5,
        low=0.1,
        high=0.9,
        representation="percentile",
        percentiles=(0.05, 0.95),
        threshold=None,
        include_percentiles=True,
    )
    assert payload["legacy_interval"] == [0.1, 0.9]
    assert payload["raw_percentiles"] == [0.05, 0.95]

    # ranking when both feature_weights and width provided
    feature_weights = [0.1, -0.9, 0.2]
    width = [0.05, 0.2, 0.1]
    top = inst.rank_features(feature_weights=feature_weights, width=width, num_to_show=2)
    assert set(top) == {1, 2}


def test_convert_condition_value_fallback_and_parse_raw():
    # non-numeric fallback path
    assert CalibratedExplanation.convert_condition_value("not-a-number", 7) == "not-a-number"
    # parse_condition raw when empty
    inst = make_instance()
    op, val = inst.parse_condition("age", "")
    assert op == "raw" and val is None
