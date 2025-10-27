from __future__ import annotations

import numpy as np
import pytest
from pandas import Categorical

from calibrated_explanations.explanations import explanation as explanation_module


class DiscretizerStub:
    def __init__(self):
        self.names = [
            ["f0 <= 0.2", "f0 > 0.2"],
            ["f1 <= 0.2", "f1 > 0.2"],
            ["f2 <= 0.2", "f2 > 0.2"],
        ]

    def discretize(self, x):  # pragma: no cover - simple passthrough
        # mimic the discretizer contract where each column maps to a bin index
        return np.arange(x.shape[-1], dtype=int)


class ExplainerStub:
    def __init__(self, mode="regression", y_cal=None):
        self.y_cal = np.array([0.1, 0.9]) if y_cal is None else y_cal
        self.mode = mode
        self.class_labels = None
        self.feature_names = ["f0", "f1", "f2"]
        self.categorical_features: list[int] = []
        self.categorical_labels = None
        self.feature_values = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
        self.sample_percentiles = [25, 75]
        self.num_features = 2
        self.discretizer = DiscretizerStub()
        self.x_cal = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.rule_boundaries = lambda _instance: []

    def assign_threshold(self, threshold):  # pragma: no cover - passthrough for tests
        return threshold

    def is_multiclass(self):  # pragma: no cover - not exercised here
        return False

    def _predict(self, data, **_kwargs):
        rows = data.shape[0]
        predict = np.ones((rows, 1))
        low = np.full((rows, 1), 0.5)
        high = np.full((rows, 1), 1.5)
        return predict, low, high, np.zeros(rows)


class ContainerStub:
    def __init__(self, *, mode="regression", y_cal=None):
        self.low_high_percentiles = (5, 95)
        self._explainer = ExplainerStub(mode=mode, y_cal=y_cal)
        self.features_to_ignore = []
        self.explanations = []

    def _get_explainer(self):
        return self._explainer

    def get_low_percentile(self):
        low = self.low_high_percentiles[0]
        if low == -np.inf:
            return -np.inf
        return low / 100.0

    def get_high_percentile(self):
        high = self.low_high_percentiles[1]
        if high == np.inf:
            return np.inf
        return high / 100.0


class SimpleExplanation(explanation_module.CalibratedExplanation):
    def __repr__(self):  # pragma: no cover - trivial
        return "simple"

    def plot(self, filter_top=None, **kwargs):  # pragma: no cover - not needed in tests
        return None

    def add_conjunctions(self, n_top_features=5, max_rule_size=2):  # pragma: no cover - stub
        return self

    def _check_preconditions(self):  # pragma: no cover - stub
        return None

    def _get_rules(self):
        if not hasattr(self, "_rules"):
            self._rules = {"rule": ["r1", "r2"]}
        return self._rules

    def _is_lesser(self, other):  # pragma: no cover - comparison unused in tests
        return False


@pytest.fixture
def simple_explanation():
    container = ContainerStub()
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    rule_values = np.empty((2, 2), dtype=object)
    rule_values[0, 0] = np.array([[0.1]])
    rule_values[0, 1] = np.array([[0.2]])
    rule_values[1, 0] = np.array([[0.3]])
    rule_values[1, 1] = np.array([[0.4]])
    binned = {
        "predict": np.array([[0.2, 0.3], [0.4, 0.5]]),
        "low": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "high": np.array([[0.3, 0.4], [0.5, 0.6]]),
        "rule_values": rule_values,
    }
    feature_weights = {
        "predict": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "low": np.array([[0.05, 0.1], [0.15, 0.2]]),
        "high": np.array([[0.15, 0.25], [0.35, 0.45]]),
    }
    feature_predict = {
        "predict": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "low": np.array([[0.4, 0.5], [0.6, 0.7]]),
        "high": np.array([[0.6, 0.7], [0.8, 0.9]]),
    }
    prediction = {
        "predict": np.array([0.2, 0.6]),
        "low": np.array([0.1, 0.5]),
        "high": np.array([0.4, 0.9]),
        "classes": np.array([1, 1]),
    }
    return SimpleExplanation(
        container,
        index=1,
        x=x,
        binned=binned,
        feature_weights=feature_weights,
        feature_predict=feature_predict,
        prediction=prediction,
        y_threshold=None,
        instance_bin=0,
    )


def test_prediction_interval(simple_explanation):
    assert simple_explanation.prediction_interval == (0.5, 0.9)
    assert simple_explanation.predict == pytest.approx(0.6)


def test_categorical_target_uses_default_minmax():
    categorical = Categorical(["a", "b"])
    explanation = SimpleExplanation(
        ContainerStub(y_cal=categorical),
        index=1,
        x=np.array([[0.1, 0.2], [0.3, 0.4]]),
        binned={"weights": np.array([[1, 2], [3, 4]])},
        feature_weights={"weights": np.array([[0.1, 0.2], [0.3, 0.4]])},
        feature_predict={"weights": np.array([[0.5, 0.6], [0.7, 0.8]])},
        prediction={
            "predict": np.array([0.2, 0.6]),
            "low": np.array([0.1, 0.5]),
            "high": np.array([0.4, 0.9]),
        },
        y_threshold=None,
        instance_bin=0,
    )
    assert explanation.y_minmax == [0, 0]


def test_length_and_metadata_helpers(simple_explanation):
    explainer = simple_explanation._get_explainer()
    explainer.class_labels = ["negative", "positive"]
    assert len(simple_explanation) == 2
    assert simple_explanation.get_mode() == "regression"
    assert simple_explanation.get_class_labels() == ["negative", "positive"]
    assert simple_explanation.is_thresholded() is False
    assert simple_explanation.is_probabilistic() is False


def test_rank_features_requires_input(simple_explanation):
    with pytest.raises(ValueError):
        simple_explanation._rank_features()


def test_rank_features_orders_by_width_and_weight(simple_explanation):
    weights = np.array([0.4, -0.1, 0.2])
    widths = np.array([0.3, 0.5, 0.1])
    ranked = simple_explanation._rank_features(feature_weights=weights, width=widths)
    assert set(ranked) == {0, 1, 2}
    assert ranked[-1] == 0  # highest absolute weight retained last
    limited = simple_explanation._rank_features(feature_weights=weights, num_to_show=2)
    assert len(limited) == 2
    width_only = simple_explanation._rank_features(width=widths)
    assert set(width_only) == {0, 1, 2}


def test_percentile_helpers():
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(
        95
    ) == pytest.approx(0.95)
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(-np.inf) == -np.inf
    assert explanation_module.CalibratedExplanation._normalize_percentile_value("invalid") is None
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(None) is None

    percentiles = (0.05, 0.95)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(
        percentiles
    ) == pytest.approx(0.9)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(
        (-np.inf, 0.9)
    ) == pytest.approx(0.9)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(
        (0.1, np.inf)
    ) == pytest.approx(0.9)


def test_get_percentiles_and_one_sided(simple_explanation):
    assert simple_explanation._get_percentiles() == (0.05, 0.95)
    assert not simple_explanation.is_one_sided()

    simple_explanation.calibrated_explanations.low_high_percentiles = (-np.inf, 95)
    assert simple_explanation.is_one_sided()


def test_to_python_number_handles_nested_arrays():
    arr = np.array([np.array([1.0]), np.array([np.nan])])
    result = explanation_module.CalibratedExplanation._to_python_number(arr)
    assert result == [[1.0], [None]]
