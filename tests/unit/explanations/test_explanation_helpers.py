from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.explanations import explanation as explanation_module


class ExplainerStub:
    def __init__(self):
        self.y_cal = np.array([0.1, 0.9])
        self.mode = "regression"
        self.class_labels = None

    def is_multiclass(self):  # pragma: no cover - not exercised here
        return False


class ContainerStub:
    def __init__(self):
        self.low_high_percentiles = (5, 95)
        self._explainer = ExplainerStub()

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
    binned = {"weights": np.array([[1, 2], [3, 4]])}
    feature_weights = {"weights": np.array([[0.1, 0.2], [0.3, 0.4]])}
    feature_predict = {"weights": np.array([[0.5, 0.6], [0.7, 0.8]])}
    prediction = {
        "predict": np.array([0.2, 0.6]),
        "low": np.array([0.1, 0.5]),
        "high": np.array([0.4, 0.9]),
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


def test_rank_features_requires_input(simple_explanation):
    with pytest.raises(ValueError):
        simple_explanation._rank_features()


def test_percentile_helpers():
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(95) == pytest.approx(0.95)
    assert explanation_module.CalibratedExplanation._normalize_percentile_value(-np.inf) == -np.inf
    assert explanation_module.CalibratedExplanation._normalize_percentile_value("invalid") is None

    percentiles = (0.05, 0.95)
    assert explanation_module.CalibratedExplanation._compute_confidence_level(percentiles) == pytest.approx(0.9)
    assert (
        explanation_module.CalibratedExplanation._compute_confidence_level((-np.inf, 0.9))
        == pytest.approx(0.9)
    )
    assert (
        explanation_module.CalibratedExplanation._compute_confidence_level((0.1, np.inf))
        == pytest.approx(0.9)
    )


def test_get_percentiles_and_one_sided(simple_explanation):
    assert simple_explanation._get_percentiles() == (0.05, 0.95)
    assert not simple_explanation.is_one_sided()

    simple_explanation.calibrated_explanations.low_high_percentiles = (-np.inf, 95)
    assert simple_explanation.is_one_sided()


def test_to_python_number_handles_nested_arrays():
    arr = np.array([np.array([1.0]), np.array([np.nan])])
    result = explanation_module.CalibratedExplanation._to_python_number(arr)
    assert result == [[1.0], [None]]
