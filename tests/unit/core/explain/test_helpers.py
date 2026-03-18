"""Tests for explain helpers module.

This module tests the helper functions that were previously called as private
methods on CalibratedExplainer. Tests are refactored to test behavior through
the public explain module API rather than implementation details.
"""

import numpy as np
import pytest
from typing import Any

from calibrated_explanations.core.explain import helpers as helpers
from calibrated_explanations.core.explain import _helpers as helper_impl


class TestSliceThreshold:
    """Test behavior of slice_threshold helper function."""

    def test_slice_threshold_returns_scalar_as_is(self):
        assert helper_impl.slice_threshold(0.5, 0, 1, 3) == 0.5

    def test_slice_threshold_returns_input_when_length_mismatch(self):
        threshold = np.array([0.1, 0.2])
        out = helper_impl.slice_threshold(threshold, 0, 1, total_len=3)
        assert out is threshold

    def test_slice_threshold_slices_numpy_array(self):
        threshold = np.array([0.1, 0.2, 0.3, 0.4])
        out = helper_impl.slice_threshold(threshold, 1, 3, total_len=4)
        np.testing.assert_allclose(out, np.array([0.2, 0.3]))


class TestSliceBins:
    """Test behavior of slice_bins helper function."""

    def test_slice_bins_none_returns_none(self):
        assert helper_impl.slice_bins(None, 0, 1) is None

    def test_slice_bins_array_like(self):
        bins = np.array([10, 20, 30, 40])
        out = helper_impl.slice_bins(bins, 1, 3)
        np.testing.assert_array_equal(out, np.array([20, 30]))


class TestComputeWeightDelta:
    """Test behavior of compute_weight_delta helper function."""

    def test_compute_weight_delta_fallback_path_for_object_values(self):
        """Object subtraction should fall back to element-wise scalar extraction."""

        class UnevenDelta:
            def __init__(self, value: float, width: int) -> None:
                self.value = value
                self.width = width

            def __rsub__(self, other: float):
                return [float(other) - self.value] * self.width

        baseline = [4.0, 6.0]
        perturbed = [UnevenDelta(1.0, 2), UnevenDelta(2.0, 1)]

        result = helpers.compute_weight_delta(baseline, perturbed)

        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0]))

    def test_compute_weight_delta_scalar(self):
        out = helper_impl.compute_weight_delta(2.0, 1.5)
        np.testing.assert_allclose(out, np.array(0.5))


def test_merge_ignore_features_union_behavior():
    class Explainer:
        features_to_ignore = [0, 2]

    out = helper_impl.merge_ignore_features(Explainer(), [1, 2, 3])
    np.testing.assert_array_equal(out, np.array([0, 1, 2, 3]))


def test_feature_effect_for_index_uses_prediction_orchestrator():
    class PredictionOrchestrator:
        @staticmethod
        def predict_internal(x, threshold=None, low_high_percentiles=None, bins=None, feature=None):
            predict = np.array([0.4, 0.5]) + feature * 0.1
            low = predict - 0.1
            high = predict + 0.2
            return predict, low, high, None

    class Explainer:
        prediction_orchestrator = PredictionOrchestrator()

    baseline = {"predict": np.array([0.7, 0.8])}
    res = helper_impl.feature_effect_for_index(
        Explainer(),
        1,
        x=np.array([[1.0], [2.0]]),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        baseline_prediction=baseline,
    )
    assert res[0] == 1
    np.testing.assert_allclose(res[1], np.array([0.2, 0.2]))


def test_compute_feature_effects_with_executor_path():
    class PredictionOrchestrator:
        @staticmethod
        def predict_internal(x, threshold=None, low_high_percentiles=None, bins=None, feature=None):
            predict = np.array([0.2, 0.3]) + feature * 0.1
            low = predict - 0.05
            high = predict + 0.05
            return predict, low, high, None

    class Explainer:
        prediction_orchestrator = PredictionOrchestrator()

    class DummyExecutor:
        def __init__(self):
            self.work_items = None

        def map(self, fn, items, work_items=None):
            self.work_items = work_items
            return [fn(item) for item in items]

    executor = DummyExecutor()
    results = helper_impl.compute_feature_effects(
        Explainer(),
        features_to_process=[0, 1],
        x=np.array([[0.0], [1.0]]),
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        prediction={"predict": np.array([0.4, 0.5])},
        executor=executor,
    )
    assert len(results) == 2
    assert executor.work_items == 4


def test_merge_feature_result_populates_buffers_and_rules():
    result = (
        1,
        np.array([1.0, 2.0]),
        np.array([0.8, 1.8]),
        np.array([1.2, 2.2]),
        np.array([10.0, 20.0]),
        np.array([9.0, 19.0]),
        np.array([11.0, 21.0]),
        [{"rule": "a"}, None],
        [
            (
                np.array([1.0]),
                np.array([0.8]),
                np.array([1.2]),
                np.array([1]),
                np.array([3]),
                np.array([0.5]),
            ),
            None,
        ],
        np.array([0.0, 0.1]),
        np.array([0.9, 1.0]),
    )
    weights_predict = np.zeros((2, 3))
    weights_low = np.zeros((2, 3))
    weights_high = np.zeros((2, 3))
    predict_matrix = np.zeros((2, 3))
    low_matrix = np.zeros((2, 3))
    high_matrix = np.zeros((2, 3))
    rule_values = [{}, {}]
    instance_binned = [
        {"predict": {}, "low": {}, "high": {}, "current_bin": {}, "counts": {}, "fractions": {}},
        {"predict": {}, "low": {}, "high": {}, "current_bin": {}, "counts": {}, "fractions": {}},
    ]
    rule_boundaries = np.zeros((2, 3, 2))

    helper_impl.merge_feature_result(
        result,
        weights_predict,
        weights_low,
        weights_high,
        predict_matrix,
        low_matrix,
        high_matrix,
        rule_values,
        instance_binned,
        rule_boundaries,
    )

    np.testing.assert_allclose(weights_predict[:, 1], np.array([1.0, 2.0]))
    assert rule_values[0][1] == {"rule": "a"}
    np.testing.assert_allclose(rule_boundaries[:, 1, 0], np.array([0.0, 0.1]))


@pytest.mark.parametrize(
    "func_name,args,kwargs",
    [
        ("compute_feature_effects", ("payload",), {}),
        ("initialize_explanation", ("init",), {"mode": "factual"}),
        ("explain_predict_step", ("state",), {"step": 2}),
        ("feature_effect_for_index", ("state",), {"index": 1}),
        ("validate_and_prepare_input", ("x",), {"y": "z"}),
    ],
)
def test_should_delegate_to_internal_helpers_when_called(func_name, args, kwargs, monkeypatch):
    """Each public helper should forward to the internal implementation."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def stub(*stub_args: Any, **stub_kwargs: Any) -> str:
        calls.append((stub_args, stub_kwargs))
        return "sentinel"

    monkeypatch.setattr(helpers.impl, func_name, stub)

    result = getattr(helpers, func_name)(*args, **kwargs)

    assert result == "sentinel"
    assert calls == [(args, kwargs)]
