"""Additional branch coverage for :mod:`calibrated_explanations.core.calibrated_explainer`."""

from __future__ import annotations

import types

import numpy as np
import pytest

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)
from calibrated_explanations.utils.discretizers import RegressorDiscretizer


def test_slice_threshold_branches_exercised():
    """Test threshold slicing behavior through explain helpers.
    
    Tests should call explain module functions directly.
    """
    from calibrated_explanations.core.explain._helpers import slice_threshold

    sentinel = object()
    assert slice_threshold(sentinel, 0, 1, 1) is sentinel

    mismatched = [0.1, 0.2]
    assert slice_threshold(mismatched, 0, 1, 3) is mismatched

    data = [1, 2, 3, 4]
    assert slice_threshold(data, 1, 3, len(data)) == [2, 3]

    array = np.arange(5)
    result = slice_threshold(array, 1, 4, len(array))
    assert np.all(result == np.array([1, 2, 3]))


def test_slice_bins_handles_collections():
    """Test bins slicing behavior through explain helpers.
    
    Tests should call explain module functions directly.
    """
    from calibrated_explanations.core.explain._helpers import slice_bins

    assert slice_bins(None, 0, 1) is None

    bins = ["a", "b", "c"]
    assert slice_bins(bins, 1, 3) == ["b", "c"]

    array_bins = np.array([[1, 2], [3, 4], [5, 6]])
    sliced = slice_bins(array_bins, 0, 2)
    assert np.all(sliced == array_bins[:2])


def test_infer_explanation_mode_prefers_discretizer(explainer_factory):
    explainer = explainer_factory()
    assert explainer._infer_explanation_mode() == "factual"

    data = np.array([[0.0], [1.0]])
    labels = np.array([0.0, 1.0])
    discretizer = RegressorDiscretizer(
        data,
        categorical_features=[],
        feature_names=["f0"],
        labels=labels,
        random_state=0,
    )
    explainer.discretizer = discretizer
    assert explainer._infer_explanation_mode() == "alternative"


def test_set_mode_variants(monkeypatch, explainer_factory):
    explainer = explainer_factory()

    explainer._CalibratedExplainer__set_mode("classification", initialize=False)
    assert explainer.mode == "classification"
    assert explainer.num_classes == 2

    explainer._CalibratedExplainer__set_mode("regression", initialize=False)
    assert explainer.mode == "regression"
    assert explainer.num_classes == 0

    with pytest.raises(ValidationError):
        explainer._CalibratedExplainer__set_mode("unsupported", initialize=False)


def test_get_sigma_test_uses_difficulty_estimator(explainer_factory):
    explainer = explainer_factory()
    values = explainer._get_sigma_test(np.zeros((3, explainer.num_features)))
    assert np.all(values == 1)

    class _Estimator:
        def apply(self, x):
            return np.full(x.shape[0], 0.42)

    explainer.difficulty_estimator = _Estimator()
    updated = explainer._get_sigma_test(np.zeros((2, explainer.num_features)))
    assert np.all(updated == 0.42)


def test_reinitialize_updates_state(monkeypatch, explainer_factory):
    explainer = explainer_factory()

    appended: list[tuple] = []

    def append_cal(self, xs, ys):
        appended.append((xs, ys))

    explainer.append_cal = types.MethodType(append_cal, explainer)
    explainer.bins = np.array([0.5, 0.6])

    checked: list[object] = []

    def fake_check_is_fitted(learner):
        checked.append(learner)

    monkeypatch.setattr(explainer_module, "check_is_fitted", fake_check_is_fitted)


    update_calls: list[tuple] = []

    def fake_update(self, xs, ys, bins=None):
        update_calls.append((self, xs, ys, bins))

    monkeypatch.setattr(
        "calibrated_explanations.core.calibration.interval_learner.update_interval_learner",
        fake_update,
    )

    learner = object()
    xs = np.array([[2.0, 3.0], [4.0, 5.0]])
    ys = np.array([1, 0])
    bins = np.array([7, 8])

    explainer.reinitialize(learner, xs=xs, ys=ys, bins=bins)

    assert checked == [learner]
    assert appended == [(xs, ys)]
    assert np.all(explainer.bins == np.array([0.5, 0.6, 7, 8]))
    assert update_calls and update_calls[0][0] is explainer
    assert explainer._CalibratedExplainer__initialized is True

    with pytest.raises(DataShapeError):
        explainer.reinitialize(learner, xs=xs, ys=ys, bins=np.array([1]))

    init_calls: list[tuple] = []

    def fake_init(self):
        init_calls.append((self,))

    monkeypatch.setattr(
        "calibrated_explanations.core.calibration.interval_learner.initialize_interval_learner",
        fake_init,
    )

    explainer.reinitialize(learner)
    assert init_calls and init_calls[0][0] is explainer

