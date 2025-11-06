"""Tests for the VennAbers calibrator helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibrated_explanations.core import venn_abers as venn_abers_module
from src.calibrated_explanations.core.venn_abers import VennAbers


class _StubVennAbers:
    """Lightweight stand-in for :mod:`venn_abers`' VennAbers implementation."""

    def fit(self, probs, targets, precision=4):  # noqa: ARG002 - interface compatibility
        # Store the shapes to mimic basic stateful behaviour used by the calibrator.
        self._last_fit = (np.asarray(probs).shape, np.asarray(targets).shape)
        return self

    def predict_proba(self, probs):
        probs = np.asarray(probs)
        if probs.size == 0:
            return (None, np.empty((0, 2)))
        positive = probs[:, 1]
        calibrated = np.column_stack([1 - positive, positive])
        return (None, calibrated)


class _DummyLearner:
    """Simple learner returning deterministic probability tables."""

    def __init__(self, base_probs: np.ndarray):
        self._base_probs = np.asarray(base_probs, dtype=float)

    def predict_proba(self, x, bins=None):  # noqa: ARG002 - signature compatible with scikit-learn
        n_samples = len(x)
        repeats = int(np.ceil(n_samples / len(self._base_probs)))
        tiled = np.vstack([self._base_probs] * repeats)
        return tiled[:n_samples]


class _DifficultyEstimator:
    """Difficulty estimator returning a deterministic vector."""

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=float)

    def apply(self, x):  # noqa: ARG002 - interface compatibility
        n_samples = len(x)
        repeats = int(np.ceil(n_samples / len(self._values)))
        tiled = np.concatenate([self._values] * repeats)
        return tiled[:n_samples]


@pytest.fixture(autouse=True)
def patch_venn_abers(monkeypatch):
    """Replace the external dependency with a lightweight stub for predictable behaviour."""

    monkeypatch.setattr(venn_abers_module.va, "VennAbers", _StubVennAbers)


def test_predict_with_difficulty_and_class_selection():
    """The calibrator should scale probabilities when a difficulty estimator is provided."""

    x_cal = np.array([[0.0], [1.0], [2.0]])
    y_cal = np.array(["cat", "dog", "mouse"])
    learner = _DummyLearner(
        np.array(
            [
                [0.60, 0.25, 0.15],
                [0.20, 0.55, 0.25],
                [0.25, 0.35, 0.40],
            ]
        )
    )
    difficulty = _DifficultyEstimator(np.array([0.2, 0.5, 0.7]))

    calibrator = VennAbers(x_cal, y_cal, learner, difficulty_estimator=difficulty)

    raw_probs = learner.predict_proba(x_cal)
    assert not np.allclose(calibrator.cprobs, raw_probs)

    predictions = calibrator.predict(np.array([[3.0], [4.0]]))
    assert predictions.shape == (2,)

    probs, selected_classes = calibrator.predict_proba(np.array([[5.0], [6.0]]), classes=1)
    assert probs.shape == (2, 3)
    assert selected_classes == [1]
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    interval_probs = calibrator.predict_proba(np.array([[7.0], [8.0]]), output_interval=True)
    calibrated, low, high, inferred_classes = interval_probs
    assert calibrated.shape == (2, 3)
    assert low.shape == high.shape == (2, 3)
    assert len(inferred_classes) == 2

    class_intervals = calibrator.predict_proba(
        np.array([[9.0], [10.0]]), classes=[0, 2], output_interval=True
    )
    cal_probs, low_bounds, high_bounds, returned_classes = class_intervals
    assert cal_probs.shape == (2, 3)
    assert returned_classes == [0, 2]
    assert len(low_bounds) == len(high_bounds) == 2


def test_predict_proba_requires_bins_for_mondrian_multiclass():
    """A Mondrian multi-class calibrator requires bins at prediction time."""

    x_cal = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_cal = np.array(["a", "b", "c", "a", "b", "c"])
    bins = np.array([0, 0, 1, 1, 0, 1])
    learner = _DummyLearner(
        np.array(
            [
                [0.55, 0.25, 0.20],
                [0.30, 0.40, 0.30],
                [0.25, 0.35, 0.40],
            ]
        )
    )

    calibrator = VennAbers(x_cal, y_cal, learner, bins=bins)

    test_bins = np.array([0, 1])
    with pytest.raises(ValueError, match="bins must be provided if Mondrian"):
        calibrator.predict_proba(np.array([[1.0], [2.0]]))

    probs, predicted_classes = calibrator.predict_proba(np.array([[1.0], [2.0]]), bins=test_bins)
    assert probs.shape == (2, 3)
    assert predicted_classes.shape == (2,)

    interval = calibrator.predict_proba(
        np.array([[1.5], [2.5]]), bins=test_bins, output_interval=True
    )
    cal_probs, low, high, classes = interval
    assert cal_probs.shape == (2, 3)
    assert low.shape == high.shape == (2, 3)
    assert classes.shape == (2,)


def test_predict_proba_requires_bins_for_mondrian_binary():
    """A Mondrian binary calibrator also requires bins at prediction time."""

    x_cal = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_cal = np.array([0, 1, 0, 1])
    bins = np.array([0, 0, 1, 1])
    learner = _DummyLearner(np.array([[0.7, 0.3], [0.4, 0.6]]))

    calibrator = VennAbers(x_cal, y_cal, learner, bins=bins)

    test_bins = np.array([0, 1])
    with pytest.raises(ValueError, match="bins must be provided if Mondrian"):
        calibrator.predict_proba(np.array([[1.0], [2.0]]))

    probs = calibrator.predict_proba(np.array([[1.0], [2.0]]), bins=test_bins)
    assert probs.shape == (2, 2)

    interval = calibrator.predict_proba(
        np.array([[1.5], [2.5]]), bins=test_bins, output_interval=True
    )
    calibrated, low, high = interval
    assert calibrated.shape == (2, 2)
    assert low.shape == high.shape == (2,)


def test_binary_predict_rounds_probabilities():
    """Binary predictions round the calibrated probabilities."""

    x_cal = np.array([[0.0], [1.0], [2.0]])
    y_cal = np.array([0, 1, 0])
    learner = _DummyLearner(np.array([[0.6, 0.4], [0.35, 0.65], [0.45, 0.55]]))

    calibrator = VennAbers(x_cal, y_cal, learner)

    predictions = calibrator.predict(np.array([[3.0], [4.0], [5.0]]))
    assert predictions.shape == (3,)
    assert set(np.unique(predictions)).issubset({0, 1})

    probs = calibrator.predict_proba(np.array([[6.0], [7.0]]))
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    interval = calibrator.predict_proba(np.array([[6.5], [7.5]]), output_interval=True)
    calibrated, low, high = interval
    assert calibrated.shape == (2, 2)
    assert low.shape == high.shape == (2,)


def test_predict_function_without_bins_argument_scales_binary_difficulty():
    """Binary calibration uses the difficulty estimator when bins are unsupported."""

    x_cal = np.array([[0.0], [1.0], [2.0]])
    y_cal = np.array([0, 1, 0])

    def predict_without_bins(x):
        base = np.array([[0.55, 0.45], [0.40, 0.60], [0.35, 0.65]])
        repeats = int(np.ceil(len(x) / len(base)))
        tiled = np.vstack([base] * repeats)
        return tiled[: len(x)]

    difficulty = _DifficultyEstimator(np.array([0.3, 0.6, 0.2]))

    calibrator = VennAbers(
        x_cal,
        y_cal,
        learner=_DummyLearner(np.array([[0.6, 0.4], [0.4, 0.6]])),
        difficulty_estimator=difficulty,
        predict_function=predict_without_bins,
    )

    baseline = predict_without_bins(x_cal)
    assert not np.allclose(calibrator.cprobs, baseline)

    probs = calibrator.predict_proba(np.array([[3.0], [4.0]]))
    assert probs.shape == (2, 2)
