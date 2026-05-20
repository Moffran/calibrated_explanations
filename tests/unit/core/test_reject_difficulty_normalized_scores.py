"""Focused unit tests for experimental difficulty-normalized reject scoring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.core.reject import orchestrator as orch
from calibrated_explanations.core.reject.orchestrator import default_score_cal, default_score_test
from calibrated_explanations.utils.exceptions import ValidationError


class DifficultyEstimator:
    def __init__(self, values):
        self.values = values

    def apply(self, x):
        return self.values


class RecordingConformalClassifier:
    def __init__(self):
        self.fit_alphas = None
        self.predict_p_alphas = None

    def fit(self, *, alphas, bins=None):
        self.fit_alphas = np.asarray(alphas, dtype=float)
        return self

    def predict_p(self, alphas, **kwargs):
        self.predict_p_alphas = np.asarray(alphas, dtype=float)
        return np.ones_like(self.predict_p_alphas, dtype=float)


class IntervalLearnerStub:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, x, bins=None):
        return np.array(self.proba, copy=True)


def make_explainer(proba, labels, difficulty_values=None):
    explainer = SimpleNamespace(
        mode="classification",
        x_cal=np.array([[0.0], [1.0]], dtype=float),
        y_cal=np.asarray(labels, dtype=int),
        bins=None,
        interval_learner=IntervalLearnerStub(proba),
        reject_learner=None,
        difficulty_estimator=(
            None if difficulty_values is None else DifficultyEstimator(difficulty_values)
        ),
        seed=7,
    )
    explainer.is_multiclass = lambda: False
    setattr(explainer, "_reject_difficulty_normalized", True)
    return explainer


def test_should_return_unchanged_calibration_scores_when_difficulty_estimator_is_missing(
    monkeypatch,
):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    expected = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, expected)


def test_should_return_unchanged_test_scores_when_difficulty_estimator_is_missing(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels)
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orchestrator = orch.RejectOrchestrator(explainer)
    orchestrator.initialize_reject_learner(ncf="default", w=0.5)
    orchestrator.predict_reject_breakdown(explainer.x_cal, confidence=0.9)
    expected = default_score_test(proba, "hinge")

    np.testing.assert_allclose(conformal.predict_p_alphas, expected)


def test_should_lower_calibration_scores_when_difficulty_increases(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, 2.0])
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, base_scores / np.array([1.0, 2.0]))
    assert conformal.fit_alphas[1] < base_scores[1]


def test_should_normalize_test_scores_row_wise_when_difficulty_varies_by_instance(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[2.0, 4.0])
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orchestrator = orch.RejectOrchestrator(explainer)
    orchestrator.initialize_reject_learner(ncf="default", w=0.5)
    orchestrator.predict_reject_breakdown(explainer.x_cal, confidence=0.9)
    base_scores = default_score_test(proba, "hinge")

    expected = base_scores / np.array([[2.0], [4.0]])
    np.testing.assert_allclose(conformal.predict_p_alphas, expected)


def test_should_raise_validation_error_when_difficulty_shape_is_invalid(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[[1.0, 2.0, 3.0]])
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)

    with pytest.raises(ValidationError, match="one value per instance"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


@pytest.mark.parametrize("values", ([1.0, np.nan], [1.0, np.inf]))
def test_should_raise_validation_error_when_difficulty_values_are_not_finite(monkeypatch, values):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=values)
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)

    with pytest.raises(ValidationError, match="difficulty values must be finite"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


def test_should_raise_validation_error_when_difficulty_values_are_negative(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[1.0, -0.5])
    monkeypatch.setattr(orch, "ConformalClassifier", RecordingConformalClassifier)

    with pytest.raises(ValidationError, match="difficulty values must be non-negative"):
        orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)


def test_should_clip_near_zero_difficulty_when_normalizing_scores(monkeypatch):
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    labels = np.array([1, 0])
    explainer = make_explainer(proba, labels, difficulty_values=[0.0, 1e-12])
    conformal = RecordingConformalClassifier()
    monkeypatch.setattr(orch, "ConformalClassifier", lambda: conformal)

    orch.RejectOrchestrator(explainer).initialize_reject_learner(ncf="default", w=0.5)
    base_scores = default_score_cal(proba, np.unique(labels), labels, "hinge")

    np.testing.assert_allclose(conformal.fit_alphas, base_scores / np.array([1e-6, 1e-6]))
