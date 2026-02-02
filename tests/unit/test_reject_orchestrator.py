from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.utils.exceptions import ValidationError


class DummyIntervalLearner:
    def predict_proba(self, x, bins=None):
        proba = np.array([0.4, 0.6])
        return np.tile(proba, (len(x), 1))


class DummyRejectLearner:
    def __init__(self):
        self.seeds = []

    def predict_p(self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None):
        self.seeds.append(seed)
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.ones((n_rows, n_cols), dtype=float)

    def predict_set(self, alphas, bins=None, confidence=0.95, smoothing=True, seed=None):
        self.seeds.append(seed)
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.ones((n_rows, n_cols), dtype=bool)


class DummyExplainer:
    def __init__(self):
        self.mode = "classification"
        self.y_cal = np.array([0, 1])
        self.seed = 123
        self.interval_learner = DummyIntervalLearner()
        self.reject_learner = DummyRejectLearner()

    def is_multiclass(self):
        return False


def test_should_pass_seed_once_when_predicting_reject():
    explainer = DummyExplainer()
    orchestrator = RejectOrchestrator(explainer)

    orchestrator.predict_reject(np.zeros((3, 2)), confidence=0.9)

    assert explainer.reject_learner.seeds == [explainer.seed]


def test_should_raise_validation_error_when_calibration_set_is_invalid():
    explainer = DummyExplainer()
    orchestrator = RejectOrchestrator(explainer)

    with pytest.raises(ValidationError):
        orchestrator.initialize_reject_learner(calibration_set="invalid")


class DummyRejectLearnerMonotonic:
    def predict_p(self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None):
        n = len(alphas)
        num_classes = 3
        # p-values chosen so that as epsilon decreases (confidence increases),
        # membership expands and singleton count cannot increase.
        out = np.full((n, num_classes), 0.2, dtype=float)
        # first 9 rows: only one class has high p-value -> singleton at most eps thresholds
        for i in range(min(n, 9)):
            out[i, i % num_classes] = 0.99
        # last row: two classes have moderate p-values -> can become non-singleton at high confidence
        if n >= 10:
            out[9, 0] = 0.60
            out[9, 1] = 0.60
        return out


def test_should_have_monotonic_ambiguity_and_uncertainty_rates_when_confidence_changes():
    explainer = DummyExplainer()
    explainer.reject_learner = DummyRejectLearnerMonotonic()
    orchestrator = RejectOrchestrator(explainer)

    x = np.zeros((10, 2))
    ambiguity_rates = []
    uncertainty_rates = []
    for conf in (0.9, 0.95, 0.99):
        out = orchestrator.predict_reject_breakdown(x, confidence=conf)
        ambiguity_rates.append(out["ambiguity_rate"])
        uncertainty_rates.append(out["novelty_rate"])

    assert ambiguity_rates[0] <= ambiguity_rates[1] <= ambiguity_rates[2]
    assert uncertainty_rates[0] >= uncertainty_rates[1] >= uncertainty_rates[2]


class DummyRejectLearnerAlwaysEmpty:
    def predict_p(self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None):
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.zeros((n_rows, n_cols), dtype=float)


def test_should_reject_as_uncertainty_when_prediction_set_is_empty():
    explainer = DummyExplainer()
    explainer.reject_learner = DummyRejectLearnerAlwaysEmpty()
    orchestrator = RejectOrchestrator(explainer)

    x = np.zeros((5, 2))
    out = orchestrator.predict_reject_breakdown(x, confidence=0.9)

    assert np.asarray(out["rejected"], dtype=bool).tolist() == [True] * 5
    assert np.asarray(out["novelty"], dtype=bool).tolist() == [True] * 5
    assert out["novelty_rate"] == 1.0
