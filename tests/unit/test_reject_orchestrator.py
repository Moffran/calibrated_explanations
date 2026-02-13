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

    def predict_p(
        self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None
    ):
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




def test_should_raise_validation_error_when_calibration_set_is_invalid():
    explainer = DummyExplainer()
    orchestrator = RejectOrchestrator(explainer)

    with pytest.raises(ValidationError):
        orchestrator.initialize_reject_learner(calibration_set="invalid")


class DummyRejectLearnerMonotonic:
    def predict_p(
        self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None
    ):
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




class DummyRejectLearnerAlwaysEmpty:
    def predict_p(
        self, alphas, bins=None, all_classes=True, classes=None, y=None, smoothing=True, seed=None
    ):
        n_rows = len(alphas)
        n_cols = alphas.shape[1] if getattr(alphas, "ndim", 1) == 2 else 2
        return np.zeros((n_rows, n_cols), dtype=float)


