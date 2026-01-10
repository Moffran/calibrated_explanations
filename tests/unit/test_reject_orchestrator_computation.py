import numpy as np
import pytest

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator


class DummyConformal:
    def __init__(self):
        self.fitted = False

    def fit(self, alphas=None, bins=None):
        self.fitted = True
        return self

    def predict_set(self, alphas_test, classes_array, confidence=0.95):
        # Return a prediction set of shape (n_samples, n_classes)
        n = len(alphas_test)
        k = len(np.atleast_1d(classes_array))
        # For simplicity, mark all as singleton on first class
        out = np.zeros((n, k), dtype=bool)
        out[:, 0] = True
        return out


def fake_hinge(proba, unique_classes=None, classes=None):
    # produce an alpha array of length n_samples
    proba = np.asarray(proba)
    if proba.ndim == 2:
        return np.ones(len(proba)) * 0.5
    return np.ones(proba.shape[0]) * 0.5


class DummyIntervalLearner:
    def __init__(self, proba, classes=None):
        self.proba = np.asarray(proba)
        self.classes = np.asarray(classes) if classes is not None else None

    def predict_proba(self, x, bins=None):
        # return proba, classes
        n = len(x)
        if self.classes is None:
            # return simple 2-class probs
            proba = np.tile(np.array([0.3, 0.7]), (n, 1))
            classes = np.zeros(n, dtype=int)
            return proba, classes
        return self.proba, self.classes

    def predict_probability(self, x, y_threshold=None, bins=None):
        # regression style returns proba_1, _, _, _
        n = len(x)
        return np.ones(n) * 0.2, None, None, None


class DummyExplainer:
    def __init__(self, x_cal, y_cal, mode="classification", is_multiclass=False):
        self.x_cal = np.asarray(x_cal)
        self.y_cal = np.asarray(y_cal)
        self.mode = mode
        self.bins = None
        self.interval_learner = None
        self.reject_threshold = None
        self.reject_learner = None
        self.is_multiclass_flag = is_multiclass

    def is_multiclass(self):
        return bool(self.is_multiclass_flag)

    def predict(self, x, **kwargs):
        # simple predict returning zeros
        return np.zeros(len(x), dtype=int)


def test_initialize_and_predict_reject_multiclass(monkeypatch):
    # Monkeypatch ConformalClassifier and hinge
    import calibrated_explanations.core.reject.orchestrator as ro_mod

    monkeypatch.setattr(ro_mod, "ConformalClassifier", DummyConformal)
    monkeypatch.setattr(ro_mod, "hinge", fake_hinge)

    x_cal = [[1], [2], [3]]
    y_cal = np.array([0, 1, 1])
    expl = DummyExplainer(x_cal, y_cal, mode="classification")
    expl.is_multiclass_flag = True
    # prepare interval learner to return per-sample class indices matching y_cal
    proba = np.tile(np.array([0.2, 0.8]), (3, 1))
    classes = np.array([0, 1, 1])
    expl.interval_learner = DummyIntervalLearner(proba, classes)

    orch = RejectOrchestrator(expl)
    learner = orch.initialize_reject_learner()
    assert learner is not None
    assert expl.reject_learner is not None

    # assign the fitted learner for predict_reject and adjust interval learner
    expl.reject_learner = DummyConformal()
    # ensure interval_learner returns per-sample probs for the predict call
    proba2 = np.tile(np.array([0.4, 0.6]), (2, 1))
    classes2 = np.array([0, 1])
    expl.interval_learner = DummyIntervalLearner(proba2, classes2)
    rejected, error_rate, reject_rate = orch.predict_reject(x=[[10], [20]], bins=None, confidence=0.95)
    assert isinstance(rejected, np.ndarray)
    assert rejected.shape[0] == 2


def test_initialize_reject_learner_regression(monkeypatch):
    import calibrated_explanations.core.reject.orchestrator as ro_mod

    monkeypatch.setattr(ro_mod, "ConformalClassifier", DummyConformal)
    monkeypatch.setattr(ro_mod, "hinge", fake_hinge)

    x_cal = [[1], [2], [3], [4]]
    y_cal = np.array([0.1, 0.4, 0.6, 0.9])
    expl = DummyExplainer(x_cal, y_cal, mode="regression")
    expl.is_multiclass_flag = False
    expl.bins = None
    expl.interval_learner = DummyIntervalLearner(None)

    orch = RejectOrchestrator(expl)
    # call initialize with threshold
    learner = orch.initialize_reject_learner(threshold=0.5)
    assert learner is not None
    assert expl.reject_learner is not None
