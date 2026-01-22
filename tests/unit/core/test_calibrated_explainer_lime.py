"""Focused tests covering LIME and SHAP helper paths for ``CalibratedExplainer``."""

from __future__ import annotations

import sys
import types

import numpy as np

from calibrated_explanations.explanations import CalibratedExplanations


class StubLearner:
    """Minimal learner exposing ``predict_proba`` used during LIME preload."""

    def predict_proba(self, x):
        x = np.asarray(x)
        n = x.shape[0]
        positive = 0.2 + 0.05 * np.arange(n)
        positive = np.clip(positive, 0.05, 0.95)
        return np.column_stack([1 - positive, positive])


class StubLimeExplanation:
    def __init__(self, weights, proba):
        self.local_exp = {1: weights}
        self.predict_proba = proba


class StubLimeTabularExplainer:
    """Lightweight drop-in replacement for ``LimeTabularExplainer``."""

    def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - exercised indirectly
        pass

    def explain_instance(self, instance, predict_fn, num_features):
        arr = np.asarray(instance, dtype=float)
        proba = np.asarray(predict_fn(arr.reshape(1, -1))).squeeze()
        scale = float(proba[-1]) if proba.size else 0.0
        weights = [(idx, scale * (idx + 1) / (num_features + 1)) for idx in range(num_features)]
        return StubLimeExplanation(weights, proba)


class RecordingShapExplainer:
    """Collects construction/call details to assert SHAP preload toggles."""

    instances: list["RecordingShapExplainer"] = []

    def __init__(self, f, data, feature_names=None) -> None:  # pragma: no cover - indirect
        self.f = f
        self.data = np.asarray(data)
        self.feature_names = feature_names
        self.calls: list[np.ndarray] = []
        RecordingShapExplainer.instances.append(self)

    def __call__(self, x):
        arr = np.asarray(x)
        self.calls.append(arr)
        # Exercise the wrapped prediction function to mimic SHAP behaviour.
        self.f(arr)
        return np.full((arr.shape[0], self.data.shape[1]), 0.5)


def register_stub_libraries(monkeypatch):
    lime_module = types.ModuleType("lime")
    lime_tabular = types.ModuleType("lime.lime_tabular")
    lime_tabular.LimeTabularExplainer = StubLimeTabularExplainer
    lime_module.lime_tabular = lime_tabular

    shap_module = types.ModuleType("shap")
    shap_module.Explainer = RecordingShapExplainer

    monkeypatch.setitem(sys.modules, "lime", lime_module)
    monkeypatch.setitem(sys.modules, "lime.lime_tabular", lime_tabular)
    monkeypatch.setitem(sys.modules, "shap", shap_module)


def make_stub_explainer(explainer_factory):
    explainer = explainer_factory()
    explainer.mode = "classification"
    x_cal = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]], dtype=float)
    y_cal = np.array([0.0, 1.0, 0.5], dtype=float)
    explainer.x_cal = x_cal
    explainer.y_cal = y_cal
    explainer.feature_names = ["f0", "f1"]
    explainer.bins = None
    explainer.learner = StubLearner()
    explainer.latest_explanation = None
    explainer.feature_values = []
    explainer.categorical_features = []
    explainer.categorical_labels = {}
    explainer.initialized = True

    def predict_stub(self, x, **_kwargs):
        x = np.asarray(x)
        n = x.shape[0]
        base = 0.2 + 0.1 * np.arange(n)
        low = base - 0.05
        high = base + 0.05
        return base, low, high, np.zeros(n, dtype=int)

    def return_false(self):
        return False

    explainer.predict = types.MethodType(predict_stub, explainer)
    explainer.is_multiclass = types.MethodType(return_false, explainer)
    explainer.is_fast = types.MethodType(return_false, explainer)

    return explainer


def test_explain_lime_populates_fast_collection(monkeypatch, explainer_factory):
    register_stub_libraries(monkeypatch)
    explainer = make_stub_explainer(explainer_factory)

    x_test = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=float)

    result = explainer.explain_lime(x_test)

    assert isinstance(result, CalibratedExplanations)
    assert len(result.explanations) == x_test.shape[0]
    assert result.total_explain_time is not None

    for explanation in result.explanations:
        weights = explanation.feature_weights["predict"]
        preds = explanation.feature_predict["predict"]
        assert len(weights) == explainer.num_features
        assert len(preds) == explainer.num_features
        assert not np.allclose(weights, 0)
        assert not np.allclose(preds, 0)
        assert explanation.explain_time is not None
