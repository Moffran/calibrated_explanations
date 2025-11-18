"""Focused tests covering LIME and SHAP helper paths for ``CalibratedExplainer``."""

from __future__ import annotations

import sys
import types

import numpy as np

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.integrations import LimeHelper, ShapHelper


class _StubLearner:
    """Minimal learner exposing ``predict_proba`` used during LIME preload."""

    def predict_proba(self, x):
        x = np.asarray(x)
        n = x.shape[0]
        positive = 0.2 + 0.05 * np.arange(n)
        positive = np.clip(positive, 0.05, 0.95)
        return np.column_stack([1 - positive, positive])


class _StubLimeExplanation:
    def __init__(self, weights, proba):
        self.local_exp = {1: weights}
        self.predict_proba = proba


class _StubLimeTabularExplainer:
    """Lightweight drop-in replacement for ``LimeTabularExplainer``."""

    def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - exercised indirectly
        pass

    def explain_instance(self, instance, predict_fn, num_features):
        arr = np.asarray(instance, dtype=float)
        proba = np.asarray(predict_fn(arr.reshape(1, -1))).squeeze()
        scale = float(proba[-1]) if proba.size else 0.0
        weights = [(idx, scale * (idx + 1) / (num_features + 1)) for idx in range(num_features)]
        return _StubLimeExplanation(weights, proba)


class _RecordingShapExplainer:
    """Collects construction/call details to assert SHAP preload toggles."""

    instances: list["_RecordingShapExplainer"] = []

    def __init__(self, f, data, feature_names=None) -> None:  # pragma: no cover - indirect
        self.f = f
        self.data = np.asarray(data)
        self.feature_names = feature_names
        self.calls: list[np.ndarray] = []
        _RecordingShapExplainer.instances.append(self)

    def __call__(self, x):
        arr = np.asarray(x)
        self.calls.append(arr)
        # Exercise the wrapped prediction function to mimic SHAP behaviour.
        self.f(arr)
        return np.full((arr.shape[0], self.data.shape[1]), 0.5)


def _register_stub_libraries(monkeypatch):
    lime_module = types.ModuleType("lime")
    lime_tabular = types.ModuleType("lime.lime_tabular")
    lime_tabular.LimeTabularExplainer = _StubLimeTabularExplainer
    lime_module.lime_tabular = lime_tabular

    shap_module = types.ModuleType("shap")
    shap_module.Explainer = _RecordingShapExplainer

    monkeypatch.setitem(sys.modules, "lime", lime_module)
    monkeypatch.setitem(sys.modules, "lime.lime_tabular", lime_tabular)
    monkeypatch.setitem(sys.modules, "shap", shap_module)


def _make_stub_explainer() -> CalibratedExplainer:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer.mode = "classification"
    x_cal = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]], dtype=float)
    y_cal = np.array([0.0, 1.0, 0.5], dtype=float)
    explainer.x_cal = x_cal
    explainer.y_cal = y_cal
    explainer._feature_names = ["f0", "f1"]
    explainer.bins = None
    explainer.learner = _StubLearner()
    explainer.latest_explanation = None
    explainer.feature_values = []
    explainer.categorical_features = []
    explainer.categorical_labels = {}
    explainer._CalibratedExplainer__initialized = True
    explainer._lime_helper = LimeHelper(explainer)
    explainer._shap_helper = ShapHelper(explainer)
    
    # Initialize the prediction orchestrator
    from calibrated_explanations.core.prediction import PredictionOrchestrator
    explainer._prediction_orchestrator = PredictionOrchestrator(explainer)
    explainer.interval_learner = None

    def _predict_stub(self, x, **_kwargs):
        x = np.asarray(x)
        n = x.shape[0]
        base = 0.2 + 0.1 * np.arange(n)
        low = base - 0.05
        high = base + 0.05
        return base, low, high, np.zeros(n, dtype=int)

    def _return_false(self):
        return False

    explainer._predict = types.MethodType(_predict_stub, explainer)
    explainer._is_mondrian = types.MethodType(_return_false, explainer)
    explainer.is_multiclass = types.MethodType(_return_false, explainer)
    explainer.is_fast = types.MethodType(_return_false, explainer)

    return explainer


def test_explain_lime_populates_fast_collection(monkeypatch):
    _register_stub_libraries(monkeypatch)
    explainer = _make_stub_explainer()

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


def test_preload_shap_refreshes_when_shape_changes(monkeypatch):
    _register_stub_libraries(monkeypatch)
    _RecordingShapExplainer.instances.clear()
    explainer = _make_stub_explainer()

    shap_1, shap_exp_1 = explainer._preload_shap()

    assert explainer._is_shap_enabled() is True
    assert shap_exp_1.shape == (1, explainer.num_features)
    assert len(_RecordingShapExplainer.instances) == 1

    shap_2, shap_exp_2 = explainer._preload_shap()

    assert shap_2 is shap_1
    assert shap_exp_2 is shap_exp_1

    shap_3, shap_exp_3 = explainer._preload_shap(num_test=2)

    assert shap_3 is not shap_1
    assert shap_exp_3.shape == (2, explainer.num_features)
    assert len(_RecordingShapExplainer.instances) == 2
