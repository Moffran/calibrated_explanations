"""Tests for controlled preprocessing wiring in WrapCalibratedExplainer.

These tests use simple stubs and monkeypatching to validate that when a
user-supplied preprocessor is provided via ExplainerConfig, it's used to:
- fit/transform training data before learner.fit
- fit/transform (or transform) calibration data before CalibratedExplainer
- transform inference data before explain_* calls

When no preprocessor is provided, behavior is unchanged (covered by existing tests).
"""

from __future__ import annotations

import numpy as np

from calibrated_explanations.api.config import ExplainerConfig
from calibrated_explanations.core import wrap_explainer as we


class DummyPreprocessor:
    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor
        self.fitted = False

    def fit_transform(self, X):
        self.fitted = True
        return np.asarray(X) * self.factor

    def transform(self, X):
        assert self.fitted
        return np.asarray(X) * self.factor


class StubModel:
    def __init__(self) -> None:
        self.last_fit_X = None
        self.fitted_ = False  # sklearn-style fitted marker

    # provide predict_proba so wrapper picks classification mode
    def predict_proba(self, X):  # pragma: no cover - not used
        return np.zeros((len(X), 2))

    def predict(self, X):  # minimal implementation for validation
        return np.zeros(len(X))

    def fit(self, X, y, **kwargs):
        self.last_fit_X = np.asarray(X)
        self.fitted_ = True


def test_preprocessor_applied_on_fit():
    model = StubModel()
    pre = DummyPreprocessor(factor=2.5)
    cfg = ExplainerConfig(model=model, preprocessor=pre)
    w = we.WrapCalibratedExplainer._from_config(cfg)

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    w.fit(X, y)

    assert model.last_fit_X is not None
    np.testing.assert_allclose(model.last_fit_X, X * 2.5)


def test_preprocessor_applied_on_calibrate_and_inference(monkeypatch):
    model = StubModel()
    pre = DummyPreprocessor(factor=3.0)
    cfg = ExplainerConfig(model=model, preprocessor=pre)
    w = we.WrapCalibratedExplainer._from_config(cfg)
    w.fitted = True

    captured = {}

    class DummyCE:
        def __init__(self, learner, X_cal, y_cal, **kwargs):  # noqa: D401
            captured["X_cal"] = np.asarray(X_cal)

        def explain_factual(self, X, **kwargs):  # noqa: D401
            captured["X_test"] = np.asarray(X)
            return np.asarray(X)

    monkeypatch.setattr(we, "CalibratedExplainer", DummyCE)

    X_cal = np.array([[1.0, 1.0], [2.0, 2.0]])
    y_cal = np.array([0, 1])
    w.calibrate(X_cal, y_cal)
    np.testing.assert_allclose(captured["X_cal"], X_cal * 3.0)

    # Use the installed DummyCE to check inference transform
    X_test = np.array([[4.0, 5.0]])
    out = w.explain_factual(X_test)
    np.testing.assert_allclose(out, X_test * 3.0)
