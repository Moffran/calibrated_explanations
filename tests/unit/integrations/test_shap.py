"""Tests for the SHAP integration helper."""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.integrations import shap as shap_module
from calibrated_explanations.integrations.shap import ShapHelper


class DummyExplainer:
    """Lightweight explainer stub for exercising ShapHelper."""

    def __init__(self, x_cal):
        self.x_cal = x_cal
        self.feature_names = ["f1", "f2"]
        self.predict_calls: list[np.ndarray] = []

        class PredOrch:
            def __init__(self, outer):
                self.outer = outer

            def predict_internal(self, x, **_kw):
                array = np.asarray(x)
                self.outer.predict_calls.append(array)
                # ShapHelper expects tuple-like; index [0] is the prediction array.
                return (array,)

        self.prediction_orchestrator = PredOrch(self)


class FakeExplanation:
    """Simple container mirroring SHAP explanation shape."""

    def __init__(self, data):
        self.shape = getattr(data, "shape", (len(data),))
        self.data = data


class FakeShap:
    """Minimal SHAP stand-in with a callable Explainer."""

    class Explainer:
        def __init__(self, predict_fn, x_cal, feature_names):
            self.predict_fn = predict_fn
            self.x_cal = x_cal
            self.feature_names = feature_names
            self.calls: list[np.ndarray] = []

        def __call__(self, x):
            prediction = self.predict_fn(x)
            self.calls.append(prediction)
            return FakeExplanation(prediction)




def test_should_disable_helper_when_safe_import_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """ImportError disables the helper and returns empty result."""
    helper = ShapHelper(explainer=DummyExplainer(np.ones((1, 2))))
    monkeypatch.setattr(
        shap_module, "safe_import", lambda _name: (_ for _ in ()).throw(ImportError("missing"))
    )

    explainer, reference = helper.preload()

    assert explainer is None
    assert reference is None
    assert helper.enabled is False






def test_should_build_and_cache_explainer_when_shap_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path should build and reuse the cached explainer."""
    x_cal = np.arange(4, dtype=float).reshape(2, 2)
    helper = ShapHelper(explainer=DummyExplainer(x_cal))
    fake_shap = FakeShap()
    monkeypatch.setattr(shap_module, "safe_import", lambda _name: fake_shap)

    explainer, reference = helper.preload(num_test=1)

    assert isinstance(explainer, FakeShap.Explainer)
    assert isinstance(reference, FakeExplanation)
    assert reference.shape[0] == 1
    assert helper.enabled is True

    # Subsequent call with matching shape should reuse the cache
    monkeypatch.setattr(
        shap_module, "safe_import", lambda _name: (_ for _ in ()).throw(AssertionError)
    )
    cached_same_shape = helper.preload(num_test=1)
    cached_default = helper.preload()

    assert cached_same_shape == (explainer, reference)
    assert cached_default == (explainer, reference)


@pytest.mark.parametrize(
    "x_cal",
    [
        None,
        np.empty((0, 2)),
        object(),
    ],
)
def test_preload_returns_none_for_unusable_calibration_data(
    monkeypatch: pytest.MonkeyPatch,
    x_cal,
) -> None:
    """Unusable calibration payloads should short-circuit without building SHAP explainers."""
    helper = ShapHelper(explainer=DummyExplainer(x_cal))
    monkeypatch.setattr(shap_module, "safe_import", lambda _name: FakeShap())

    explainer, reference = helper.preload()

    assert explainer is None
    assert reference is None
    assert helper.enabled is False


def test_legacy_aliases_delegate_to_enabled_state() -> None:
    helper = ShapHelper(explainer=DummyExplainer(np.ones((1, 2))))

    assert helper.isenabled() is False

    helper.setenabled(True)
    assert helper.is_enabled() is True
    assert helper.isenabled() is True

    helper.setenabled(False)
    assert helper.is_enabled() is False
