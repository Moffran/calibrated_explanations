"""Unit tests covering CalibratedExplainer runtime helper utilities."""

from __future__ import annotations


import numpy as np

from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
)


def stub_explainer(explainer_factory, mode: str = "classification") -> CalibratedExplainer:
    """Construct a fully initialized explainer instance for unit tests."""

    explainer = explainer_factory(mode=mode)
    explainer.bins = None
    explainer.plot_style_override = None
    explainer.interval_plugin_override = None
    explainer.fast_interval_plugin_override = None
    explainer.interval_plugin_hints = {}
    explainer.interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer.interval_preferred_identifier = {"default": None, "fast": None}
    explainer.telemetry_interval_sources = {"default": None, "fast": None}
    explainer.plugin_manager.interval_context_metadata = {"default": {}, "fast": {}}
    explainer.explanation_plugin_overrides = {
        key: None for key in ("factual", "alternative", "fast")
    }
    explainer.pyproject_explanations = {}
    explainer.pyproject_intervals = {}
    explainer.pyproject_plots = {}
    explainer.plot_style_override = None
    explainer.plugin_manager.explanation_plugin_fallbacks = {}
    return explainer


def testinstantiate_plugin_handles_multiple_paths(monkeypatch, explainer_factory):
    explainer = stub_explainer(explainer_factory)

    class CallableWithMeta:
        plugin_meta = {}

        def __call__(self):  # pragma: no cover - guard against accidental call
            raise AssertionError("should not be invoked")

    callable_with_meta = CallableWithMeta()
    assert explainer.instantiate_plugin(callable_with_meta) is callable_with_meta

    class SimplePlugin:
        def __init__(self) -> None:
            self.token = object()

    prototype = SimplePlugin()
    clone = explainer.instantiate_plugin(prototype)
    assert isinstance(clone, SimplePlugin)
    assert clone is not prototype

    class BrokenPlugin:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    broken = BrokenPlugin.__new__(BrokenPlugin)
    sentinel = object()
    # Patch copy.deepcopy in the orchestrator module where it's imported
    from calibrated_explanations.core.explain import orchestrator as explain_orch

    monkeypatch.setattr(explain_orch.copy, "deepcopy", lambda value: sentinel)
    assert explainer.instantiate_plugin(broken) is sentinel

    # Test fallback when deepcopy itself fails
    def raising_deepcopy(value):
        raise RuntimeError("fail")

    monkeypatch.setattr(explain_orch.copy, "deepcopy", raising_deepcopy)
    assert explainer.instantiate_plugin(broken) is broken


class RaisingInterval:
    def predict_uncertainty(self, *_, **__):
        raise RuntimeError("boom")

    def predict_probability(self, *_, **__):
        raise RuntimeError("boom")


def test_predict_impl_returns_degraded_arrays_when_suppressed(explainer_factory):
    explainer = stub_explainer(explainer_factory, mode="regression")
    explainer.initialized = True
    explainer.fast = False
    explainer.interval_learner = RaisingInterval()
    explainer.suppress_crepes_errors = True
    explainer.x_cal = np.zeros((1, 1))

    x = np.ones((2, 1))

    predict, low, high, classes = explainer.prediction_orchestrator.predict_impl(x)

    assert np.all(predict == 0)
    assert np.all(low == 0)
    assert np.all(high == 0)
    assert classes is None

    threshold = [0.5, 0.5]
    predict_t, low_t, high_t, classes_t = explainer.prediction_orchestrator.predict_impl(
        x, threshold=threshold
    )

    assert np.all(predict_t == 0)
    assert np.all(low_t == 0)
    assert np.all(high_t == 0)
    assert classes_t is None


class ListInterval:
    def __init__(self, proba=None, low=None, high=None):
        self.proba = proba
        self.low = low
        self.high = high

    def predict_probability(self, x, *_args, **_kwargs):
        return self.proba, self.low, self.high, None

    def predict_proba(self, x, *_args, **_kwargs):
        return self.proba, self.low, self.high, None
