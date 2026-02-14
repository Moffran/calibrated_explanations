"""Behavioural tests for the in-tree plugin adapters."""

from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest

from calibrated_explanations.core import (
    ConfigurationError,
)
from calibrated_explanations.plugins import builtins
from calibrated_explanations.plugins.explanations import (
    ExplanationBatch,
    ExplanationContext,
    ExplanationRequest,
)
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext
from tests.helpers.plugin_utils import make_plot_context


class SentinelExplainer:
    """Minimal stub exposing the interfaces exercised by the legacy bridge."""

    def __init__(self, prediction, *, calibrated_classes=None):
        self.prediction_data = prediction
        self.calibrated_classes_data = calibrated_classes or ["c0", "c1"]
        self.calls: list[tuple[str, tuple, dict]] = []

    def predict(self, *args, **kwargs):
        self.calls.append(("predict", args, kwargs))
        if kwargs.get("calibrated"):
            return self.calibrated_classes_data
        return self.prediction_data

    def predict_proba(self, *args, **kwargs):  # pragma: no cover - passthrough proxy
        self.calls.append(("predict_proba", args, kwargs))
        return (args, kwargs)


def make_interval_context(**overrides):
    base = {
        "learner": "learner",
        "calibration_splits": [("x_cal", "y_cal")],
        "bins": {"calibration": "bins"},
        "residuals": {},
        "difficulty": {"estimator": "difficulty"},
        "metadata": {},
        "fast_flags": {},
    }
    base.update(overrides)
    return IntervalCalibratorContext(**base)


def make_explanation_context(explainer, predict_bridge, **overrides):
    context = {
        "task": "classification",
        "mode": "factual",
        "feature_names": (),
        "categorical_features": (),
        "categorical_labels": {},
        "discretizer": None,
        "helper_handles": {"explainer": explainer},
        "predict_bridge": predict_bridge,
        "interval_settings": {},
        "plot_settings": {},
    }
    context.update(overrides)
    return ExplanationContext(**context)


def test_derive_threshold_labels_logs_interval_failure(caplog):
    caplog.set_level("DEBUG")
    labels = builtins.derive_threshold_labels(["bad", "value"])
    assert labels == ("Target within threshold", "Outside threshold")
    assert "Failed to parse threshold" in caplog.text


def test_execution_plugin_supports_false_falls_back_to_legacy(monkeypatch):
    class DummyExecutionPlugin:
        def supports(self, *_args, **_kwargs):
            return False

    class DummyExplanation:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    class DummyExplainer:
        def __init__(self):
            self.last_collection = None
            self.feature_filter_config = None

        def explain_factual(self, _x, **_kwargs):
            collection = types.SimpleNamespace(
                mode="factual",
                explanations=[DummyExplanation()],
            )
            self.last_collection = collection
            return collection

    def fake_build_plan(_explainer, _x, _request):
        explain_request = types.SimpleNamespace()
        explain_config = types.SimpleNamespace(executor=None)
        runtime = contextlib.nullcontext()
        return explain_request, explain_config, runtime

    def raise_cfg(_base):
        raise builtins.CalibratedError("filter config boom")

    monkeypatch.setattr(
        "calibrated_explanations.core.explain.parallel_runtime.build_explain_execution_plan",
        fake_build_plan,
    )
    monkeypatch.setattr(builtins.FeatureFilterConfig, "from_base_and_env", staticmethod(raise_cfg))

    explainer = DummyExplainer()
    plugin = builtins.SequentialExplanationPlugin()
    plugin.execution_plugin_class = DummyExecutionPlugin
    plugin.initialize(
        make_explanation_context(
            explainer=explainer, predict_bridge=builtins.LegacyPredictBridge(SentinelExplainer([1]))
        )
    )

    request = ExplanationRequest(
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=(1,),
        extras={},
        feature_filter_per_instance_ignore=((0, 2),),
    )

    with pytest.warns(UserWarning):
        batch = plugin.explain_batch(np.asarray([[1.0]]), request)

    assert isinstance(batch, ExplanationBatch)
    assert explainer.last_collection.feature_filter_per_instance_ignore == ((0, 2),)
    assert explainer.last_collection.explanations[0].reset_called is True
    assert "filter_error" in explainer.last_collection.filter_telemetry


def test_legacy_plot_builder_global_payload():
    builder = builtins.LegacyPlotBuilder()
    context = make_plot_context(
        intent={"type": "global"},
        options={"payload": {"x": [1], "y": [2], "threshold": 0.5}},
        show=True,
        path="out.png",
        save_ext=".png",
    )
    payload = builder.build(context)
    assert payload["legacy_function"] == "global"
    assert payload["x"] == [1]
    assert payload["y"] == [2]
    assert payload["threshold"] == 0.5
    assert payload["show"] is True
    assert payload["path"] == "out.png"
    assert payload["save_ext"] == ".png"


def test_legacy_plot_builder_rejects_bad_payload():
    builder = builtins.LegacyPlotBuilder()
    context = make_plot_context(intent={"type": "global"}, options={"payload": 1})
    with pytest.raises(ConfigurationError):
        builder.build(context)


def test_legacy_plot_renderer_invokes_global(monkeypatch):
    calls = {}

    def fake_plot_global(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(
        "calibrated_explanations.legacy.plotting.plot_global",
        fake_plot_global,
    )

    renderer = builtins.LegacyPlotRenderer()
    context = make_plot_context()
    artifact = {
        "legacy_function": "global",
        "explainer": "explainer",
        "x": [1],
        "y": [2],
        "threshold": 0.5,
        "show": False,
        "path": "out",
        "save_ext": ".png",
    }
    result = renderer.render(artifact, context=context)
    assert calls["explainer"] == "explainer"
    assert calls["x"] == [1]
    assert calls["y"] == [2]
    assert calls["threshold"] == 0.5
    assert calls["show"] is False
    assert calls["path"] == "out"
    assert calls["save_ext"] == ".png"
    assert result.saved_paths == ()
