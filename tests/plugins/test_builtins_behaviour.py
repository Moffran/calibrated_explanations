"""Behavioural tests for the in-tree plugin adapters."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from calibrated_explanations.core import (
    ConfigurationError,
    NotFittedError,
)
from calibrated_explanations.plugins import builtins
from calibrated_explanations.plugins.explanations import (
    ExplanationBatch,
    ExplanationContext,
    ExplanationRequest,
)
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext
from calibrated_explanations.plugins.plots import PlotRenderContext


class _SentinelExplainer:
    """Minimal stub exposing the interfaces exercised by the legacy bridge."""

    def __init__(self, prediction, *, calibrated_classes=None):
        self._prediction = prediction
        self._calibrated_classes = calibrated_classes or ["c0", "c1"]
        self.calls: list[tuple[str, tuple, dict]] = []

    def predict(self, *args, **kwargs):
        self.calls.append(("predict", args, kwargs))
        if kwargs.get("calibrated"):
            return self._calibrated_classes
        return self._prediction

    def predict_proba(self, *args, **kwargs):  # pragma: no cover - passthrough proxy
        self.calls.append(("predict_proba", args, kwargs))
        return (args, kwargs)


def _make_interval_context(**overrides):
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


def _make_explanation_context(explainer, predict_bridge, **overrides):
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


def _make_plot_context(**overrides):
    context = {
        "explanation": None,
        "instance_metadata": {},
        "style": "plot_spec.default",
        "intent": {},
        "show": False,
        "path": None,
        "save_ext": None,
        "options": {},
    }
    context.update(overrides)
    return PlotRenderContext(**context)


def test_derive_threshold_labels_handles_sequences():
    labels = builtins._derive_threshold_labels([1, 3.75])
    assert labels == ("1.00 <= Y < 3.75", "Outside interval")


def test_derive_threshold_labels_handles_scalars_and_errors():
    sentinel = object()
    assert builtins._derive_threshold_labels(sentinel) == (
        "Target within threshold",
        "Outside threshold",
    )
    assert builtins._derive_threshold_labels(2.5) == ("Y < 2.50", "Y ≥ 2.50")


def test_derive_threshold_labels_logs_interval_failure(caplog):
    caplog.set_level("DEBUG")
    labels = builtins._derive_threshold_labels(["bad", "value"])
    assert labels == ("Target within threshold", "Outside threshold")
    assert "Failed to parse threshold" in caplog.text


def test_legacy_predict_bridge_handles_tuple_payload_for_classification():
    bridge = builtins.LegacyPredictBridge(
        _SentinelExplainer(([1.0, 2.0], ([0.1, 0.2], [0.3, 0.4])))
    )
    payload = bridge.predict(
        np.asarray([[1.0]]), mode="factual", task="classification", bins="hist"
    )
    assert payload["mode"] == "factual"
    np.testing.assert_array_equal(payload["predict"], [1.0, 2.0])
    np.testing.assert_array_equal(payload["low"], [0.1, 0.2])
    np.testing.assert_array_equal(payload["high"], [0.3, 0.4])
    np.testing.assert_array_equal(payload["classes"], ["c0", "c1"])


def test_legacy_predict_bridge_handles_scalar_prediction():
    bridge = builtins.LegacyPredictBridge(_SentinelExplainer([4.0, 5.0]))
    payload = bridge.predict(np.asarray([[1.0]]), mode="fast", task="regression")
    assert set(payload) == {"predict", "mode", "task"}
    np.testing.assert_array_equal(payload["predict"], [4.0, 5.0])


def test_legacy_predict_bridge_interval_and_proba():
    class Explainer:
        def predict(self, *args, **kwargs):
            return "interval"

        def predict_proba(self, *args, **kwargs):
            return "proba"

    bridge = builtins.LegacyPredictBridge(Explainer())
    assert bridge.predict_interval("x", task="classification") == "interval"
    assert bridge.predict_proba("x") == "proba"


def test_supports_calibrated_explainer_uses_safe_isinstance(monkeypatch):
    module = types.ModuleType("calibrated_explanations.core.calibrated_explainer")

    class DummyExplainer:  # pragma: no cover - structure only
        pass

    module.CalibratedExplainer = DummyExplainer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert builtins._supports_calibrated_explainer(DummyExplainer()) is True
    assert builtins._supports_calibrated_explainer(object()) is False


def test_collection_to_batch_preserves_metadata():
    class DummyCollection:
        explanations = [object()]
        mode = "factual"

    collection = DummyCollection()
    batch = builtins._collection_to_batch(collection)  # noqa: SLF001
    assert isinstance(batch, ExplanationBatch)
    assert batch.collection_metadata["container"] is collection
    assert batch.collection_metadata["mode"] == "factual"


def test_collection_to_batch_defaults_to_factual():
    class EmptyCollection:
        explanations: tuple = ()

    batch = builtins._collection_to_batch(EmptyCollection())  # noqa: SLF001
    assert batch.explanation_cls is builtins.FactualExplanation


def test_interval_calibrator_create_for_regression(monkeypatch):
    created_with = {}

    class DummyIntervalRegressor:
        def __init__(self, explainer):
            created_with["explainer"] = explainer

    interval_module = types.ModuleType("calibrated_explanations.calibration.interval_regressor")
    interval_module.IntervalRegressor = DummyIntervalRegressor
    monkeypatch.setitem(sys.modules, interval_module.__name__, interval_module)

    context = _make_interval_context(
        metadata={"task": "regression", "explainer": object()},
    )
    plugin = builtins.LegacyIntervalCalibratorPlugin()
    calibrator = plugin.create(context)
    assert isinstance(calibrator, DummyIntervalRegressor)
    assert context.metadata["calibrator"] is calibrator
    assert created_with["explainer"] is context.metadata["explainer"]


def test_interval_calibrator_create_for_classification(monkeypatch):
    created_args = {}

    class DummyVennAbers:
        def __init__(self, *args, **kwargs):
            created_args["args"] = args
            created_args["kwargs"] = kwargs

    venn_module = types.ModuleType("calibrated_explanations.calibration.venn_abers")
    venn_module.VennAbers = DummyVennAbers
    monkeypatch.setitem(sys.modules, venn_module.__name__, venn_module)

    class Explainer:
        predict_function = "sentinel"

    context = _make_interval_context(
        metadata={"task": "classification", "explainer": Explainer()},
    )
    plugin = builtins.LegacyIntervalCalibratorPlugin()
    calibrator = plugin.create(context)
    assert isinstance(calibrator, DummyVennAbers)
    assert created_args["args"][:4] == ("x_cal", "y_cal", "learner", "bins")
    assert created_args["kwargs"]["predict_function"] == "sentinel"


def test_interval_calibrator_requires_predict_function(monkeypatch):
    venn_module = types.ModuleType("calibrated_explanations.calibration.venn_abers")
    venn_module.VennAbers = object
    monkeypatch.setitem(sys.modules, venn_module.__name__, venn_module)

    context = _make_interval_context(
        metadata={"task": "classification"},
    )
    plugin = builtins.LegacyIntervalCalibratorPlugin()
    with pytest.raises(NotFittedError):
        plugin.create(context)


def test_interval_calibrator_requires_explainer_for_regression(monkeypatch):
    interval_module = types.ModuleType("calibrated_explanations.calibration.interval_regressor")
    interval_module.IntervalRegressor = object
    monkeypatch.setitem(sys.modules, interval_module.__name__, interval_module)

    context = _make_interval_context(metadata={"task": "regression"})
    plugin = builtins.LegacyIntervalCalibratorPlugin()
    with pytest.raises(NotFittedError):
        plugin.create(context)


def test_explanation_plugin_requires_initialisation():
    plugin = builtins.LegacyFactualExplanationPlugin()
    with pytest.raises(NotFittedError):
        plugin.explain_batch("x", ExplanationRequest(None, None, None, (), {}))


def test_explanation_initialise_requires_explainer():
    plugin = builtins.LegacyFactualExplanationPlugin()
    context = _make_explanation_context(
        explainer=None, predict_bridge=builtins.LegacyPredictBridge(_SentinelExplainer([1]))
    )
    with pytest.raises(NotFittedError):
        plugin.initialize(context)


def test_explanation_batch_adapts_legacy_collection(monkeypatch):
    bridge_calls = []

    class DummyBridge:
        def predict(self, *args, **kwargs):  # pragma: no cover - required by protocol
            bridge_calls.append((args, kwargs))
            return {"predict": np.array([1])}

    request = ExplanationRequest(
        threshold=0.5,
        low_high_percentiles=(0.1, 0.9),
        bins="bins",
        features_to_ignore=(1,),
        extras={},
    )

    class DummyExplainer:
        def __init__(self):
            self.calls = []

        def explain_factual(self, x, **kwargs):
            self.calls.append((x, kwargs))

            class DummyCollection:
                explanations = [object()]
                mode = "factual"

            return DummyCollection()

    explainer = DummyExplainer()
    plugin = builtins.LegacyFactualExplanationPlugin()
    context = _make_explanation_context(
        explainer=explainer, predict_bridge=DummyBridge(), task="classification"
    )
    plugin.initialize(context)
    batch = plugin.explain_batch("sample", request)
    assert isinstance(batch, ExplanationBatch)
    assert explainer.calls[0][1]["_use_plugin"] is False
    assert explainer.calls[0][1]["features_to_ignore"] == (1,)
    assert bridge_calls and bridge_calls[0][0][0] == "sample"


def test_explanation_supports_helpers(monkeypatch):
    module = types.ModuleType("calibrated_explanations.core.calibrated_explainer")

    class DummyExplainer:
        pass

    module.CalibratedExplainer = DummyExplainer
    monkeypatch.setitem(sys.modules, module.__name__, module)

    plugin = builtins.LegacyFactualExplanationPlugin()
    assert plugin.supports(DummyExplainer()) is True
    assert plugin.supports(object()) is False
    assert plugin.supports_mode("factual", task="classification") is True
    assert plugin.supports_mode("alternative", task="classification") is False


def test_explanation_rejects_unsupported_model(monkeypatch):
    plugin = builtins.LegacyFactualExplanationPlugin()

    class DummyModel:
        def explain_factual(self, x, **kwargs):  # pragma: no cover - shouldn't run
            raise AssertionError

    with pytest.raises(ConfigurationError):
        plugin.explain(DummyModel(), "x")


def test_legacy_plot_builder_round_trips_context():
    context = _make_plot_context(intent={"type": "anything"})
    builder = builtins.LegacyPlotBuilder()
    assert builder.build(context)["context"] is context


def test_legacy_plot_renderer_produces_empty_result():
    renderer = builtins.LegacyPlotRenderer()
    context = _make_plot_context()
    result = renderer.render({"context": context}, context=context)
    assert result.saved_paths == ()
    assert result.figure is None


def test_plotspec_builder_global_payload(monkeypatch):
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return {"kind": "global"}

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_global_plotspec_dict",
        fake_builder,
    )

    context = _make_plot_context(
        intent={"type": "global", "title": "Global"},
        options={"payload": {"y": [1, 2], "threshold": 0.5, "extra": 1}},
    )
    builder = builtins.PlotSpecDefaultBuilder()
    payload = builder.build(context)
    assert payload == {"kind": "global"}
    assert "threshold" not in captured
    assert "y_test" in captured


def test_plotspec_builder_rejects_non_mapping_payloads():
    builder = builtins.PlotSpecDefaultBuilder()
    with pytest.raises(ConfigurationError):
        builder.build(_make_plot_context(intent={"type": "global"}, options={"payload": 3}))
    with pytest.raises(ConfigurationError):
        builder.build(_make_plot_context(intent={"type": "alternative"}, options={"payload": 3}))
    with pytest.raises(ConfigurationError):
        builder.build(
            _make_plot_context(
                intent={"type": "alternative"},
                options={"payload": {"feature_predict": None, "feature_weights": None}},
            )
        )


def test_plotspec_builder_handles_non_mapping_predict_payload(monkeypatch):
    captured = {}

    def fake_prob(**kwargs):
        captured["prob"] = kwargs
        return {"kind": "prob"}

    def fake_reg(**kwargs):
        captured["reg"] = kwargs
        return {"kind": "reg"}

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_prob,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_reg,
    )

    class Explanation:
        def is_thresholded(self):
            return False

        def get_mode(self):
            return "classification"

    context = _make_plot_context(
        explanation=Explanation(),
        intent={"type": "alternative"},
        options={
            "payload": {
                "feature_predict": [1.0, 2.0],
                "predict": [("predict", 0.5)],
                "features_to_plot": ["0", "bad"],
            }
        },
    )

    builder = builtins.PlotSpecDefaultBuilder()
    builder.build(context)
    assert captured["prob"]["predict"]["low"] == captured["prob"]["predict"]["high"]


def test_plotspec_builder_alternative_regression_threshold(monkeypatch):
    outputs = {}

    def fake_probabilistic(**kwargs):
        outputs.setdefault("prob", []).append(kwargs)
        return {"kind": "prob"}

    def fake_regression(**kwargs):
        outputs.setdefault("reg", []).append(kwargs)
        return {"kind": "reg"}

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_probabilistic,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_regression,
    )

    class Explanation:
        def __init__(self):
            self._threshold = (0.25, 0.5)

        def is_thresholded(self):
            return True

        def get_mode(self):
            return "regression"

        y_minmax = (0.1, 0.9)
        y_threshold = (0.25, 0.5)

    context = _make_plot_context(
        explanation=Explanation(),
        intent={"type": "alternative", "title": "Alt"},
        options={
            "payload": {
                "feature_predict": {"predict": [0.1, 0.2], "low": [0.0, 0.1], "high": [0.3, 0.4]},
                "predict": {"predict": 0.2},
                "features_to_plot": [0, 1],
                "column_names": ("a", "b"),
                "instance": [1.0, 2.0],
            }
        },
    )

    builder = builtins.PlotSpecDefaultBuilder()
    payload = builder.build(context)
    assert payload == {"kind": "prob"}
    kwargs = outputs["prob"][0]
    assert kwargs["neg_label"] == "Outside interval"
    assert kwargs["pos_label"] == "0.25 <= Y < 0.50"
    assert kwargs["xlabel"].startswith("Probability of target")


def test_plotspec_builder_alternative_probability_fallback(monkeypatch):
    outputs = {}

    def fake_probabilistic(**kwargs):
        outputs["call"] = kwargs
        return {"kind": "prob"}

    def fake_regression(**_):  # pragma: no cover - fallback not used in this test
        raise AssertionError("regression path should not be used")

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_probabilistic,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_regression,
    )

    class Explanation:
        def is_thresholded(self):
            return True

        def get_mode(self):
            return "classification"

        y_threshold = 0.7

    context = _make_plot_context(
        explanation=Explanation(),
        intent={"type": "alternative"},
        options={
            "payload": {
                "feature_predict": [0.2, 0.3],
                "predict": {"predict": 0.4},
            }
        },
    )

    builder = builtins.PlotSpecDefaultBuilder()
    payload = builder.build(context)
    assert payload == {"kind": "prob"}
    assert outputs["call"]["neg_label"] is None


def test_plotspec_builder_handles_invalid_sequences(monkeypatch):
    outputs = {}

    def fake_prob(**kwargs):
        outputs["prob"] = kwargs
        return {"kind": "prob"}

    def fake_reg(**kwargs):
        outputs["reg"] = kwargs
        return {"kind": "reg"}

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_prob,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_reg,
    )

    class Explanation:
        def is_thresholded(self):
            return True

        def get_mode(self):
            return "classification"

        y_threshold = object()  # not convertible

    context = _make_plot_context(
        explanation=Explanation(),
        intent={"type": "alternative"},
        options={
            "payload": {
                "feature_predict": {"predict": np.array([1.0]), "low": "bad"},
                "predict": {"predict": "bad", "low": "nan"},
                "features_to_plot": [5],
                "y_minmax": ["bad", 1.0],
            }
        },
    )

    builder = builtins.PlotSpecDefaultBuilder()
    builder.build(context)
    assert outputs["prob"]["features_to_plot"] == [0]


def test_plotspec_builder_regression_without_threshold(monkeypatch):
    outputs = {}

    def fake_prob(**kwargs):
        outputs["prob"] = kwargs
        return {"kind": "prob"}

    def fake_reg(**kwargs):
        outputs["reg"] = kwargs
        return {"kind": "reg"}

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_prob,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_reg,
    )

    class Explanation:
        def is_thresholded(self):
            return False

        def get_mode(self):
            return "regression"

    context = _make_plot_context(
        explanation=Explanation(),
        intent={"type": "alternative"},
        options={
            "payload": {
                "feature_predict": {"predict": [0.1, 0.2]},
                "predict": {"predict": 0.3},
            }
        },
    )

    builder = builtins.PlotSpecDefaultBuilder()
    payload = builder.build(context)
    assert payload == {"kind": "reg"}


def test_plotspec_builder_requires_supported_intent():
    builder = builtins.PlotSpecDefaultBuilder()
    with pytest.raises(ConfigurationError):
        builder.build(_make_plot_context(intent={"type": "unsupported"}))


def test_plotspec_renderer_handles_multiple_outputs(monkeypatch):
    calls = []

    def fake_render(artifact, show, save_path):
        calls.append((artifact, show, save_path))

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        fake_render,
    )

    context = _make_plot_context(show=True, save_ext=(".png", ".svg"), path="/tmp/plot")
    renderer = builtins.PlotSpecDefaultRenderer()
    result = renderer.render({"kind": "prob"}, context=context)
    assert result.saved_paths == ("/tmp/plot.png", "/tmp/plot.svg")
    assert calls[-1][1] is True


def test_plotspec_renderer_without_save(monkeypatch):
    calls = []

    def fake_render(artifact, show, save_path):
        calls.append((artifact, show, save_path))

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        fake_render,
    )

    context = _make_plot_context(show=False, path="base")
    renderer = builtins.PlotSpecDefaultRenderer()
    renderer.render({"kind": "prob"}, context=context)
    assert calls == [({"kind": "prob"}, False, "base")]


def test_register_builtins_uses_registry(monkeypatch):
    recorded = {"interval": [], "explanation": [], "builder": [], "renderer": [], "style": []}

    monkeypatch.setattr(
        builtins, "register_interval_plugin", lambda *a: recorded["interval"].append(a)
    )
    monkeypatch.setattr(
        builtins, "register_explanation_plugin", lambda *a: recorded["explanation"].append(a)
    )
    monkeypatch.setattr(builtins, "register_plot_builder", lambda *a: recorded["builder"].append(a))
    monkeypatch.setattr(
        builtins, "register_plot_renderer", lambda *a: recorded["renderer"].append(a)
    )
    monkeypatch.setattr(
        builtins, "register_plot_style", lambda *a, **k: recorded["style"].append((a, k))
    )

    builtins._register_builtins()  # noqa: SLF001
    assert recorded["interval"]
    assert len(recorded["explanation"]) >= 2
    assert any(item[0] == "core.plot.plot_spec.default" for item in recorded["builder"])


def test_register_builtins_handles_missing_fast_plugins(monkeypatch):
    monkeypatch.setattr(builtins, "register_interval_plugin", lambda *a: None)
    monkeypatch.setattr(builtins, "register_explanation_plugin", lambda *a: None)
    monkeypatch.setattr(builtins, "register_plot_builder", lambda *a: None)
    monkeypatch.setattr(builtins, "register_plot_renderer", lambda *a: None)
    monkeypatch.setattr(builtins, "register_plot_style", lambda *a, **k: None)
    # Ensure import fails even if module existed previously
    sys.modules.pop("external_plugins.fast_explanations", None)
    builtins._register_builtins()  # noqa: SLF001
