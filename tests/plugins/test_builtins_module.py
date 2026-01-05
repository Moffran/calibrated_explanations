"""Additional coverage for built-in plugin adapters."""

from __future__ import annotations

import importlib
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable

import numpy as np
import pytest

from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    NotFittedError,
)
from calibrated_explanations.plugins import builtins as builtins_mod
from calibrated_explanations.plugins.builtins import (
    LegacyAlternativeExplanationPlugin,
    LegacyFactualExplanationPlugin,
    LegacyIntervalCalibratorPlugin,
    LegacyPlotBuilder,
    LegacyPlotRenderer,
    LegacyPredictBridge,
    PlotSpecDefaultBuilder,
    PlotSpecDefaultRenderer,
    collection_to_batch,
    _register_builtins,
)
from calibrated_explanations.plugins.explanations import (
    ExplanationBatch,
    ExplanationContext,
    ExplanationRequest,
)
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext
from calibrated_explanations.plugins.plots import PlotRenderContext


def make_interval_context(
    task: str, metadata: Dict[str, Any] | None = None
) -> IntervalCalibratorContext:
    """Create a minimal interval context for exercising the plugin."""

    metadata_map: Dict[str, Any] = {"task": task}
    if metadata:
        metadata_map.update(metadata)
    learner = SimpleNamespace()
    calibration_splits = ((np.asarray([[1.0]]), np.asarray([1.0])),)
    return IntervalCalibratorContext(
        learner=learner,
        calibration_splits=calibration_splits,
        bins={"calibration": "cal_bins"},
        residuals={},
        difficulty={"estimator": "difficulty"},
        metadata=metadata_map,
        fast_flags={},
    )


class DummyPredictBridge(LegacyPredictBridge):
    """Bridge with predictable output for testing explanation batches."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Dict[str, Any]]] = []

        class Explainer:
            def predict(self, *args: Any, **kwargs: Any) -> Any:
                return np.asarray([0.5])

        super().__init__(Explainer())

    def predict(self, x: Any, *, mode: str, task: str, bins: Any | None = None) -> Dict[str, Any]:
        payload = {"x": x, "mode": mode, "task": task, "bins": bins}
        self.calls.append((x, payload))
        return payload


class DummyExplanation(SimpleNamespace):
    """Small explanation object used to populate collections."""


@pytest.fixture
def explanation_context(monkeypatch: pytest.MonkeyPatch) -> ExplanationContext:
    class Explainer:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        def explain_factual(self, data: Any, **kwargs: Any):
            self.calls.append({"data": data, "kwargs": kwargs})
            collection = make_collection(with_instances=True)
            return collection

    bridge = DummyPredictBridge()
    context = ExplanationContext(
        task="classification",
        mode="factual",
        feature_names=("f0", "f1"),
        categorical_features=(),
        categorical_labels={},
        discretizer=None,
        helper_handles={"explainer": Explainer()},
        predict_bridge=bridge,
        interval_settings={},
        plot_settings={},
    )
    return context


def make_collection(*, with_instances: bool) -> builtins_mod.CalibratedExplanations:
    explainer = SimpleNamespace(
        x_cal=np.asarray([[0.0]]),
        y_cal=np.asarray([0.0]),
        num_features=1,
        categorical_features=(),
        continuous_features=(),
        ordinal_features=(),
        cat_feat_labels={},
        feature_names=("f0",),
    )
    collection = builtins_mod.CalibratedExplanations(
        explainer,
        np.asarray([[1.0]]),
        y_threshold=None,
        bins=("b0",),
        features_to_ignore=None,
    )
    if with_instances:
        collection.explanations.append(DummyExplanation(label="dummy"))
    return collection


def make_local_plot_context(**kwargs: Any) -> PlotRenderContext:
    base_kwargs = {
        "explanation": None,
        "instance_metadata": MappingProxyType({"type": "alternative"}),
        "style": "plot_spec.default",
        "intent": MappingProxyType({"type": "global", "title": "demo"}),
        "show": False,
        "path": None,
        "save_ext": None,
        "options": MappingProxyType({"payload": {"y": np.asarray([1, 2, 3])}}),
    }
    base_kwargs.update(kwargs)
    return PlotRenderContext(**base_kwargs)


def test_legacy_predict_bridge_includes_intervals_and_classes():
    class Explainer:
        def __init__(self) -> None:
            self.predict_calls: list[Dict[str, Any]] = []

        def predict(self, *args: Any, **kwargs: Any) -> Any:
            self.predict_calls.append({"args": args, "kwargs": kwargs})
            if kwargs.get("calibrated"):
                return ["a", "b"]
            return (
                np.asarray([0.1, 0.9]),
                (np.asarray([0.0, 0.5]), np.asarray([0.2, 1.0])),
            )

    bridge = LegacyPredictBridge(Explainer())
    payload = bridge.predict("item", mode="alt", task="classification", bins="b")

    assert payload["mode"] == "alt"
    np.testing.assert_allclose(payload["predict"], [0.1, 0.9])
    np.testing.assert_allclose(payload["low"], [0.0, 0.5])
    np.testing.assert_allclose(payload["high"], [0.2, 1.0])
    np.testing.assert_array_equal(payload["classes"], ["a", "b"])


def test_legacy_predict_bridge_handles_scalar_predictions():
    class Explainer:
        def predict(self, *args: Any, **kwargs: Any) -> Any:
            return np.asarray([0.42])

    bridge = LegacyPredictBridge(Explainer())
    payload = bridge.predict(123, mode="factual", task="regression", bins=None)

    np.testing.assert_allclose(payload["predict"], [0.42])
    assert "low" not in payload and "classes" not in payload


def test_legacy_predict_bridge_passes_through_expected_flags():
    class Explainer:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        def predict(self, *args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if kwargs.get("calibrated"):
                return np.asarray([1])
            return (np.asarray([0.2]), (np.asarray([0.1]), np.asarray([0.3])))

    explainer = Explainer()
    bridge = LegacyPredictBridge(explainer)
    bridge.predict("x", mode="factual", task="classification", bins="bucket")

    assert explainer.calls[0] == {"uq_interval": True, "bins": "bucket"}
    assert explainer.calls[1] == {"calibrated": True, "bins": "bucket"}


def test_predict_bridge_interval_and_proba():
    class Explainer:
        def __init__(self) -> None:
            self.interval_called = False

        def predict(self, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("calibrated"):
                return np.asarray([0.9])
            return (
                np.asarray([0.5]),
                (np.asarray([0.25]), np.asarray([0.75])),
            )

        def predict_proba(self, *args: Any, **kwargs: Any) -> Any:
            return np.asarray([0.1, 0.9])

    bridge = LegacyPredictBridge(Explainer())
    interval = bridge.predict_interval([1], task="classification")
    proba = bridge.predict_proba([1])
    assert np.allclose(interval, [0.9])
    assert np.allclose(proba, [0.1, 0.9])


def test_supports_and_explain_methods(monkeypatch: pytest.MonkeyPatch):
    module_name = "calibrated_explanations.core.calibrated_explainer"
    dummy_module = ModuleType(module_name)
    dummy_explainer_cls = type(
        "DummyExplainer", (), {"explain_factual": lambda self, *a, **k: "ok"}
    )
    dummy_module.CalibratedExplainer = dummy_explainer_cls
    original = sys.modules.get(module_name)
    sys.modules[module_name] = dummy_module
    try:
        instance = dummy_explainer_cls()
        plugin = LegacyFactualExplanationPlugin()
        assert plugin.supports(instance)
        assert plugin.supports_mode("factual", task="classification")
        with pytest.raises(ConfigurationError):
            plugin.explain(object(), "x")
        assert plugin.explain(instance, "x") == "ok"
    finally:
        if original is not None:
            sys.modules[module_name] = original
        else:
            del sys.modules[module_name]


def test_collection_to_batch_handles_empty_and_populated_collections():
    empty_batch = collection_to_batch(make_collection(with_instances=False))
    assert empty_batch.instances == ()
    assert empty_batch.explanation_cls is builtins_mod.FactualExplanation

    populated = make_collection(with_instances=True)
    batch = collection_to_batch(populated)
    assert isinstance(batch, ExplanationBatch)
    assert batch.instances[0]["explanation"].label == "dummy"
    assert batch.collection_metadata["container"] is populated


def test_legacy_plot_renderer_creates_result():
    renderer = LegacyPlotRenderer()
    ctx = make_local_plot_context()
    result = renderer.render({"artifact": 1}, context=ctx)
    assert getattr(result, "artifact", None) == {"artifact": 1}
    assert result.saved_paths == ()


def test_interval_plugin_requires_handles_and_returns_calibrator(monkeypatch: pytest.MonkeyPatch):
    created = {}

    class DummyCalibrator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            created["args"] = args
            created["kwargs"] = kwargs

    monkeypatch.setattr(
        "calibrated_explanations.calibration.interval_regressor.IntervalRegressor",
        DummyCalibrator,
        raising=False,
    )
    context = make_interval_context("regression")

    with pytest.raises(NotFittedError):
        LegacyIntervalCalibratorPlugin().create(context)

    context.metadata["explainer"] = SimpleNamespace()
    calibrator = LegacyIntervalCalibratorPlugin().create(context)
    assert created["args"] == (context.metadata["explainer"],)
    assert calibrator is not None


def test_interval_plugin_uses_predict_function_and_sets_metadata(monkeypatch: pytest.MonkeyPatch):
    created = {}

    class DummyCalibrator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            created["args"] = args
            created["kwargs"] = kwargs

    monkeypatch.setattr(
        "calibrated_explanations.calibration.venn_abers.VennAbers",
        DummyCalibrator,
        raising=False,
    )
    context = make_interval_context("classification")
    context.metadata["explainer"] = SimpleNamespace(predict_function=lambda x: x)
    calibrator = LegacyIntervalCalibratorPlugin().create(context)

    assert created["args"][0] is context.calibration_splits[0][0]
    # Plugin does not cache the calibrator; orchestrator does that via capture_interval_calibrators


def test_interval_plugin_requires_predict_callable(monkeypatch: pytest.MonkeyPatch):
    class DummyCalibrator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "calibrated_explanations.calibration.venn_abers.VennAbers",
        DummyCalibrator,
        raising=False,
    )
    context = make_interval_context("classification")

    with pytest.raises(NotFittedError):
        LegacyIntervalCalibratorPlugin().create(context)


def test_explanation_plugin_initialization_and_batch(explanation_context: ExplanationContext):
    plugin = LegacyFactualExplanationPlugin()
    plugin.initialize(explanation_context)
    request = ExplanationRequest(
        threshold=None,
        low_high_percentiles=(0.1, 0.9),
        bins="binning",
        features_to_ignore=(1,),
        extras={},
    )
    batch = plugin.explain_batch("data", request)
    assert isinstance(batch, ExplanationBatch)
    assert batch.collection_metadata["container"].explanations


def test_explanation_plugin_requires_initialization():
    plugin = LegacyAlternativeExplanationPlugin()
    request = ExplanationRequest(
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=(),
        extras={},
    )
    with pytest.raises(NotFittedError):
        plugin.explain_batch("x", request)


def test_explanation_plugin_missing_explainer_raises(explanation_context: ExplanationContext):
    plugin = LegacyAlternativeExplanationPlugin()
    context = ExplanationContext(
        task="classification",
        mode="alternative",
        feature_names=("a",),
        categorical_features=(),
        categorical_labels={},
        discretizer=None,
        helper_handles={},
        predict_bridge=DummyPredictBridge(),
        interval_settings={},
        plot_settings={},
    )
    with pytest.raises(NotFittedError):
        plugin.initialize(context)


def test_plot_builder_and_renderer_behaviour(monkeypatch: pytest.MonkeyPatch):
    builder = LegacyPlotBuilder()
    # Use a non-global intent to exercise the individual-plot build path
    from types import MappingProxyType

    context = make_local_plot_context(intent=MappingProxyType({"type": "individual"}))
    assert builder.build(context)["context"] is context

    renderer_calls: list[Dict[str, Any]] = []

    def fake_render(artifact: Dict[str, Any], *, show: bool, save_path: str | None) -> None:
        renderer_calls.append({"artifact": artifact, "show": show, "save_path": save_path})

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        fake_render,
        raising=False,
    )
    renderer = PlotSpecDefaultRenderer()
    ctx = make_local_plot_context(save_ext=[".png", ".svg"], path="/tmp/output", show=True)
    result = renderer.render({"spec": 1}, context=ctx)

    assert len(renderer_calls) == 3
    assert result.saved_paths == ("/tmp/output.png", "/tmp/output.svg")


def test_plot_spec_builder_global_and_error_paths(monkeypatch: pytest.MonkeyPatch):
    builder = PlotSpecDefaultBuilder()
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_global_plotspec_dict",
        lambda **kwargs: kwargs,
        raising=False,
    )
    ctx = make_local_plot_context()
    payload = builder.build(ctx)
    assert isinstance(payload, dict)
    assert "y_test" in payload
    assert payload.get("title") == "demo"

    bad_ctx = make_local_plot_context(
        options=MappingProxyType({"payload": 1}), intent=MappingProxyType({"type": "global"})
    )
    with pytest.raises(ConfigurationError):
        builder.build(bad_ctx)

    alt_ctx = make_local_plot_context(
        intent=MappingProxyType({"type": "alternative", "title": "alt"}),
        options=MappingProxyType({"payload": {"predict": {}}}),
    )
    with pytest.raises(ConfigurationError):
        builder.build(alt_ctx)

    alt_bad_payload = make_local_plot_context(
        intent=MappingProxyType({"type": "alternative", "title": "alt"}),
        options=MappingProxyType({"payload": 1}),
    )
    with pytest.raises(ConfigurationError):
        builder.build(alt_bad_payload)


def test_plot_spec_builder_thresholded_classification(monkeypatch: pytest.MonkeyPatch):
    builder = PlotSpecDefaultBuilder()

    def fake_prob_spec(**kwargs: Any) -> Dict[str, Any]:
        return kwargs

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_prob_spec,
        raising=False,
    )
    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_prob_spec,
        raising=False,
    )

    explanation = SimpleNamespace(
        is_thresholded=lambda: True,
        y_threshold=(0.1, 0.9),
        get_mode=lambda: "regression",
    )
    ctx = make_local_plot_context(
        intent=MappingProxyType({"type": "alternative", "mode": "regression", "title": "alt"}),
        options=MappingProxyType(
            {
                "payload": {
                    "predict": {"predict": 0.4},
                    "feature_predict": {"predict": [0.2, 0.1]},
                    "features_to_plot": [0, "invalid", 1],
                    "neg_label": "no",
                    "pos_label": "yes",
                }
            }
        ),
        explanation=explanation,
    )

    spec = builder.build(ctx)
    assert spec["neg_label"] == "Outside interval"
    assert spec["pos_label"] == "0.10 <= Y < 0.90"


def test_plot_spec_builder_handles_full_feature_payload(monkeypatch: pytest.MonkeyPatch):
    builder = PlotSpecDefaultBuilder()

    def fake_prob_spec(**kwargs: Any) -> Dict[str, Any]:
        return kwargs

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_prob_spec,
        raising=False,
    )

    payload = {
        "predict": [("predict", "bad"), ("low", "not"), ("high", np.nan)],
        "feature_predict": {
            "predict": {"0": [0.1, "err", 0.2]},
            "low": np.array([0.0, np.nan, 0.1]),
            "high": (0.2, 0.3, 0.4),
        },
        "features_to_plot": ["x", 1, 2],
        "feature_names": ("f0", "f1", "f2"),
        "rule_labels": ("r0", "r1", "r2"),
        "instance_values": [1, 2, 3],
        "y_minmax": ("a", 5),
        "mode": "classification",
        "neg_label": "neg",
        "pos_label": "pos",
        "legacy_solid_behavior": False,
        "uncertainty_color": "red",
        "uncertainty_alpha": 0.4,
    }
    explanation = SimpleNamespace(
        is_thresholded=lambda: False,
        get_mode=lambda: "classification",
    )
    ctx = make_local_plot_context(
        intent=MappingProxyType({"type": "alternative", "title": "full"}),
        options=MappingProxyType({"payload": payload}),
        explanation=explanation,
    )

    spec = builder.build(ctx)
    assert spec["feature_weights"]
    assert spec["features_to_plot"] == [1, 2]
    assert spec["column_names"] == ["f0", "f1", "f2"]
    assert spec["neg_label"] == "neg"


def test_plot_spec_builder_uses_context_minmax(monkeypatch: pytest.MonkeyPatch):
    builder = PlotSpecDefaultBuilder()

    def fake_prob_spec(**kwargs: Any) -> Dict[str, Any]:
        return kwargs

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_probabilistic_spec",
        fake_prob_spec,
        raising=False,
    )

    explanation = SimpleNamespace(
        y_minmax=(0.2, 0.8),
        is_thresholded=lambda: True,
        y_threshold="bad",
        get_mode=lambda: "classification",
    )
    payload = {
        "predict": {"predict": 0.5},
        "feature_predict": [0.1, "oops", 0.3],
        "features_to_plot": ["no", "bad"],
        "instance": [1, 2, 3],
    }
    ctx = make_local_plot_context(
        intent=MappingProxyType({"type": "alternative", "title": "ctx"}),
        options=MappingProxyType({"payload": payload}),
        explanation=explanation,
    )

    spec = builder.build(ctx)
    assert spec["y_minmax"] == (0.2, 0.8)
    assert spec["features_to_plot"] == [0, 1, 2]
    assert "threshold_label" not in spec


def test_plot_spec_builder_regression_without_threshold(monkeypatch: pytest.MonkeyPatch):
    builder = PlotSpecDefaultBuilder()

    def fake_regression(**kwargs: Any) -> Dict[str, Any]:
        return kwargs

    monkeypatch.setattr(
        "calibrated_explanations.viz.builders.build_alternative_regression_spec",
        fake_regression,
        raising=False,
    )

    explanation = SimpleNamespace(
        is_thresholded=lambda: False,
        y_threshold=None,
        get_mode=lambda: "regression",
    )
    payload = {
        "predict": {"predict": 0.1},
        "feature_predict": {"predict": [0.1]},
    }
    ctx = make_local_plot_context(
        intent=MappingProxyType({"type": "alternative", "mode": "regression"}),
        options=MappingProxyType({"payload": payload}),
        explanation=explanation,
    )

    spec = builder.build(ctx)
    assert spec["predict"]["predict"] == 0.1


def test_plot_spec_builder_unsupported_intent():
    builder = PlotSpecDefaultBuilder()
    ctx = make_local_plot_context(intent=MappingProxyType({"type": "other"}))
    with pytest.raises(ConfigurationError):
        builder.build(ctx)


def test_register_builtins_invokes_registry(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, Iterable[str]]] = []

    monkeypatch.setattr(
        builtins_mod,
        "register_interval_plugin",
        lambda *args, **kwargs: calls.append(("interval", args)),
    )
    monkeypatch.setattr(
        builtins_mod,
        "register_explanation_plugin",
        lambda *args, **kwargs: calls.append(("explanation", args)),
    )
    monkeypatch.setattr(
        builtins_mod,
        "register_plot_builder",
        lambda *args, **kwargs: calls.append(("builder", args)),
    )
    monkeypatch.setattr(
        builtins_mod,
        "register_plot_renderer",
        lambda *args, **kwargs: calls.append(("renderer", args)),
    )
    monkeypatch.setattr(
        builtins_mod, "register_plot_style", lambda *args, **kwargs: calls.append(("style", args))
    )

    _register_builtins()

    kinds = {kind for kind, _ in calls}
    assert {"interval", "explanation", "builder", "renderer", "style"}.issubset(kinds)

    importlib.reload(builtins_mod)


def test_plot_spec_renderer_without_save(monkeypatch: pytest.MonkeyPatch):
    renderer = PlotSpecDefaultRenderer()

    def fake_render(artifact: Dict[str, Any], *, show: bool, save_path: str | None) -> None:
        if save_path == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        fake_render,
        raising=False,
    )
    ctx = make_local_plot_context(save_ext=None, path=None, show=False)
    result = renderer.render({"spec": 1}, context=ctx)
    assert result.saved_paths == ()


def test_plot_spec_renderer_raises_runtime_error(monkeypatch: pytest.MonkeyPatch):
    renderer = PlotSpecDefaultRenderer()

    def blow_up(*args: Any, **kwargs: Any) -> None:
        raise ValueError("fail")

    monkeypatch.setattr(
        "calibrated_explanations.viz.matplotlib_adapter.render",
        blow_up,
        raising=False,
    )
    ctx = make_local_plot_context(save_ext=[".png"], path="bad", show=False)
    with pytest.raises(ConfigurationError):
        renderer.render({"spec": 1}, context=ctx)


def test_register_builtins_imports_fast_plugins(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    fake_module = ModuleType("external_plugins.fast_explanations")

    def fake_register():
        calls.append("register")

    fake_module.register = fake_register
    monkeypatch.setitem(sys.modules, "external_plugins.fast_explanations", fake_module)
    monkeypatch.setattr(builtins_mod, "register_interval_plugin", lambda *args, **kwargs: None)
    monkeypatch.setattr(builtins_mod, "register_explanation_plugin", lambda *args, **kwargs: None)
    monkeypatch.setattr(builtins_mod, "register_plot_builder", lambda *args, **kwargs: None)
    monkeypatch.setattr(builtins_mod, "register_plot_renderer", lambda *args, **kwargs: None)
    monkeypatch.setattr(builtins_mod, "register_plot_style", lambda *args, **kwargs: None)

    _register_builtins()
    assert calls == ["register"]
