"""Unit tests covering CalibratedExplainer runtime helper utilities."""

from __future__ import annotations

import types

import numpy as np
import pytest

from calibrated_explanations.core.prediction import orchestrator as prediction_orchestrator_module
from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
)
from calibrated_explanations.plugins.registry import EXPLANATION_PROTOCOL_VERSION
from calibrated_explanations.plugins.predict_monitor import (
    PredictBridgeMonitor as _PredictBridgeMonitor,
)
from calibrated_explanations.core.exceptions import ConfigurationError


def _stub_explainer(explainer_factory, mode: str = "classification") -> CalibratedExplainer:
    """Construct a fully initialized explainer instance for unit tests."""

    explainer = explainer_factory(mode=mode)
    explainer.bins = None
    explainer._plot_style_override = None
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._interval_plugin_hints = {}
    explainer._interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._telemetry_interval_sources = {"default": None, "fast": None}
    explainer._interval_context_metadata = {"default": {}, "fast": {}}
    explainer._explanation_plugin_overrides = {
        key: None for key in ("factual", "alternative", "fast")
    }
    explainer._pyproject_explanations = {}
    explainer._pyproject_intervals = {}
    explainer._pyproject_plots = {}
    explainer._plot_style_override = None
    explainer._explanation_plugin_fallbacks = {}
    return explainer


def test_coerce_plugin_override_supports_multiple_sources(explainer_factory):
    explainer = _stub_explainer(explainer_factory)
    explainer._plugin_manager = None

    assert explainer._coerce_plugin_override(None) is None
    assert explainer._coerce_plugin_override("tests.override") == "tests.override"

    sentinel = object()

    def factory():
        return sentinel

    assert explainer._coerce_plugin_override(factory) is sentinel

    override = object()
    assert explainer._coerce_plugin_override(override) is override

    def bad_factory():
        raise RuntimeError("boom")

    with pytest.raises(ConfigurationError):
        explainer._coerce_plugin_override(bad_factory)


def test_require_plugin_manager_raises_when_missing(explainer_factory):
    explainer = _stub_explainer(explainer_factory)
    explainer._plugin_manager = None

    with pytest.raises(RuntimeError, match="PluginManager is not initialized"):
        explainer._require_plugin_manager()


def test_build_chains_fall_back_without_plugin_manager(explainer_factory):
    explainer = _stub_explainer(explainer_factory)
    delattr(explainer, "_plugin_manager")

    assert explainer._build_explanation_chain("factual") == ()
    assert explainer._build_interval_chain(fast=False) == ()
    assert explainer._build_plot_style_chain() == ()


def test_predict_bridge_monitor_tracks_usage():
    class DummyBridge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple]] = []

        def predict(self, x, *, mode, task, bins=None):
            self.calls.append(("predict", (mode, task, bins)))
            return {"result": "predict"}

        def predict_interval(self, x, *, task, bins=None):
            self.calls.append(("predict_interval", (task, bins)))
            return ("interval",)

        def predict_proba(self, x, bins=None):
            self.calls.append(("predict_proba", (bins,)))
            return (0.1, 0.9)

    bridge = DummyBridge()
    monitor = _PredictBridgeMonitor(bridge)

    assert monitor.used is False

    assert monitor.predict({}, mode="factual", task="classification") == {"result": "predict"}
    assert monitor.predict_interval({}, task="classification", bins=None) == ("interval",)
    assert monitor.predict_proba({}, bins="sentinel") == (0.1, 0.9)

    assert monitor.calls == ("predict", "predict_interval", "predict_proba")
    assert monitor.used is True
    assert bridge.calls[0][0] == "predict"
    assert bridge.calls[1][0] == "predict_interval"
    assert bridge.calls[2][0] == "predict_proba"


def test_check_explanation_runtime_metadata_reports_errors(explainer_factory):
    explainer = _stub_explainer(explainer_factory, mode="classification")

    assert (
        explainer._check_explanation_runtime_metadata(None, identifier="missing", mode="factual")
        == "missing: plugin metadata unavailable"
    )

    base = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ("classification",),
        "modes": ("factual",),
        "capabilities": (
            "explain",
            "explanation:factual",
            "task:classification",
        ),
    }

    wrong_schema = dict(base, schema_version=-1)
    assert "unsupported" in explainer._check_explanation_runtime_metadata(
        wrong_schema, identifier="id", mode="factual"
    )

    missing_tasks = dict(base, tasks=())
    assert "missing tasks" in explainer._check_explanation_runtime_metadata(
        missing_tasks, identifier="id", mode="factual"
    )

    missing_mode = dict(base, modes=("alternative",))
    assert "does not declare mode" in explainer._check_explanation_runtime_metadata(
        missing_mode, identifier="id", mode="factual"
    )

    missing_caps = dict(base, capabilities=("explain",))
    message = explainer._check_explanation_runtime_metadata(
        missing_caps, identifier="id", mode="factual"
    )
    assert "missing required capabilities" in message

    ok = dict(base)
    assert (
        explainer._check_explanation_runtime_metadata(ok, identifier="id", mode="factual") is None
    )


def test_check_interval_runtime_metadata_validates_requirements(explainer_factory):
    explainer = _stub_explainer(explainer_factory, mode="regression")

    assert (
        explainer._check_interval_runtime_metadata(None, identifier="missing", fast=False)
        == "missing: interval metadata unavailable"
    )

    base = {
        "schema_version": 1,
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
        "fast_compatible": True,
    }

    wrong_schema = dict(base, schema_version=5)
    assert "unsupported interval schema_version" in explainer._check_interval_runtime_metadata(
        wrong_schema, identifier="id", fast=False
    )

    missing_modes = dict(base)
    del missing_modes["modes"]
    assert "missing modes declaration" in explainer._check_interval_runtime_metadata(
        missing_modes, identifier="id", fast=False
    )

    wrong_mode = dict(base, modes=("classification",))
    assert "does not support mode" in explainer._check_interval_runtime_metadata(
        wrong_mode, identifier="id", fast=False
    )

    missing_cap = dict(base, capabilities=("interval:classification",))
    assert "missing capability" in explainer._check_interval_runtime_metadata(
        missing_cap, identifier="id", fast=False
    )

    not_fast = dict(base, fast_compatible=False)
    assert "not marked fast_compatible" in explainer._check_interval_runtime_metadata(
        not_fast, identifier="id", fast=True
    )

    requires_bins = dict(base, requires_bins=True)
    assert "requires bins" in explainer._check_interval_runtime_metadata(
        requires_bins, identifier="id", fast=False
    )

    explainer.bins = ("bin",)
    assert explainer._check_interval_runtime_metadata(base, identifier="id", fast=True) is None


def test_fast_interval_initializer_delegates_to_registry(monkeypatch, explainer_factory):
    explainer = explainer_factory(mode="regression")
    sentinel = object()

    class StubIntervalRegistry:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def initialize_for_fast_explainer(self) -> None:
            self.calls.append(sentinel)

    registry = StubIntervalRegistry()

    class StubPredictionOrchestrator:
        def __init__(self, interval_registry: StubIntervalRegistry) -> None:
            self._interval_registry = interval_registry
            self._prediction_orchestrator = self

    stub_manager = StubPredictionOrchestrator(registry)
    monkeypatch.setattr(explainer, "_require_plugin_manager", lambda: stub_manager)
    explainer._CalibratedExplainer__initialize_interval_learner_for_fast_explainer()

    assert registry.calls == [sentinel]


def test_ensure_interval_runtime_state_populates_defaults(explainer_factory):
    explainer = explainer_factory()
    explainer._interval_plugin_hints = {}
    explainer._interval_plugin_fallbacks = {}
    explainer._interval_plugin_identifiers = {}
    explainer._telemetry_interval_sources = {}
    explainer._interval_preferred_identifier = {}
    explainer._interval_context_metadata = {}
    explainer._ensure_interval_runtime_state()

    assert explainer._interval_plugin_hints == {}
    assert explainer._interval_plugin_fallbacks == {}
    assert explainer._interval_plugin_identifiers == {"default": None, "fast": None}
    assert explainer._telemetry_interval_sources == {"default": None, "fast": None}
    assert explainer._interval_preferred_identifier == {"default": None, "fast": None}
    assert explainer._interval_context_metadata == {"default": {}, "fast": {}}


def test_verbose_repr_includes_metadata(explainer_factory):
    explainer = explainer_factory()
    explainer.verbose = True
    explainer._feature_names = ["f1", "f2"]  # noqa: SLF001 - testing repr content
    explainer.categorical_features = [0]
    explainer.categorical_labels = {0: {0: "no", 1: "yes"}}
    explainer.class_labels = {0: "neg", 1: "pos"}

    result = repr(explainer)

    assert "CalibratedExplainer" in result
    assert "feature_names=['f1', 'f2']" in result
    assert "categorical_features=[0]" in result
    assert "categorical_labels={0: {0: 'no', 1: 'yes'}}" in result
    assert "class_labels={0: 'neg', 1: 'pos'}" in result


def test_instantiate_plugin_handles_multiple_paths(monkeypatch, explainer_factory):
    explainer = _stub_explainer(explainer_factory)

    class CallableWithMeta:
        plugin_meta = {}

        def __call__(self):  # pragma: no cover - guard against accidental call
            raise AssertionError("should not be invoked")

    callable_with_meta = CallableWithMeta()
    assert explainer._instantiate_plugin(callable_with_meta) is callable_with_meta

    class SimplePlugin:
        def __init__(self) -> None:
            self.token = object()

    prototype = SimplePlugin()
    clone = explainer._instantiate_plugin(prototype)
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
    assert explainer._instantiate_plugin(broken) is sentinel

    # Test fallback when deepcopy itself fails
    def raising_deepcopy(value):
        raise RuntimeError("fail")

    monkeypatch.setattr(explain_orch.copy, "deepcopy", raising_deepcopy)
    assert explainer._instantiate_plugin(broken) is broken


def test_resolve_interval_plugin_handles_denied_and_success(monkeypatch, explainer_factory):
    explainer = _stub_explainer(explainer_factory, mode="regression")
    explainer._interval_plugin_fallbacks = {"default": ("denied.plugin", "ok.plugin"), "fast": ()}
    explainer._instantiate_plugin = lambda plugin: plugin
    explainer._check_interval_runtime_metadata = lambda metadata, **_: None

    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        prediction_orchestrator_module,
        "is_identifier_denied",
        lambda identifier: identifier == "denied.plugin",
    )

    descriptor = types.SimpleNamespace(
        metadata={
            "schema_version": 1,
            "modes": ("regression",),
            "capabilities": ("interval:regression",),
            "fast_compatible": True,
        },
        trusted=True,
        plugin=types.SimpleNamespace(plugin_meta={"name": "ok.plugin"}),
    )

    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_descriptor",
        lambda identifier: descriptor if identifier == "ok.plugin" else None,
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None
    )
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    plugin, identifier = explainer._resolve_interval_plugin(fast=False)

    assert identifier == "ok.plugin"
    assert plugin is descriptor.plugin


def test_resolve_interval_plugin_denied_override_raises(monkeypatch, explainer_factory):
    explainer = _stub_explainer(explainer_factory, mode="regression")
    explainer._interval_plugin_override = "denied.plugin"
    explainer._interval_plugin_fallbacks = {"default": ("denied.plugin",), "fast": ()}

    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        prediction_orchestrator_module,
        "is_identifier_denied",
        lambda identifier: identifier == "denied.plugin",
    )

    with pytest.raises(ConfigurationError) as excinfo:
        explainer._resolve_interval_plugin(fast=False)

    assert "denied via CE_DENY_PLUGIN" in str(excinfo.value)


def test_build_interval_context_enriches_metadata(explainer_factory):
    explainer = _stub_explainer(explainer_factory, mode="regression")
    explainer.x_cal = np.asarray([[1.0, 2.0]])
    explainer.y_cal = np.asarray([1.5])
    explainer._X_cal = explainer.x_cal
    explainer.bins = np.asarray([0])
    explainer.difficulty_estimator = "difficulty"
    explainer._interval_context_metadata = {"default": {"preexisting": {"value": 1}}, "fast": {}}
    explainer._CalibratedExplainer__noise_type = "gaussian"
    explainer.categorical_features = [1]
    explainer.learner = object()

    context = explainer._build_interval_context(fast=False, metadata={"extra": 2})

    assert context.learner is explainer.learner
    assert context.calibration_splits[0] == (explainer.x_cal, explainer.y_cal)
    assert context.bins["calibration"].shape == (1,)
    assert context.difficulty["estimator"] == "difficulty"
    assert context.fast_flags == {"fast": False}
    assert context.metadata["preexisting"] == {"value": 1}
    assert context.metadata["extra"] == 2
    assert context.metadata["task"] == "regression"
    assert context.metadata["mode"] == "regression"
    assert context.metadata["categorical_features"] == (1,)
    assert context.metadata["num_features"] == 2
    assert context.metadata["noise_config"]["noise_type"] == "gaussian"


def test_get_calibration_summaries_caches_results(explainer_factory):
    explainer = _stub_explainer(explainer_factory)
    explainer.x_cal = np.asarray([[0, "a"], [1, "b"], [0, "a"]], dtype=object)
    explainer._X_cal = explainer.x_cal
    explainer.categorical_features = [1]
    explainer._categorical_value_counts_cache = None
    explainer._numeric_sorted_cache = None
    explainer._calibration_summary_shape = None

    counts, numeric = explainer._get_calibration_summaries()

    assert counts[1]["a"] == 2
    assert counts[1]["b"] == 1
    assert np.array_equal(numeric[0], np.array([0, 0, 1]))

    counts_cached, numeric_cached = explainer._get_calibration_summaries()

    assert counts_cached is counts
    assert numeric_cached is numeric


class _RaisingInterval:
    def predict_uncertainty(self, *_, **__):
        raise RuntimeError("boom")

    def predict_probability(self, *_, **__):
        raise RuntimeError("boom")


def test_predict_impl_returns_degraded_arrays_when_suppressed(explainer_factory):
    explainer = _stub_explainer(explainer_factory, mode="regression")
    explainer._CalibratedExplainer__initialized = True
    explainer._CalibratedExplainer__fast = False
    explainer.interval_learner = _RaisingInterval()
    explainer.suppress_crepes_errors = True
    explainer._X_cal = np.zeros((1, 1))

    x = np.ones((2, 1))

    predict, low, high, classes = explainer._predict_impl(x)

    assert np.all(predict == 0)
    assert np.all(low == 0)
    assert np.all(high == 0)
    assert classes is None

    threshold = [0.5, 0.5]
    predict_t, low_t, high_t, classes_t = explainer._predict_impl(x, threshold=threshold)

    assert np.all(predict_t == 0)
    assert np.all(low_t == 0)
    assert np.all(high_t == 0)
    assert classes_t is None


def test_infer_explanation_mode_detects_entropy_discretizer(explainer_factory):
    """_infer_explanation_mode should detect alternative mode when EntropyDiscretizer is set."""
    from calibrated_explanations.utils.discretizers import EntropyDiscretizer

    explainer = _stub_explainer(explainer_factory)
    x_cal = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_cal = np.asarray([0, 1, 0])
    explainer.discretizer = EntropyDiscretizer(
        x_cal,
        categorical_features={},
        feature_names=["f0", "f1"],
        labels=y_cal,
        random_state=0,
    )

    assert explainer._infer_explanation_mode() == "alternative"


def test_infer_explanation_mode_detects_regressor_discretizer(explainer_factory):
    """_infer_explanation_mode should detect alternative mode when RegressorDiscretizer is set."""
    from calibrated_explanations.utils.discretizers import RegressorDiscretizer

    explainer = _stub_explainer(explainer_factory, mode="regression")
    x_cal = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_cal = np.asarray([1.0, 2.0, 3.0])
    explainer.discretizer = RegressorDiscretizer(
        x_cal,
        categorical_features={},
        feature_names=["f0", "f1"],
        labels=y_cal,
        random_state=0,
    )

    assert explainer._infer_explanation_mode() == "alternative"


def test_infer_explanation_mode_defaults_to_factual(explainer_factory):
    """_infer_explanation_mode should return 'factual' for None or other discretizers."""
    explainer = _stub_explainer(explainer_factory)
    assert explainer._infer_explanation_mode() == "factual"


def test_preprocessor_metadata_round_trip(explainer_factory):
    """preprocessor_metadata property should preserve and return metadata dict."""
    explainer = _stub_explainer(explainer_factory)

    metadata = {"scaling": "StandardScaler", "features": 10}
    explainer.set_preprocessor_metadata(metadata)

    assert explainer.preprocessor_metadata == metadata
    assert explainer.preprocessor_metadata is not metadata  # Should be a copy


def test_preprocessor_metadata_none_handling(explainer_factory):
    """set_preprocessor_metadata(None) should clear metadata."""
    explainer = _stub_explainer(explainer_factory)
    explainer.set_preprocessor_metadata({"key": "value"})
    explainer.set_preprocessor_metadata(None)

    assert explainer.preprocessor_metadata is None


def test_coerce_plugin_override_factory_error_handling(explainer_factory):
    """_coerce_plugin_override should convert factory exceptions to ConfigurationError."""
    explainer = _stub_explainer(explainer_factory)
    explainer._plugin_manager = None

    def failing_factory():
        raise ValueError("Factory failed")

    from calibrated_explanations.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="Callable explanation plugin override raised"):
        explainer._coerce_plugin_override(failing_factory)


def test_prediction_orchestrator_raises_when_missing_attribute(explainer_factory):
    """_prediction_orchestrator should raise when manager lacks the attribute."""
    explainer = _stub_explainer(explainer_factory)

    class BrokenManager:
        pass

    explainer._plugin_manager = BrokenManager()

    with pytest.raises(
        AttributeError, match="PluginManager has no '_prediction_orchestrator'"
    ):
        _ = explainer._prediction_orchestrator


def test_explanation_orchestrator_raises_when_missing_attribute(explainer_factory):
    """_explanation_orchestrator should raise when manager lacks the attribute."""
    explainer = _stub_explainer(explainer_factory)

    class BrokenManager:
        pass

    explainer._plugin_manager = BrokenManager()

    with pytest.raises(
        AttributeError, match="PluginManager has no '_explanation_orchestrator'"
    ):
        _ = explainer._explanation_orchestrator


def test_reject_orchestrator_raises_when_missing_attribute(explainer_factory):
    """_reject_orchestrator should raise when manager lacks the attribute."""
    explainer = _stub_explainer(explainer_factory)

    class BrokenManager:
        pass

    explainer._plugin_manager = BrokenManager()

    with pytest.raises(AttributeError, match="PluginManager has no '_reject_orchestrator'"):
        _ = explainer._reject_orchestrator


def test_explanation_plugin_instances_empty_without_manager(explainer_factory):
    """_explanation_plugin_instances should return empty dict without manager."""
    explainer = _stub_explainer(explainer_factory)
    delattr(explainer, "_plugin_manager")

    assert explainer._explanation_plugin_instances == {}


def test_explanation_plugin_identifiers_empty_without_manager(explainer_factory):
    """_explanation_plugin_identifiers should return empty dict without manager."""
    explainer = _stub_explainer(explainer_factory)
    delattr(explainer, "_plugin_manager")

    assert explainer._explanation_plugin_identifiers == {}


def test_interval_plugin_hints_property_operations(explainer_factory):
    """_interval_plugin_hints property should get/set via manager."""
    explainer = _stub_explainer(explainer_factory)

    # Create a mock manager
    class MockManager:
        def __init__(self):
            self._interval_plugin_hints = {"default": ("hint1", "hint2")}

    explainer._plugin_manager = MockManager()

    assert explainer._interval_plugin_hints == {"default": ("hint1", "hint2")}

    explainer._interval_plugin_hints = {"fast": ("fast_hint",)}
    assert explainer._plugin_manager._interval_plugin_hints == {"fast": ("fast_hint",)}


def test_explanation_plugin_fallbacks_property_operations(explainer_factory):
    """_explanation_plugin_fallbacks property should get/set via manager."""
    explainer = _stub_explainer(explainer_factory)

    class MockManager:
        def __init__(self):
            self._explanation_plugin_fallbacks = {"factual": ("a", "b")}

    explainer._plugin_manager = MockManager()

    assert explainer._explanation_plugin_fallbacks == {"factual": ("a", "b")}

    explainer._explanation_plugin_fallbacks = {"alternative": ("c",)}
    assert explainer._plugin_manager._explanation_plugin_fallbacks == {"alternative": ("c",)}


def test_plot_plugin_fallbacks_property_operations(explainer_factory):
    """_plot_plugin_fallbacks property should get/set via manager."""
    explainer = _stub_explainer(explainer_factory)

    class MockManager:
        def __init__(self):
            self._plot_plugin_fallbacks = {"default": ("plot_a",)}

    explainer._plugin_manager = MockManager()

    assert explainer._plot_plugin_fallbacks == {"default": ("plot_a",)}

    explainer._plot_plugin_fallbacks = {"alt": ("plot_b",)}
    assert explainer._plugin_manager._plot_plugin_fallbacks == {"alt": ("plot_b",)}
