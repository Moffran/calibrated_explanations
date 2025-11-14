"""Unit tests covering CalibratedExplainer runtime helper utilities."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.prediction import orchestrator as prediction_orchestrator_module
from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    EXPLANATION_PROTOCOL_VERSION,
)
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.plugins.predict_monitor import (
    PredictBridgeMonitor as _PredictBridgeMonitor,
)
from calibrated_explanations.core.exceptions import ConfigurationError


def _stub_explainer(mode: str = "classification") -> CalibratedExplainer:
    """Construct a lightweight explainer instance for unit tests."""

    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer.mode = mode
    explainer.bins = None
    explainer._plot_style_override = None
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._interval_plugin_hints = {}
    explainer._interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._telemetry_interval_sources = {"default": None, "fast": None}
    explainer._interval_context_metadata = {"default": {}, "fast": {}}
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._explanation_plugin_overrides = {
        mode: None for mode in ("factual", "alternative", "fast")
    }
    explainer._pyproject_explanations = {}
    explainer._pyproject_intervals = {}
    explainer._pyproject_plots = {}
    explainer._plot_style_override = None
    explainer._explanation_plugin_fallbacks = {}
    # Initialize orchestrators so tests can call methods that delegate to them
    explainer._explanation_orchestrator = ExplanationOrchestrator(explainer)
    explainer._prediction_orchestrator = PredictionOrchestrator(explainer)
    return explainer


def test_coerce_plugin_override_supports_multiple_sources():
    explainer = _stub_explainer()

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


def test_build_explanation_chain_merges_overrides(monkeypatch):
    explainer = _stub_explainer()
    explainer._explanation_plugin_overrides["factual"] = "tests.override"
    explainer._pyproject_explanations = {
        "factual": "tests.pyproject",
        "factual_fallbacks": ["tests.pyproject.fallback"],
    }

    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", " env.direct ")
    monkeypatch.setenv(
        "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS",
        "env.one, env.shared, env.two",
    )

    descriptors = {
        "tests.override": types.SimpleNamespace(metadata={"fallbacks": ("env.shared",)}),
        "tests.pyproject": types.SimpleNamespace(
            metadata={"fallbacks": ("tests.metadata.fallback",)}
        ),
    }

    # Patch in the explain orchestrator module where the function is directly imported
    from calibrated_explanations.core.explain import orchestrator as explain_orchestrator_module
    monkeypatch.setattr(
        explain_orchestrator_module,
        "find_explanation_descriptor",
        lambda identifier: descriptors.get(identifier),
    )

    chain = explainer._build_explanation_chain("factual")

    assert chain[0] == "tests.override"
    assert "env.direct" in chain
    assert chain.count("env.shared") == 1
    assert "tests.metadata.fallback" in chain
    assert chain[-1] == "core.explanation.factual"


def test_check_explanation_runtime_metadata_reports_errors():
    explainer = _stub_explainer(mode="classification")

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


def test_check_interval_runtime_metadata_validates_requirements():
    explainer = _stub_explainer(mode="regression")

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


def test_ensure_interval_runtime_state_populates_defaults():
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    # Initialize orchestrators so the delegation method works
    explainer._prediction_orchestrator = PredictionOrchestrator(explainer)
    explainer._explanation_orchestrator = ExplanationOrchestrator(explainer)
    explainer._ensure_interval_runtime_state()

    assert explainer._interval_plugin_hints == {}
    assert explainer._interval_plugin_fallbacks == {}
    assert explainer._interval_plugin_identifiers == {"default": None, "fast": None}
    assert explainer._telemetry_interval_sources == {"default": None, "fast": None}
    assert explainer._interval_preferred_identifier == {"default": None, "fast": None}
    assert explainer._interval_context_metadata == {"default": {}, "fast": {}}


def test_build_interval_chain_merges_sources_and_metadata(monkeypatch):
    explainer = _stub_explainer()
    explainer._interval_plugin_override = "tests.override"
    explainer._pyproject_intervals = {
        "default": "tests.pyproject",
        "default_fallbacks": ("tests.pyproject.fallback",),
    }

    monkeypatch.setenv("CE_INTERVAL_PLUGIN", " env.direct ")
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", "env.shared, env.extra")
    descriptors = {
        "tests.override": types.SimpleNamespace(metadata={"fallbacks": ("env.shared",)}),
        "env.direct": types.SimpleNamespace(metadata={"fallbacks": ()}),
        "tests.pyproject": types.SimpleNamespace(
            metadata={"fallbacks": ("tests.metadata.fallback",)}
        ),
    }

    # Patch in the prediction orchestrator module where find_interval_descriptor is used
    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_descriptor",
        lambda identifier: descriptors.get(identifier),
    )

    chain = explainer._build_interval_chain(fast=False)

    assert chain[0] == "tests.override"
    assert chain.count("env.shared") == 1
    assert "env.direct" in chain
    assert "tests.metadata.fallback" in chain
    assert chain[-1] == "core.interval.legacy"
    assert explainer._interval_preferred_identifier["default"] == "tests.override"


def test_build_interval_chain_fast_skips_missing_default(monkeypatch):
    explainer = _stub_explainer()
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST", "fast.direct")
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST_FALLBACKS", "fast.extra")

    descriptors = {"fast.direct": types.SimpleNamespace(metadata={"fallbacks": ()})}

    # Patch in the prediction orchestrator module where find_interval_descriptor is used
    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_descriptor",
        lambda identifier: descriptors.get(identifier),
    )

    chain = explainer._build_interval_chain(fast=True)

    assert chain == ("fast.direct", "fast.extra")
    assert explainer._interval_preferred_identifier["fast"] == "fast.direct"


def test_build_plot_style_chain_inserts_defaults_when_legacy_env(monkeypatch):
    explainer = _stub_explainer()
    explainer._plot_style_override = "tests.override"
    explainer._pyproject_plots = {
        "style": "tests.pyproject",
        "style_fallbacks": ("legacy", "tests.pyproject.fallback"),
    }

    monkeypatch.setenv("CE_PLOT_STYLE", " env.direct ")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "env.extra, legacy")

    chain = explainer._build_plot_style_chain()

    assert chain[0] == "tests.override"
    assert "env.direct" in chain
    assert chain.count("legacy") == 1
    assert "plot_spec.default" in chain
    legacy_index = chain.index("legacy")
    assert chain[legacy_index - 1] == "plot_spec.default"


def test_gather_interval_hints_merges_modes():
    explainer = _stub_explainer()
    explainer._interval_plugin_hints = {
        "fast": ("fast.hint",),
        "factual": ("hint.one", "shared"),
        "alternative": ("shared", "hint.two"),
    }

    assert explainer._gather_interval_hints(fast=True) == ("fast.hint",)
    assert explainer._gather_interval_hints(fast=False) == ("hint.one", "shared", "hint.two")


def test_instantiate_plugin_handles_multiple_paths(monkeypatch):
    explainer = _stub_explainer()

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


def test_build_plot_style_chain_inserts_defaults(monkeypatch):
    explainer = _stub_explainer()
    explainer._plot_style_override = None
    explainer._pyproject_plots = {}

    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)
    monkeypatch.setenv("CE_PLOT_STYLE", " legacy ")

    chain = explainer._build_plot_style_chain()

    assert chain[0] == "plot_spec.default"
    assert chain[1] == "legacy"
    assert chain.count("legacy") == 1


def test_build_plot_style_chain_appends_legacy_once(monkeypatch):
    explainer = _stub_explainer()
    explainer._plot_style_override = "plot_spec.default"
    explainer._pyproject_plots = {"style_fallbacks": ("modern", "plot_spec.default")}

    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)

    chain = explainer._build_plot_style_chain()

    assert chain[0] == "plot_spec.default"
    assert chain[-1] == "legacy"
    assert chain.count("plot_spec.default") == 1


def test_resolve_interval_plugin_handles_denied_and_success(monkeypatch):
    explainer = _stub_explainer(mode="regression")
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
    monkeypatch.setattr(prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None)
    monkeypatch.setattr(
        prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None
    )

    plugin, identifier = explainer._resolve_interval_plugin(fast=False)

    assert identifier == "ok.plugin"
    assert plugin is descriptor.plugin


def test_resolve_interval_plugin_denied_override_raises(monkeypatch):
    explainer = _stub_explainer(mode="regression")
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


def test_build_interval_context_enriches_metadata():
    explainer = _stub_explainer(mode="regression")
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


def test_get_calibration_summaries_caches_results():
    explainer = _stub_explainer()
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


def test_predict_impl_returns_degraded_arrays_when_suppressed():
    explainer = _stub_explainer(mode="regression")
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
