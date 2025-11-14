"""Tests for lightweight helpers in :mod:`calibrated_explainer`."""

from __future__ import annotations

from typing import Any, Dict


import numpy as np

import pytest

from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    EXPLANATION_PROTOCOL_VERSION,
)
from calibrated_explanations.core.config_helpers import (
    coerce_string_tuple as _coerce_string_tuple,
    read_pyproject_section as _read_pyproject_section,
    split_csv as _split_csv,
)
from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor
from calibrated_explanations.core.exceptions import ConfigurationError


class _DummyBridge:
    """Simple bridge to validate monitoring behaviour."""

    def __init__(self) -> None:
        self.predictions: Dict[str, Any] = {
            "predict": np.array([0.1, 0.9]),
            "predict_interval": (np.array([0.2]), np.array([0.8])),
            "predict_proba": np.array([[0.2, 0.8]]),
        }

    def predict(self, x: Any, *, mode: str, task: str, bins: Any | None = None) -> Any:
        return self.predictions["predict"], mode, task, bins

    def predict_interval(self, x: Any, *, task: str, bins: Any | None = None) -> Any:
        return self.predictions["predict_interval"], task, bins

    def predict_proba(self, x: Any, bins: Any | None = None) -> Any:
        return self.predictions["predict_proba"], bins


def test_read_pyproject_section(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [tool.calibrated_explanations.explanations]
        factual = "py.identifier"
        factual_fallbacks = ["fb.one", "", "fb.two"]
        """.strip()
    )
    monkeypatch.chdir(tmp_path)

    result = _read_pyproject_section(("tool", "calibrated_explanations", "explanations"))

    assert result == {
        "factual": "py.identifier",
        "factual_fallbacks": ["fb.one", "", "fb.two"],
    }


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ()),
        ("", ()),
        ("a, b ,c", ("a", "b", "c")),
    ],
)
def test_split_csv(value, expected):
    assert _split_csv(value) == expected


def test_coerce_string_tuple_handles_iterables():
    assert _coerce_string_tuple("value") == ("value",)
    assert _coerce_string_tuple(["x", "", "y", 1, None]) == ("x", "y")


def test_predict_bridge_monitor_tracks_usage():
    """Test that PredictBridgeMonitor correctly tracks bridge method calls."""
    bridge = _DummyBridge()
    monitor = PredictBridgeMonitor(bridge)

    predict_result = monitor.predict(np.array([[1.0]]), mode="factual", task="classification")
    interval_result = monitor.predict_interval(np.array([[1.0]]), task="classification")
    proba_result = monitor.predict_proba(np.array([[1.0]]))

    assert monitor.calls == ("predict", "predict_interval", "predict_proba")
    assert monitor.used is True
    # Ensure the wrapped bridge is called transparently.
    assert predict_result[0] is bridge.predictions["predict"]
    assert interval_result[0] is bridge.predictions["predict_interval"]
    assert proba_result[0] is bridge.predictions["predict_proba"]


def _make_explainer_stub() -> CalibratedExplainer:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer._explanation_plugin_overrides = {
        mode: None for mode in ("factual", "alternative", "fast")
    }
    explainer._pyproject_explanations = {}
    explainer._pyproject_intervals = {}
    explainer._pyproject_plots = {}
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._plot_style_override = None
    explainer.mode = "classification"
    explainer.bins = None
    return explainer


def test_build_explanation_chain_resolves_sources(monkeypatch):
    explainer = _make_explainer_stub()
    explainer._explanation_plugin_overrides["factual"] = "override.id"
    explainer._pyproject_explanations = {
        "factual": "py.identifier",
        "factual_fallbacks": ("py.fb1", "py.fb2"),
    }

    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", " env.identifier ")
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", "env.fb1, env.fb2")

    class _Descriptor:
        def __init__(self, metadata: Dict[str, Any]):
            self.metadata = metadata

    descriptor_map = {
        "override.id": _Descriptor({"fallbacks": ("env.fb1", "override.fb")}),
        "env.identifier": _Descriptor({"fallbacks": ()}),
        "env.fb1": _Descriptor({"fallbacks": ("env.fb1.extra",)}),
        "py.identifier": _Descriptor({"fallbacks": ("py.fb3",)}),
    }

    def fake_find_descriptor(identifier: str):
        # Normalise whitespace to match production lookup behaviour.
        identifier = identifier.strip()
        return descriptor_map.get(identifier)

    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.find_explanation_descriptor",
        fake_find_descriptor,
    )
    # Also patch in the orchestrator module where the function is directly imported
    from calibrated_explanations.core.explain import orchestrator as explain_orch
    monkeypatch.setattr(explain_orch, "find_explanation_descriptor", fake_find_descriptor)

    chain = explainer._build_explanation_chain("factual")

    assert chain[0] == "override.id"
    # Deduplicated fallbacks should appear only once in order of discovery.
    assert "env.fb1" in chain and chain.count("env.fb1") == 1
    # The metadata fallback appended during override processing should be present.
    assert "override.fb" in chain
    # Default identifier is always appended when available.
    assert chain[-1] == "core.explanation.factual"


def test_build_interval_chain_tracks_preferred_identifier(monkeypatch):
    explainer = _make_explainer_stub()
    explainer._interval_plugin_override = "override.interval"
    explainer._pyproject_intervals = {
        "default": "py.interval",
        "default_fallbacks": ("py.ifb1",),
        "fast": "py.fast",
        "fast_fallbacks": ("py.fast.fb",),
    }

    monkeypatch.setenv("CE_INTERVAL_PLUGIN", " env.interval ")
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", "env.ifb1")
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST", "fast.interval")

    class _Descriptor:
        def __init__(self, metadata: Dict[str, Any]):
            self.metadata = metadata

    descriptor_map = {
        "override.interval": _Descriptor({"fallbacks": ("override.ifb",)}),
        "env.interval": _Descriptor({"fallbacks": ()}),
        "env.ifb1": _Descriptor({"fallbacks": ()}),
        "py.interval": _Descriptor({"fallbacks": ("py.ifb2",)}),
        "fast.interval": _Descriptor({"fallbacks": ()}),
        "py.fast": _Descriptor({"fallbacks": ()}),
    }

    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.find_interval_descriptor",
        lambda identifier: descriptor_map.get(identifier.strip()),
    )
    # Also patch in the prediction orchestrator module
    from calibrated_explanations.core.prediction import orchestrator as pred_orch
    monkeypatch.setattr(
        pred_orch,
        "find_interval_descriptor",
        lambda identifier: descriptor_map.get(identifier.strip()),
    )

    default_chain = explainer._build_interval_chain(fast=False)
    assert default_chain[0] == "override.interval"
    assert explainer._interval_preferred_identifier["default"] == "override.interval"
    assert default_chain[-1] == "core.interval.legacy"

    # For the fast chain simulate missing default descriptor to exercise the skip branch.
    def find_with_skip(identifier: str):
        identifier = identifier.strip()
        if identifier == "core.interval.fast":
            return None
        return descriptor_map.get(identifier)

    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.find_interval_descriptor",
        find_with_skip,
    )
    monkeypatch.setattr(
        pred_orch,
        "find_interval_descriptor",
        find_with_skip,
    )

    fast_chain = explainer._build_interval_chain(fast=True)
    assert fast_chain[0] == "fast.interval"
    assert explainer._interval_preferred_identifier["fast"] == "fast.interval"
    assert "core.interval.fast" not in fast_chain


def test_build_plot_style_chain_merges_sources(monkeypatch):
    explainer = _make_explainer_stub()
    explainer._plot_style_override = "override.style"
    explainer._pyproject_plots = {
        "style": "py.style",
        "style_fallbacks": ("py.fallback", "override.style"),
    }

    monkeypatch.setenv("CE_PLOT_STYLE", " env.style ")
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "env.fallback")

    chain = explainer._build_plot_style_chain()

    assert chain[0] == "override.style"
    # The default plot_spec.default should appear before the legacy entry.
    assert chain.index("plot_spec.default") < chain.index("legacy")
    # Deduplication should prevent duplicates when override repeats in fallbacks.
    assert chain.count("override.style") == 1


def test_coerce_plugin_override_instantiates_callable():
    explainer = _make_explainer_stub()

    class DummyPlugin:
        pass

    def factory():
        return DummyPlugin()

    assert explainer._coerce_plugin_override(None) is None
    assert explainer._coerce_plugin_override("identifier") == "identifier"
    plugin = explainer._coerce_plugin_override(factory)
    assert isinstance(plugin, DummyPlugin)

    class Exploding:
        def __call__(self):
            raise RuntimeError("boom")

    with pytest.raises(ConfigurationError):
        explainer._coerce_plugin_override(Exploding())


def test_check_explanation_runtime_metadata_validations():
    explainer = _make_explainer_stub()

    assert (
        explainer._check_explanation_runtime_metadata(None, identifier="plugin", mode="factual")
        == "plugin: plugin metadata unavailable"
    )

    incompatible = {
        "schema_version": "0",
        "tasks": ["classification"],
        "modes": ["factual"],
        "capabilities": {"requires_predict_proba": False},
    }
    assert "unsupported" in explainer._check_explanation_runtime_metadata(
        incompatible, identifier="plugin", mode="factual"
    )

    missing_tasks = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "modes": ["factual"],
        "capabilities": {"requires_predict_proba": False},
    }
    assert "missing tasks" in explainer._check_explanation_runtime_metadata(
        missing_tasks, identifier="plugin", mode="factual"
    )

    wrong_mode = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ["classification"],
        "modes": ["alternative"],
        "capabilities": {"requires_predict_proba": False},
    }
    assert "does not declare mode" in explainer._check_explanation_runtime_metadata(
        wrong_mode, identifier="plugin", mode="factual"
    )


def test_check_explanation_runtime_metadata_capabilities():
    explainer = _make_explainer_stub()
    metadata = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ["classification"],
        "modes": ["factual"],
        "capabilities": ["explain", "mode:factual"],
    }
    message = explainer._check_explanation_runtime_metadata(
        metadata, identifier="plugin", mode="factual"
    )
    assert "missing required capabilities" in message


def test_ensure_interval_runtime_state_creates_defaults():
    explainer = _make_explainer_stub()
    explainer.__dict__.pop("_interval_plugin_hints", None)
    explainer.__dict__.pop("_interval_plugin_fallbacks", None)
    explainer.__dict__.pop("_interval_plugin_identifiers", None)
    explainer.__dict__.pop("_telemetry_interval_sources", None)
    explainer.__dict__.pop("_interval_preferred_identifier", None)
    explainer.__dict__.pop("_interval_context_metadata", None)

    explainer._ensure_interval_runtime_state()

    assert explainer._interval_plugin_hints == {}
    assert explainer._interval_plugin_fallbacks == {}
    assert explainer._interval_plugin_identifiers == {"default": None, "fast": None}
    assert explainer._telemetry_interval_sources == {"default": None, "fast": None}
    assert explainer._interval_preferred_identifier == {"default": None, "fast": None}
    assert explainer._interval_context_metadata == {"default": {}, "fast": {}}


def test_instantiate_plugin_prefers_fresh_instances():
    explainer = _make_explainer_stub()

    class Proto:
        plugin_meta = {}

        def __call__(self):
            return self

    proto = Proto()
    assert explainer._instantiate_plugin(None) is None
    assert explainer._instantiate_plugin(proto) is proto

    class RequiresArgs:
        def __init__(self, value):
            self.value = value

    original = RequiresArgs(3)
    cloned = explainer._instantiate_plugin(original)
    assert isinstance(cloned, RequiresArgs)
    assert cloned is not original
    assert cloned.value == 3


def test_gather_interval_hints_merges_modes():
    explainer = _make_explainer_stub()
    explainer._interval_plugin_hints = {
        "factual": ("a", "b"),
        "alternative": ("b", "c"),
        "fast": ("fast-only",),
    }

    default_hints = explainer._gather_interval_hints(fast=False)
    fast_hints = explainer._gather_interval_hints(fast=True)

    assert default_hints == ("a", "b", "c")
    assert fast_hints == ("fast-only",)


def test_check_interval_runtime_metadata_validations():
    explainer = _make_explainer_stub()

    assert (
        explainer._check_interval_runtime_metadata(None, identifier="interval", fast=False)
        == "interval: interval metadata unavailable"
    )

    incompatible = {
        "schema_version": 2,
    }
    assert "unsupported" in explainer._check_interval_runtime_metadata(
        incompatible, identifier="interval", fast=False
    )

    missing_modes = {
        "schema_version": 1,
        "capabilities": ["interval:classification"],
    }
    assert "missing modes" in explainer._check_interval_runtime_metadata(
        missing_modes, identifier="interval", fast=False
    )

    missing_cap = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:regression"],
    }
    assert "missing capability" in explainer._check_interval_runtime_metadata(
        missing_cap, identifier="interval", fast=False
    )

    not_fast = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:classification"],
        "fast_compatible": False,
    }
    assert "not marked fast_compatible" in explainer._check_interval_runtime_metadata(
        not_fast, identifier="interval", fast=True
    )

    requires_bins = {
        "schema_version": 1,
        "modes": ["classification"],
        "capabilities": ["interval:classification"],
        "fast_compatible": True,
        "requires_bins": True,
    }
    assert "requires bins" in explainer._check_interval_runtime_metadata(
        requires_bins, identifier="interval", fast=True
    )
