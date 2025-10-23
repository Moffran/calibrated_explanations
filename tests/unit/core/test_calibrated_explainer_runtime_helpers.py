"""Unit tests covering CalibratedExplainer runtime helper utilities."""

from __future__ import annotations

import types

import pytest

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.calibrated_explainer import (
    _PredictBridgeMonitor,
    CalibratedExplainer,
    EXPLANATION_PROTOCOL_VERSION,
)
from calibrated_explanations.core.exceptions import ConfigurationError


def _stub_explainer(mode: str = "classification") -> CalibratedExplainer:
    """Construct a lightweight explainer instance for unit tests."""

    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer.mode = mode
    explainer.bins = None
    explainer._interval_plugin_hints = {}
    explainer._interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._telemetry_interval_sources = {"default": None, "fast": None}
    explainer._interval_context_metadata = {"default": {}, "fast": {}}
    explainer._explanation_plugin_overrides = {
        mode: None for mode in ("factual", "alternative", "fast")
    }
    explainer._pyproject_explanations = {}
    explainer._pyproject_plots = {}
    explainer._explanation_plugin_fallbacks = {}
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

    monkeypatch.setattr(
        explainer_module,
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
    explainer._ensure_interval_runtime_state()

    assert explainer._interval_plugin_hints == {}
    assert explainer._interval_plugin_fallbacks == {}
    assert explainer._interval_plugin_identifiers == {"default": None, "fast": None}
    assert explainer._telemetry_interval_sources == {"default": None, "fast": None}
    assert explainer._interval_preferred_identifier == {"default": None, "fast": None}
    assert explainer._interval_context_metadata == {"default": {}, "fast": {}}
