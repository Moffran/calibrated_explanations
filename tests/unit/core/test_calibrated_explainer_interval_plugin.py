from __future__ import annotations

from types import SimpleNamespace

import pytest

from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    ConfigurationError,
)
from calibrated_explanations.core.prediction import orchestrator as prediction_orchestrator_module


class DummyDescriptor:
    """Descriptor stub matching the plugin registry contract."""

    def __init__(self, *, plugin, metadata=None, trusted=False):
        self.plugin = plugin
        self.metadata = metadata or {}
        self.trusted = trusted


def _make_explainer_stub() -> CalibratedExplainer:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer.mode = "regression"
    explainer.bins = None
    explainer._interval_plugin_override = None
    explainer._fast_interval_plugin_override = None
    explainer._interval_plugin_hints = {}
    explainer._interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer._interval_plugin_identifiers = {"default": None, "fast": None}
    explainer._telemetry_interval_sources = {"default": None, "fast": None}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._interval_context_metadata = {"default": {}, "fast": {}}
    explainer._instantiate_plugin = lambda plugin: plugin
    return explainer


@pytest.fixture(autouse=True)
def _no_builtin_plugins(monkeypatch):
    monkeypatch.setattr(prediction_orchestrator_module, "ensure_builtin_plugins", lambda: None)


def test_resolve_interval_plugin_prefers_override_instance(monkeypatch):
    explainer = _make_explainer_stub()

    class InlinePlugin:
        plugin_meta = {"name": "inline"}

    override = InlinePlugin()
    explainer._interval_plugin_override = override

    plugin, identifier = explainer._resolve_interval_plugin(fast=False)

    assert plugin is override
    assert identifier == "inline"


def test_resolve_interval_plugin_fast_mode_enforces_metadata(monkeypatch):
    explainer = _make_explainer_stub()
    explainer._fast_interval_plugin_override = "fast-plugin"
    explainer._interval_plugin_fallbacks["fast"] = ("fast-plugin",)
    explainer._interval_preferred_identifier["fast"] = "fast-plugin"

    descriptor = DummyDescriptor(
        plugin=SimpleNamespace(),
        metadata={
            "name": "fast-plugin",
            "modes": ("regression",),
            "capabilities": ("interval:regression",),
            # fast_compatible omitted on purpose
        },
    )

    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_descriptor",
        lambda identifier: descriptor if identifier == "fast-plugin" else None,
    )
    monkeypatch.setattr(prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None)
    monkeypatch.setattr(prediction_orchestrator_module, "find_interval_plugin_trusted", lambda identifier: None)

    with pytest.raises(ConfigurationError, match="not marked fast_compatible"):
        explainer._resolve_interval_plugin(fast=True)


def test_resolve_interval_plugin_accumulates_errors(monkeypatch):
    explainer = _make_explainer_stub()
    explainer._interval_plugin_fallbacks["default"] = ("first", "second")

    descriptor_map = {
        "hinted": DummyDescriptor(
            plugin=SimpleNamespace(),
            metadata={
                "name": "hinted",
                "modes": ("regression",),
                "capabilities": (),
            },
        ),
        "second": DummyDescriptor(
            plugin=SimpleNamespace(),
            metadata={
                "name": "second",
                "modes": ("regression",),
                "capabilities": ("interval:regression",),
                "requires_bins": True,
            },
        ),
    }

    def fake_descriptor(identifier):
        return descriptor_map.get(identifier)

    monkeypatch.setattr(prediction_orchestrator_module, "find_interval_descriptor", fake_descriptor)
    monkeypatch.setattr(prediction_orchestrator_module, "find_interval_plugin", lambda identifier: None)

    def fake_find_trusted(identifier):
        if identifier in ("hinted", "second"):
            return SimpleNamespace()
        return None

    monkeypatch.setattr(
        prediction_orchestrator_module,
        "find_interval_plugin_trusted",
        fake_find_trusted,
    )

    with pytest.raises(ConfigurationError) as exc:
        explainer._resolve_interval_plugin(fast=False, hints=("hinted",))

    message = str(exc.value)
    assert "Tried: hinted, first, second" in message
    assert "hinted: missing capability" in message
    assert "first: not registered" in message
    assert "second: requires bins" in message
