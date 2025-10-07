from __future__ import annotations

import types

import pytest

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import ConfigurationError


def _make_explainer() -> CalibratedExplainer:
    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer.mode = "regression"
    explainer.bins = None
    explainer._fast_interval_plugin_override = None
    explainer._interval_plugin_override = None
    explainer._interval_plugin_fallbacks = {"default": (), "fast": ()}
    explainer._interval_plugin_hints = {}
    explainer._interval_preferred_identifier = {"default": None, "fast": None}
    explainer._telemetry_interval_sources = {"default": None, "fast": None}
    return explainer


def test_resolve_interval_plugin_returns_direct_override(monkeypatch):
    explainer = _make_explainer()
    override = types.SimpleNamespace(plugin_meta={"name": "direct"})
    explainer._interval_plugin_override = override

    ensure_calls: list[str] = []
    monkeypatch.setattr(
        explainer_module,
        "ensure_builtin_plugins",
        lambda: ensure_calls.append("ensure"),
    )

    plugin, identifier = explainer._resolve_interval_plugin(fast=False)

    assert plugin is override
    assert identifier == "direct"
    assert ensure_calls == ["ensure"]


def test_resolve_interval_plugin_rejects_non_fast_metadata(monkeypatch):
    explainer = _make_explainer()
    explainer._fast_interval_plugin_override = "core.interval.test"
    explainer._interval_plugin_fallbacks["fast"] = ("core.interval.test",)

    metadata = {
        "schema_version": 1,
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
    }
    descriptor = types.SimpleNamespace(metadata=metadata, plugin=object(), trusted=True)

    monkeypatch.setattr(explainer_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        explainer_module, "find_interval_descriptor", lambda identifier: descriptor
    )

    with pytest.raises(ConfigurationError) as exc:
        explainer._resolve_interval_plugin(fast=True)

    assert "not marked fast_compatible" in str(exc.value)


def test_resolve_interval_plugin_reports_aggregated_errors(monkeypatch):
    explainer = _make_explainer()
    explainer._interval_plugin_fallbacks["default"] = ("missing", "badmeta")

    metadata = {
        "schema_version": 1,
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
        "requires_bins": True,
    }
    descriptors = {
        "badmeta": types.SimpleNamespace(metadata=metadata, plugin=object(), trusted=True)
    }

    monkeypatch.setattr(explainer_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        explainer_module,
        "find_interval_descriptor",
        lambda identifier: descriptors.get(identifier),
    )
    monkeypatch.setattr(explainer_module, "find_interval_plugin_trusted", lambda identifier: None)
    monkeypatch.setattr(explainer_module, "find_interval_plugin", lambda identifier: None)

    with pytest.raises(ConfigurationError) as exc:
        explainer._resolve_interval_plugin(fast=False)

    message = str(exc.value)
    assert "Unable to resolve interval plugin for default mode" in message
    assert "missing: not registered" in message
    assert "badmeta: requires bins but explainer has none configured" in message


def test_resolve_interval_plugin_uses_hints_and_instantiates(monkeypatch):
    explainer = _make_explainer()
    explainer._interval_plugin_fallbacks["default"] = ("fallback.one",)

    metadata = {
        "schema_version": 1,
        "modes": ("regression",),
        "capabilities": ("interval:regression",),
    }
    descriptors = {
        "hinted": types.SimpleNamespace(metadata=metadata, plugin=None, trusted=False),
        "fallback.one": types.SimpleNamespace(metadata=metadata, plugin=None, trusted=False),
    }

    prototype = object()

    monkeypatch.setattr(explainer_module, "ensure_builtin_plugins", lambda: None)
    monkeypatch.setattr(
        explainer_module,
        "find_interval_descriptor",
        lambda identifier: descriptors.get(identifier),
    )
    monkeypatch.setattr(
        explainer_module,
        "find_interval_plugin_trusted",
        lambda identifier: prototype if identifier == "hinted" else None,
    )
    monkeypatch.setattr(explainer_module, "find_interval_plugin", lambda identifier: None)

    instantiated: list[object] = []

    def fake_instantiate(self, plugin):
        instantiated.append(plugin)
        return {"wrapped": plugin}

    monkeypatch.setattr(
        CalibratedExplainer,
        "_instantiate_plugin",
        fake_instantiate,
    )

    plugin, identifier = explainer._resolve_interval_plugin(fast=False, hints=("hinted",))

    assert plugin == {"wrapped": prototype}
    assert identifier == "hinted"
    assert instantiated == [prototype]


def test_build_interval_chain_includes_pyproject_entries(monkeypatch):
    explainer = _make_explainer()
    explainer._pyproject_intervals = {
        "default": "tests.interval.pyproject",
        "default_fallbacks": ["tests.interval.secondary"],
    }

    descriptors = {
        "tests.interval.pyproject": types.SimpleNamespace(
            metadata={"fallbacks": ("tests.interval.metadata",)},
        )
    }

    monkeypatch.setattr(
        explainer_module,
        "find_interval_descriptor",
        lambda identifier: descriptors.get(identifier),
    )

    chain = explainer._build_interval_chain(fast=False)

    assert chain == (
        "tests.interval.pyproject",
        "tests.interval.metadata",
        "tests.interval.secondary",
        "core.interval.legacy",
    )
    assert explainer._interval_preferred_identifier["default"] is None


def test_build_plot_style_chain_includes_pyproject_entries():
    explainer = _make_explainer()
    explainer._plot_style_override = None
    explainer._pyproject_plots = {
        "style": "tests.plot.pyproject",
        "style_fallbacks": ["tests.plot.secondary", "legacy"],
    }

    chain = explainer._build_plot_style_chain()

    assert chain == (
        "tests.plot.pyproject",
        "tests.plot.secondary",
        "legacy",
    )
