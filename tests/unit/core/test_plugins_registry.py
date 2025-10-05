import importlib
import types
from typing import Any

import pytest

from calibrated_explanations.plugins import registry


class _FakeEntryPoint:
    def __init__(self, plugin: Any) -> None:
        self.name = plugin.plugin_meta["name"]
        self.module = "tests.plugins.fake"
        self.attr = None
        self.group = registry._ENTRYPOINT_GROUP
        self._plugin = plugin

    def load(self):
        return self._plugin


class _FakeEntryPoints(list):
    def select(self, *, group: str):
        return [entry for entry in self if getattr(entry, "group", None) == group]


def test_register_and_find_example_plugin(tmp_path, monkeypatch):
    # Instead of writing to disk, import the packaged test plugin shipped under tests/plugins
    mod = importlib.import_module("tests.plugins.example_plugin")
    plugin = getattr(mod, "PLUGIN")

    # Start from a clean registry
    registry.clear()

    # Register and find
    registry.register(plugin)
    all_plugins = registry.list_plugins()
    assert plugin in all_plugins

    # find_for should return plugin for supported models
    found = registry.find_for("supported-model")
    assert plugin in found

    # Not supported model should return empty
    assert registry.find_for("unsupported") == ()

    # Mark as trusted and ensure trusted discovery returns it
    registry.trust_plugin(plugin)
    trusted = registry.find_for_trusted("supported-model")
    assert plugin in trusted

    # Untrust and ensure it's not returned by trusted finder
    registry.untrust_plugin(plugin)
    assert plugin not in registry.find_for_trusted("supported-model")

    # Unregister removes the plugin
    registry.unregister(plugin)
    assert plugin not in registry.list_plugins()


def test_env_trust_marks_plugin_trusted(monkeypatch):
    registry.clear()
    registry._ENV_TRUST_CACHE = None
    monkeypatch.setenv("CE_TRUST_PLUGIN", "tests.env")

    class EnvPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "tests.env",
            "version": "0.0-test",
            "provider": "tests",
            "trusted": False,
            "trust": False,
        }

        def supports(self, model):
            return True

        def explain(self, model, X, **kwargs):
            return {}

    plugin = EnvPlugin()
    registry.register(plugin)
    assert plugin in registry.list_plugins(include_untrusted=False)
    registry.unregister(plugin)
    registry._ENV_TRUST_CACHE = None


def test_validate_plugin_meta_rejects_bad_meta():
    class BadPlugin:
        plugin_meta = {"capabilities": ["explain"]}  # missing name and schema_version

        def supports(self, model):
            return False

        def explain(self, model, X, **kwargs):
            return {}

    with pytest.raises(ValueError):
        registry.register(BadPlugin())


class DummyPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "dummy",
        "version": "0.0-test",
        "provider": "tests",
        "trusted": False,
        "trust": False,
    }

    def supports(self, model):
        return getattr(model, "is_dummy", False)

    def explain(self, model, X, **kwargs):
        return {"explained": True}


def test_register_and_trust_flow(tmp_path):
    p = DummyPlugin()
    # ensure clean start
    registry.clear()
    registry.register(p)
    assert p in registry.list_plugins()
    assert p not in registry.list_plugins(include_untrusted=False)

    # trusting unregistered plugin raises
    with pytest.raises(ValueError):
        registry.trust_plugin(object())

    # trust and find
    registry.trust_plugin(p)
    assert p in registry.list_plugins(include_untrusted=False)
    trusted = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert p in trusted

    # untrust works
    registry.untrust_plugin("dummy")
    trusted2 = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert p not in trusted2
    assert p not in registry.list_plugins(include_untrusted=False)

    # cleanup
    registry.unregister(p)


class ExampleExplanationPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain", "task:classification"],
        "name": "example.explanation",
        "version": "0.0-test",
        "provider": "tests",
        "modes": ["factual", "alternative"],
        "tasks": ["classification", "regression"],
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": ("legacy",),
        "fallbacks": ["core.explanation.legacy"],
        "dependencies": ["core.interval.legacy"],
        "trusted": True,
        "trust": {"default": True},
    }

    def supports(self, model):  # pragma: no cover - unused for descriptor tests
        return False

    def explain(self, model, X, **kwargs):  # pragma: no cover - unused
        return {}


def test_register_explanation_plugin_descriptor():
    registry.clear()
    registry.clear_explanation_plugins()
    plugin = ExampleExplanationPlugin()
    descriptor = registry.register_explanation_plugin("core.explanation.example", plugin)

    assert descriptor.identifier == "core.explanation.example"
    assert registry.find_explanation_plugin("core.explanation.example") is plugin
    assert registry.find_explanation_plugin_trusted("core.explanation.example") is plugin
    assert descriptor.metadata["modes"] == ("factual", "alternative")
    assert descriptor.metadata["tasks"] == ("classification", "regression")
    assert descriptor.metadata["interval_dependency"] == ("core.interval.legacy",)
    assert descriptor.metadata["plot_dependency"] == ("legacy",)
    assert descriptor.metadata["fallbacks"] == ("core.explanation.legacy",)


def test_register_explanation_plugin_requires_modes():
    registry.clear_explanation_plugins()

    class BadExplanationPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "bad",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "tasks": "classification",
            "trust": False,
        }

    with pytest.raises(ValueError):
        registry.register_explanation_plugin("bad", BadExplanationPlugin())


def test_register_explanation_plugin_requires_tasks():
    registry.clear_explanation_plugins()

    class NoTasksExplanationPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "bad",  # pragma: no mutate - clarity
            "version": "0.0-test",
            "provider": "tests",
            "modes": ["factual"],
            "dependencies": [],
            "trust": False,
        }

    with pytest.raises(ValueError):
        registry.register_explanation_plugin("bad.tasks", NoTasksExplanationPlugin())


def test_register_explanation_plugin_translates_aliases():
    registry.clear_explanation_plugins()

    class LegacyModePlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "legacy",
            "version": "0.0-test",
            "provider": "tests",
            "modes": ["explanation:factual", "factual"],
            "tasks": "classification",
            "dependencies": [],
            "trust": True,
        }

    with pytest.warns(DeprecationWarning):
        descriptor = registry.register_explanation_plugin("legacy.mode", LegacyModePlugin())

    assert descriptor.metadata["modes"] == ("factual",)


def test_register_explanation_plugin_schema_version_future():
    registry.clear_explanation_plugins()

    class FuturePlugin:
        plugin_meta = {
            "schema_version": 999,
            "capabilities": ["explain"],
            "name": "future",
            "version": "0.0-test",
            "provider": "tests",
            "modes": ["factual"],
            "tasks": "classification",
            "dependencies": [],
            "trust": False,
        }

    with pytest.raises(ValueError):
        registry.register_explanation_plugin("future", FuturePlugin())


def _make_entry_plugin(name: str = "tests.entry"):
    class EntryPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": name,
            "version": "0.0-test",
            "provider": "tests",
            "trusted": False,
            "trust": False,
        }

        def supports(self, model):
            return True

        def explain(self, model, X, **kwargs):
            return {}

    return EntryPlugin()


def test_load_entrypoint_plugins_skips_untrusted(monkeypatch):
    registry.clear()
    registry._WARNED_UNTRUSTED.clear()
    registry._ENV_TRUST_CACHE = None

    plugin = _make_entry_plugin()
    fake_entries = _FakeEntryPoints([_FakeEntryPoint(plugin)])
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda: fake_entries,
    )

    with pytest.warns(RuntimeWarning):
        loaded = registry.load_entrypoint_plugins()

    assert loaded == ()
    assert plugin not in registry.list_plugins()


def test_load_entrypoint_plugins_trusted_by_env(monkeypatch):
    registry.clear()
    registry._WARNED_UNTRUSTED.clear()
    monkeypatch.setenv("CE_TRUST_PLUGIN", "tests.entry")
    registry._ENV_TRUST_CACHE = None

    plugin = _make_entry_plugin()
    fake_entries = _FakeEntryPoints([_FakeEntryPoint(plugin)])
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda: fake_entries,
    )

    loaded = registry.load_entrypoint_plugins()
    assert loaded == (plugin,)
    assert plugin in registry.list_plugins(include_untrusted=False)
    registry.unregister(plugin)
    registry._ENV_TRUST_CACHE = None


class ExampleIntervalPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["interval:classification"],
        "name": "example.interval",
        "version": "0.0-test",
        "provider": "tests",
        "modes": ["classification"],
        "dependencies": [],
        "trusted": False,
        "trust": {"trusted": False},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "legacy",
    }


def test_register_interval_plugin_descriptor():
    registry.clear_interval_plugins()
    descriptor = registry.register_interval_plugin("core.interval.example", ExampleIntervalPlugin())
    assert descriptor.identifier == "core.interval.example"
    assert registry.find_interval_plugin("core.interval.example") is descriptor.plugin
    assert registry.find_interval_plugin_trusted("core.interval.example") is None


def test_register_interval_plugin_requires_modes():
    registry.clear_interval_plugins()

    class BadIntervalPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["interval:classification"],
            "name": "bad.interval",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "trust": False,
            "fast_compatible": False,
            "requires_bins": False,
            "confidence_source": "legacy",
        }

    with pytest.raises(ValueError):
        registry.register_interval_plugin("bad.interval", BadIntervalPlugin())


class ExamplePlotBuilder:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["plot:builder"],
        "name": "example.plot.builder",
        "version": "0.0-test",
        "provider": "tests",
        "style": "example",
        "dependencies": ["matplotlib"],
        "trusted": True,
        "trust": True,
        "output_formats": ["png"],
        "legacy_compatible": True,
    }


class ExamplePlotRenderer:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["plot:renderer"],
        "name": "example.plot.renderer",
        "version": "0.0-test",
        "provider": "tests",
        "dependencies": ["matplotlib"],
        "trusted": True,
        "trust": True,
        "output_formats": ["png"],
        "supports_interactive": False,
    }


def test_register_plot_components():
    registry.clear_plot_plugins()
    builder = registry.register_plot_builder("core.plot.example", ExamplePlotBuilder())
    renderer = registry.register_plot_renderer("core.plot.example", ExamplePlotRenderer())
    style = registry.register_plot_style(
        "example",
        metadata={
            "style": "example",
            "builder_id": builder.identifier,
            "renderer_id": renderer.identifier,
            "fallbacks": (),
        },
    )
    assert builder.identifier == "core.plot.example"
    assert renderer.identifier == "core.plot.example"
    assert style.identifier == "example"
    assert registry.find_plot_builder("core.plot.example") is builder.builder
    assert registry.find_plot_renderer("core.plot.example") is renderer.renderer
    assert registry.find_plot_style_descriptor("example") is style


def test_register_plot_builder_requires_style():
    registry.clear_plot_plugins()

    class BadPlotPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:builder"],
            "name": "bad.plot",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "trust": False,
            "output_formats": ["png"],
            "legacy_compatible": False,
        }

    with pytest.raises(ValueError):
        registry.register_plot_builder("bad.plot", BadPlotPlugin())
