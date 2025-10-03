import importlib
import types

import pytest

from calibrated_explanations.plugins import registry


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
    plugin_meta = {"schema_version": 1, "capabilities": ["explain"], "name": "dummy"}

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

    # trusting unregistered plugin raises
    with pytest.raises(ValueError):
        registry.trust_plugin(object())

    # trust and find
    registry.trust_plugin(p)
    trusted = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert p in trusted

    # untrust works
    registry.untrust_plugin(p)
    trusted2 = registry.find_for_trusted(types.SimpleNamespace(is_dummy=True))
    assert p not in trusted2

    # cleanup
    registry.unregister(p)


class ExampleExplanationPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain", "explanation:factual"],
        "name": "example.explanation",
        "modes": ["explanation:factual"],
        "dependencies": ["core.interval.legacy"],
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
    descriptor = registry.register_explanation_plugin(
        "core.explanation.example", plugin
    )

    assert descriptor.identifier == "core.explanation.example"
    assert registry.find_explanation_plugin("core.explanation.example") is plugin
    assert (
        registry.find_explanation_plugin_trusted("core.explanation.example")
        is plugin
    )


def test_register_explanation_plugin_requires_modes():
    registry.clear_explanation_plugins()

    class BadExplanationPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "bad",
            "dependencies": [],
            "trust": False,
        }

    with pytest.raises(ValueError):
        registry.register_explanation_plugin("bad", BadExplanationPlugin())


class ExampleIntervalPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["interval:classification"],
        "name": "example.interval",
        "modes": ["classification"],
        "dependencies": [],
        "trust": {"trusted": False},
    }


def test_register_interval_plugin_descriptor():
    registry.clear_interval_plugins()
    descriptor = registry.register_interval_plugin(
        "core.interval.example", ExampleIntervalPlugin()
    )
    assert descriptor.identifier == "core.interval.example"
    assert (
        registry.find_interval_plugin("core.interval.example")
        is descriptor.plugin
    )
    assert registry.find_interval_plugin_trusted("core.interval.example") is None


def test_register_interval_plugin_requires_modes():
    registry.clear_interval_plugins()

    class BadIntervalPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["interval:classification"],
            "name": "bad.interval",
            "dependencies": [],
            "trust": False,
        }

    with pytest.raises(ValueError):
        registry.register_interval_plugin("bad.interval", BadIntervalPlugin())


class ExamplePlotPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["plot:builder"],
        "name": "example.plot",
        "style": "legacy",
        "dependencies": ["matplotlib"],
        "trust": True,
        "output_formats": ["png"],
    }


def test_register_plot_plugin_descriptor():
    registry.clear_plot_plugins()
    descriptor = registry.register_plot_plugin("core.plot.example", ExamplePlotPlugin())
    assert descriptor.identifier == "core.plot.example"
    assert registry.find_plot_plugin("core.plot.example") is descriptor.plugin
    assert (
        registry.find_plot_plugin_trusted("core.plot.example") is descriptor.plugin
    )


def test_register_plot_plugin_requires_style():
    registry.clear_plot_plugins()

    class BadPlotPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:builder"],
            "name": "bad.plot",
            "dependencies": [],
            "trust": False,
        }

    with pytest.raises(ValueError):
        registry.register_plot_plugin("bad.plot", BadPlotPlugin())
