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
