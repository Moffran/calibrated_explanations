import importlib
import sys
import types
from typing import Any

import pytest

from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.plugins import registry
from tests.helpers.deprecation import warns_or_raises, deprecations_error_enabled
from tests.support.registry_helpers import (
    clear_explanation_plugins,
    clear_env_trust_cache,
    clear_interval_plugins,
    clear_plot_plugins,
    clear_trust_warnings,
    get_entrypoint_group,
)


class FakeEntryPoint:
    def __init__(self, plugin: Any) -> None:
        self.name = plugin.plugin_meta["name"]
        self.module = "tests.plugins.fake"
        self.attr = None
        self.group = get_entrypoint_group()
        self.plugin_instance = plugin

    def load(self):
        return self.plugin_instance


class FakeEntryPoints(list):
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

    def explain(self, model, x, **kwargs):
        return {"explained": True}


def test_register_and_trust_flow(tmp_path):
    p = DummyPlugin()
    # ensure clean start
    registry.clear()
    registry.register(p)
    assert p in registry.list_plugins()
    assert p not in registry.list_plugins(include_untrusted=False)

    # trusting unregistered plugin raises
    with pytest.raises(ValidationError):
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

    def explain(self, model, x, **kwargs):  # pragma: no cover - unused
        return {}


def test_register_explanation_plugin_descriptor():
    registry.clear()
    clear_explanation_plugins()
    plugin = ExampleExplanationPlugin()
    descriptor = registry.register_explanation_plugin("core.explanation.example", plugin)
    registry.mark_explanation_trusted("core.explanation.example")

    assert descriptor.identifier == "core.explanation.example"
    assert registry.find_explanation_plugin("core.explanation.example") is plugin
    assert registry.find_explanation_plugin_trusted("core.explanation.example") is plugin
    assert descriptor.metadata["modes"] == ("factual", "alternative")
    assert descriptor.metadata["tasks"] == ("classification", "regression")
    assert descriptor.metadata["interval_dependency"] == ("core.interval.legacy",)
    assert descriptor.metadata["plot_dependency"] == ("legacy",)
    assert descriptor.metadata["fallbacks"] == ("core.explanation.legacy",)


def test_register_explanation_plugin_requires_tasks():
    clear_explanation_plugins()

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

    with pytest.raises(ValidationError):
        registry.register_explanation_plugin("bad.tasks", NoTasksExplanationPlugin())


def test_register_explanation_plugin_translates_aliases():
    clear_explanation_plugins()

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

    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            registry.register_explanation_plugin("legacy.mode", LegacyModePlugin())
    else:
        with warns_or_raises():
            descriptor = registry.register_explanation_plugin("legacy.mode", LegacyModePlugin())

        assert descriptor.metadata["modes"] == ("factual",)


def test_register_explanation_plugin_schema_version_future():
    clear_explanation_plugins()

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

    with pytest.raises(ValidationError):
        registry.register_explanation_plugin("future", FuturePlugin())


def make_entry_plugin(name: str = "tests.entry"):
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

        def explain(self, model, x, **kwargs):
            return {}

    return EntryPlugin()


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
    clear_interval_plugins()
    descriptor = registry.register_interval_plugin("core.interval.example", ExampleIntervalPlugin())
    assert descriptor.identifier == "core.interval.example"
    assert registry.find_interval_plugin("core.interval.example") is descriptor.plugin
    assert registry.find_interval_plugin_trusted("core.interval.example") is None


def test_register_interval_plugin_requires_modes():
    clear_interval_plugins()

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

    with pytest.raises(ValidationError):
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


def test_register_plot_plugin_combines_builder_and_renderer():
    clear_plot_plugins()

    class CombinedPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:builder", "plot:renderer"],
            "name": "example.plot.combined",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": ["matplotlib"],
            "trusted": True,
            "trust": True,
            "output_formats": ["png"],
            "legacy_compatible": True,
            "supports_interactive": False,
            "style": "example.plot.combined",
        }

        def build(self, *args, **kwargs):
            return ("build", args, kwargs)

        def render(self, *args, **kwargs):
            return ("render", args, kwargs)

    plugin = CombinedPlugin()
    try:
        descriptor = registry.register_plot_builder(
            "example.plot.combined", plugin, source="builtin"
        )
        registry.register_plot_renderer("example.plot.combined", plugin, source="builtin")
        registry.register_plot_style(
            "example.plot.combined",
            metadata={
                "style": "example.plot.combined",
                "builder_id": "example.plot.combined",
                "renderer_id": "example.plot.combined",
                "fallbacks": (),
            },
        )
        assert descriptor.identifier == "example.plot.combined"
        assert registry.find_plot_builder("example.plot.combined") is plugin
        assert registry.find_plot_renderer("example.plot.combined") is plugin

        combined = registry.find_plot_plugin("example.plot.combined")
        assert combined is not None
        assert combined.plugin_meta is plugin.plugin_meta
        assert combined.build(1, foo=2) == ("build", (1,), {"foo": 2})
        assert combined.render(3, bar=4) == ("render", (3,), {"bar": 4})

        trusted_combined = registry.find_plot_plugin_trusted("example.plot.combined")
        assert trusted_combined is not None
        assert trusted_combined.build(5) == ("build", (5,), {})
    finally:
        clear_plot_plugins()


def test_list_descriptors_and_trust_management(monkeypatch):
    registry.clear()
    clear_explanation_plugins()
    clear_interval_plugins()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)

    class AltExplanationPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "example.explanation.alt",
            "version": "0.0-test",
            "provider": "tests",
            "modes": ["factual"],
            "tasks": ["classification"],
            "dependencies": [],
            "trusted": False,
            "trust": False,
        }

    class TrustedIntervalPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["interval:regression"],
            "name": "example.interval.trusted",
            "version": "0.0-test",
            "provider": "tests",
            "modes": ["regression"],
            "dependencies": [],
            "trusted": True,
            "trust": True,
            "fast_compatible": True,
            "requires_bins": False,
            "confidence_source": "calibrated",
        }

    try:
        explainer = ExampleExplanationPlugin()
        alt_explainer = AltExplanationPlugin()
        descriptor = registry.register_explanation_plugin("core.explanation.example", explainer)
        alt_descriptor = registry.register_explanation_plugin("core.explanation.alt", alt_explainer)

        # Trust the plugins that should be trusted
        registry.mark_explanation_trusted("core.explanation.example")

        descriptor = registry.find_explanation_descriptor("core.explanation.example")
        assert descriptor.trusted is True
        assert alt_descriptor.trusted is False

        interval_descriptor = registry.register_interval_plugin(
            "core.interval.trusted",
            TrustedIntervalPlugin(),
        )
        registry.mark_interval_trusted("core.interval.trusted")
        interval_descriptor = registry.find_interval_descriptor("core.interval.trusted")
        untrusted_interval = registry.register_interval_plugin(
            "core.interval.example",
            ExampleIntervalPlugin(),
        )
        assert interval_descriptor.trusted is True
        assert untrusted_interval.trusted is False

        all_expl = registry.list_explanation_descriptors()
        assert [d.identifier for d in all_expl] == [
            "core.explanation.alt",
            "core.explanation.example",
        ]
        trusted_expl = registry.list_explanation_descriptors(trusted_only=True)
        assert [d.identifier for d in trusted_expl] == ["core.explanation.example"]

        registry.mark_explanation_trusted("core.explanation.alt")
        assert registry.find_explanation_plugin_trusted("core.explanation.alt") is alt_explainer
        trusted_after_mark = registry.list_explanation_descriptors(trusted_only=True)
        assert {d.identifier for d in trusted_after_mark} == {
            "core.explanation.alt",
            "core.explanation.example",
        }

        registry.mark_explanation_untrusted("core.explanation.example")
        assert registry.find_explanation_plugin_trusted("core.explanation.example") is None
        trusted_after_untrust = registry.list_explanation_descriptors(trusted_only=True)
        assert [d.identifier for d in trusted_after_untrust] == ["core.explanation.alt"]

        all_intervals = registry.list_interval_descriptors()
        assert {d.identifier for d in all_intervals} == {
            "core.interval.example",
            "core.interval.trusted",
        }
        trusted_intervals = registry.list_interval_descriptors(trusted_only=True)
        assert [d.identifier for d in trusted_intervals] == ["core.interval.trusted"]
    finally:
        registry.clear()
        clear_explanation_plugins()
        clear_interval_plugins()


def test_load_entrypoint_plugins_error_branches(monkeypatch):
    registry.clear()
    clear_trust_warnings()
    clear_env_trust_cache()

    class NoMetaPlugin:
        pass

    class InvalidPlugin:
        plugin_meta = {"name": "invalid", "data_modalities": ("tabular",)}

    class UntrustedPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "entry.untrusted",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "data_modalities": ("tabular",),
            "trust": False,
        }

    class TrustedPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "entry.trusted",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "data_modalities": ("tabular",),
            "trust": True,
        }

    attr_module_name = "tests.plugins.attr_entry"
    attr_plugin = TrustedPlugin()
    attr_plugin.plugin_meta = dict(attr_plugin.plugin_meta)
    attr_plugin.plugin_meta["name"] = "entry.attr"
    sys.modules[attr_module_name] = types.SimpleNamespace(attr_plugin=attr_plugin)

    class LoaderEntryPoint:
        def __init__(self, name, loader=None, module=None, attr=None):
            self.name = name
            self.module = module or "tests.plugins.fake"
            self.attr = attr
            self.group = get_entrypoint_group()
            self.loader_func = loader

        def load(self):
            if self.loader_func is not None:
                return self.loader_func()
            module = importlib.import_module(self.module)
            return getattr(module, self.attr)

    failing = LoaderEntryPoint(
        "failing", loader=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    no_meta = LoaderEntryPoint("no_meta", loader=lambda: NoMetaPlugin())
    invalid = LoaderEntryPoint("invalid", loader=lambda: InvalidPlugin())
    untrusted = LoaderEntryPoint("untrusted", loader=lambda: UntrustedPlugin())
    trusted = LoaderEntryPoint("trusted", loader=lambda: TrustedPlugin())
    attr_entry = LoaderEntryPoint("attr", loader=lambda: attr_plugin)

    entry_points = FakeEntryPoints([failing, no_meta, invalid, untrusted, trusted, attr_entry])
    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: entry_points)
    monkeypatch.setenv("CE_TRUST_PLUGIN", "trusted,attr")
    clear_env_trust_cache()

    try:
        with pytest.warns(UserWarning):
            loaded = registry.load_entrypoint_plugins()
        assert {plugin.plugin_meta["name"] for plugin in loaded} == {"entry.trusted", "entry.attr"}
        # untrusted plugin should not be registered when include_untrusted is False
        assert all(plugin in registry.list_plugins(include_untrusted=False) for plugin in loaded)
    finally:
        registry.clear()
        clear_trust_warnings()
        clear_env_trust_cache()
        sys.modules.pop(attr_module_name, None)
