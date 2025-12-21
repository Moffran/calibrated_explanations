import importlib
import hashlib
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest

from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.plugins import registry
from tests.helpers.deprecation import warns_or_raises, deprecations_error_enabled


class FakeEntryPoint:
    def __init__(self, plugin: Any) -> None:
        self.name = plugin.plugin_meta["name"]
        self.module = "tests.plugins.fake"
        self.attr = None
        self.group = registry._ENTRYPOINT_GROUP
        self._plugin = plugin

    def load(self):
        return self._plugin


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

        def explain(self, model, x, **kwargs):
            return {}

    plugin = EnvPlugin()
    registry.register(plugin)
    assert plugin in registry.list_plugins(include_untrusted=False)
    registry.unregister(plugin)
    registry._ENV_TRUST_CACHE = None


def test_is_identifier_denied(monkeypatch):
    monkeypatch.setenv("CE_DENY_PLUGIN", "alpha , beta")
    assert registry.is_identifier_denied("alpha")
    assert registry.is_identifier_denied("beta")
    assert not registry.is_identifier_denied("gamma")


def test_validate_plugin_meta_rejects_bad_meta():
    class BadPlugin:
        plugin_meta = {"capabilities": ["explain"]}  # missing name and schema_version

        def supports(self, model):
            return False

        def explain(self, model, x, **kwargs):
            return {}

    with pytest.raises(ValidationError):
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

    def explain(self, model, x, **kwargs):
        return {"explained": True}


def test_register_and_trust_flow(tmp_path):
    from calibrated_explanations.utils.exceptions import ValidationError

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
    from calibrated_explanations.utils.exceptions import ValidationError

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

    with pytest.raises(ValidationError):
        registry.register_explanation_plugin("bad", BadExplanationPlugin())


def test_register_explanation_plugin_requires_tasks():
    from calibrated_explanations.utils.exceptions import ValidationError

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

    with pytest.raises(ValidationError):
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

    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            registry.register_explanation_plugin("legacy.mode", LegacyModePlugin())
    else:
        with warns_or_raises():
            descriptor = registry.register_explanation_plugin("legacy.mode", LegacyModePlugin())

        assert descriptor.metadata["modes"] == ("factual",)


def test_register_explanation_plugin_schema_version_future():
    from calibrated_explanations.utils.exceptions import ValidationError

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

    with pytest.raises(ValidationError):
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

        def explain(self, model, x, **kwargs):
            return {}

    return EntryPlugin()


def test_load_entrypoint_plugins_skips_untrusted(monkeypatch):
    registry.clear()
    registry._WARNED_UNTRUSTED.clear()
    registry._ENV_TRUST_CACHE = None

    plugin = _make_entry_plugin()
    fake_entries = FakeEntryPoints([FakeEntryPoint(plugin)])
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda: fake_entries,
    )

    with pytest.warns(UserWarning):
        loaded = registry.load_entrypoint_plugins()

    assert loaded == ()
    assert plugin not in registry.list_plugins()


def test_load_entrypoint_plugins_trusted_by_env(monkeypatch):
    registry.clear()
    registry._WARNED_UNTRUSTED.clear()
    monkeypatch.setenv("CE_TRUST_PLUGIN", "tests.entry")
    registry._ENV_TRUST_CACHE = None

    plugin = _make_entry_plugin()
    fake_entries = FakeEntryPoints([FakeEntryPoint(plugin)])
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
    from calibrated_explanations.utils.exceptions import ValidationError

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
    from calibrated_explanations.utils.exceptions import ValidationError

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

    with pytest.raises(ValidationError):
        registry.register_plot_builder("bad.plot", BadPlotPlugin())


def test_register_plot_plugin_combines_builder_and_renderer():
    registry.clear_plot_plugins()

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
        if deprecations_error_enabled():
            with pytest.raises(DeprecationWarning):
                registry.register_plot_plugin("example.plot.combined", plugin)
        else:
            with warns_or_raises():
                descriptor = registry.register_plot_plugin("example.plot.combined", plugin)
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
        registry.clear_plot_plugins()


def test_find_plot_plugin_trusted_requires_trusted_components():
    registry.clear_plot_plugins()

    class Builder:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:builder"],
            "name": "example.plot.builder.untrusted",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": ["matplotlib"],
            "trusted": False,
            "trust": False,
            "output_formats": ["png"],
            "legacy_compatible": False,
            "style": "example.plot.mixed",
        }

        def build(self, value):
            return ("builder", value)

    class Renderer:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:renderer"],
            "name": "example.plot.renderer.untrusted",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": ["matplotlib"],
            "trusted": False,
            "trust": False,
            "output_formats": ["png"],
            "supports_interactive": False,
        }

        def render(self, value):
            return ("renderer", value)

    try:
        builder_desc = registry.register_plot_builder("example.plot.mixed", Builder())
        renderer_desc = registry.register_plot_renderer("example.plot.mixed", Renderer())
        registry.register_plot_style(
            "example.plot.mixed",
            metadata={
                "style": "example.plot.mixed",
                "builder_id": builder_desc.identifier,
                "renderer_id": renderer_desc.identifier,
                "fallbacks": (),
            },
        )

        combined = registry.find_plot_plugin("example.plot.mixed")
        assert combined is not None
        assert combined.build("data")[0] == "builder"
        assert combined.render("data")[0] == "renderer"

        assert registry.find_plot_plugin_trusted("example.plot.mixed") is None
    finally:
        registry.clear_plot_plugins()


def test_verify_plugin_checksum_success_and_failure(monkeypatch):
    from calibrated_explanations.utils.exceptions import ValidationError

    plugin = ExamplePlotBuilder()
    module_path = Path(__file__)
    good_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()

    meta = {"checksum": {"sha256": good_digest}, "name": "example.plot.builder"}
    # Should not raise when checksum matches
    registry._verify_plugin_checksum(plugin, dict(meta))

    meta_string = {"checksum": good_digest, "name": "example.plot.builder"}
    registry._verify_plugin_checksum(plugin, dict(meta_string))

    bad_meta = {"checksum": {"sha256": "00" * 32}, "name": "example.plot.builder"}
    with pytest.raises(ValidationError, match="Checksum mismatch"):
        registry._verify_plugin_checksum(plugin, bad_meta)

    # Simulate a plugin whose module file cannot be resolved
    meta_missing = {"checksum": {"sha256": good_digest}, "name": "missing"}
    monkeypatch.setattr(
        registry,
        "_resolve_plugin_module_file",
        lambda plugin: module_path.with_name("nonexistent_file"),
    )
    with pytest.warns(UserWarning):
        registry._verify_plugin_checksum(object(), meta_missing)

    with pytest.raises(ValidationError, match="must be a string or mapping"):
        registry._verify_plugin_checksum(plugin, {"checksum": 123, "name": "bad"})


def test_list_descriptors_and_trust_management(monkeypatch):
    registry.clear()
    registry.clear_explanation_plugins()
    registry.clear_interval_plugins()
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

        assert descriptor.trusted is True
        assert alt_descriptor.trusted is False

        interval_descriptor = registry.register_interval_plugin(
            "core.interval.trusted",
            TrustedIntervalPlugin(),
        )
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
        registry.clear_explanation_plugins()
        registry.clear_interval_plugins()


def test_list_plot_descriptors_respect_trust(monkeypatch):
    registry.clear_plot_plugins()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)

    class TrustedCombo:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:builder", "plot:renderer"],
            "name": "example.plot.trusted",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "trusted": True,
            "trust": True,
            "output_formats": ["png"],
            "legacy_compatible": True,
            "supports_interactive": False,
            "style": "example.plot.trusted",
        }

        def build(self):
            return "trusted-build"

        def render(self):
            return "trusted-render"

    class UntrustedBuilder:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:builder"],
            "name": "example.plot.untrusted.builder",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "trusted": False,
            "trust": False,
            "output_formats": ["png"],
            "legacy_compatible": False,
            "style": "example.plot.untrusted",
        }

        def build(self):
            return "untrusted-build"

    class UntrustedRenderer:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["plot:renderer"],
            "name": "example.plot.untrusted.renderer",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
            "trusted": False,
            "trust": False,
            "output_formats": ["png"],
            "supports_interactive": False,
        }

        def render(self):
            return "untrusted-render"

    try:
        trusted_plugin = TrustedCombo()
        if deprecations_error_enabled():
            with pytest.raises(DeprecationWarning):
                registry.register_plot_plugin("example.plot.trusted", trusted_plugin)
        else:
            with warns_or_raises():
                registry.register_plot_plugin("example.plot.trusted", trusted_plugin)

        builder_desc = registry.register_plot_builder("example.plot.untrusted", UntrustedBuilder())
        renderer_desc = registry.register_plot_renderer(
            "example.plot.untrusted", UntrustedRenderer()
        )
        registry.register_plot_style(
            "example.plot.untrusted",
            metadata={
                "style": "example.plot.untrusted",
                "builder_id": builder_desc.identifier,
                "renderer_id": renderer_desc.identifier,
                "fallbacks": ("example.plot.trusted",),
            },
        )

        if not deprecations_error_enabled():
            all_builders = registry.list_plot_builder_descriptors()
            assert [d.identifier for d in all_builders] == [
                "example.plot.trusted",
                "example.plot.untrusted",
            ]
        trusted_builders = registry.list_plot_builder_descriptors(trusted_only=True)
        if deprecations_error_enabled():
            # In raise-mode the deprecated registration raised and the trusted
            # plugin won't be present.
            assert [d.identifier for d in trusted_builders] == []
        else:
            assert [d.identifier for d in trusted_builders] == ["example.plot.trusted"]

        all_renderers = registry.list_plot_renderer_descriptors()
        if deprecations_error_enabled():
            assert [d.identifier for d in all_renderers] == ["example.plot.untrusted"]
        else:
            assert [d.identifier for d in all_renderers] == [
                "example.plot.trusted",
                "example.plot.untrusted",
            ]
        trusted_renderers = registry.list_plot_renderer_descriptors(trusted_only=True)
        if deprecations_error_enabled():
            assert [d.identifier for d in trusted_renderers] == []
        else:
            assert [d.identifier for d in trusted_renderers] == ["example.plot.trusted"]

        styles = registry.list_plot_style_descriptors()
        if deprecations_error_enabled():
            assert [d.identifier for d in styles] == ["example.plot.untrusted"]
        else:
            assert [d.identifier for d in styles] == [
                "example.plot.trusted",
                "example.plot.untrusted",
            ]
    finally:
        registry.clear_plot_plugins()


def test_ensure_builtin_plugins_invokes_register(monkeypatch):
    registry.clear()
    registry.clear_explanation_plugins()
    registry.clear_interval_plugins()
    registry.clear_plot_plugins()
    called = []

    monkeypatch.setattr(
        "calibrated_explanations.plugins.builtins._register_builtins",
        lambda: called.append(True),
    )

    try:
        registry.ensure_builtin_plugins()
        assert called == [True]
    finally:
        registry.clear()
        registry.clear_explanation_plugins()
        registry.clear_interval_plugins()
        registry.clear_plot_plugins()


def test_trust_normalisation_helpers(monkeypatch):
    registry._ENV_TRUST_CACHE = None
    registry._WARNED_UNTRUSTED.clear()
    monkeypatch.setenv("CE_TRUST_PLUGIN", "auto_plugin")

    meta_mapping = {"trust": {"default": True}}
    assert registry._normalise_trust(meta_mapping) is True

    names = registry._env_trusted_names()
    assert names == {"auto_plugin"}
    # Cache should persist even if env changes
    monkeypatch.setenv("CE_TRUST_PLUGIN", "")
    assert registry._env_trusted_names() == {"auto_plugin"}

    meta_env = {"name": "auto_plugin", "trust": False}
    assert registry._should_trust(meta_env) is True

    meta_untrusted = {"name": "manual", "provider": "tests"}
    with pytest.warns(UserWarning):
        registry._warn_untrusted_plugin(meta_untrusted, source="entry")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        registry._warn_untrusted_plugin(meta_untrusted, source="entry")
    assert caught == []


def test_load_entrypoint_plugins_error_branches(monkeypatch):
    registry.clear()
    registry._WARNED_UNTRUSTED.clear()
    registry._ENV_TRUST_CACHE = None

    class NoMetaPlugin:
        pass

    class InvalidPlugin:
        plugin_meta = {"name": "invalid"}

    class UntrustedPlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "entry.untrusted",
            "version": "0.0-test",
            "provider": "tests",
            "dependencies": [],
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
            self.group = registry._ENTRYPOINT_GROUP
            self._loader = loader

        def load(self):
            if self._loader is not None:
                return self._loader()
            module = importlib.import_module(self.module)
            return getattr(module, self.attr)

    failing = LoaderEntryPoint(
        "failing", loader=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    no_meta = LoaderEntryPoint("no_meta", loader=lambda: NoMetaPlugin())
    invalid = LoaderEntryPoint("invalid", loader=lambda: InvalidPlugin())
    untrusted = LoaderEntryPoint("untrusted", loader=lambda: UntrustedPlugin())
    trusted = LoaderEntryPoint("trusted", loader=lambda: TrustedPlugin())
    attr_entry = LoaderEntryPoint("attr", module=attr_module_name, attr="attr_plugin")

    entry_points = FakeEntryPoints([failing, no_meta, invalid, untrusted, trusted, attr_entry])
    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: entry_points)

    try:
        with pytest.warns(UserWarning):
            loaded = registry.load_entrypoint_plugins()
        assert {plugin.plugin_meta["name"] for plugin in loaded} == {"entry.trusted", "entry.attr"}
        # untrusted plugin should not be registered when include_untrusted is False
        assert all(plugin in registry.list_plugins(include_untrusted=False) for plugin in loaded)
    finally:
        registry.clear()
        registry._WARNED_UNTRUSTED.clear()
        registry._ENV_TRUST_CACHE = None
        sys.modules.pop(attr_module_name, None)
