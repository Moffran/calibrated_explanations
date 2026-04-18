from __future__ import annotations

from collections import UserDict


import pytest

from calibrated_explanations.plugins import registry
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError
from tests.support.registry_helpers import clear_explanation_plugins
from tests.support.registry_helpers import (
    clear_interval_plugins,
    clear_plot_builders,
    clear_plot_renderers,
    clear_plot_styles,
    set_plot_builder,
    set_plot_renderer,
    set_plot_style,
)


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    """Ensure each test works with a clean registry state."""

    clear_explanation_plugins()
    clear_interval_plugins()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)
    # Also clear plot renderers for the new tests using public helpers
    clear_plot_builders()
    clear_plot_renderers()
    clear_plot_styles()
    yield
    clear_explanation_plugins()
    clear_interval_plugins()
    clear_plot_builders()
    clear_plot_renderers()
    clear_plot_styles()


def make_metadata(name: str, trusted: bool) -> dict[str, object]:
    capabilities = (
        "explain",
        "explanation:factual",
        "task:regression",
    )
    return {
        "schema_version": 1,
        "name": name,
        "version": "0.1",
        "provider": "tests",
        "capabilities": capabilities,
        "modes": ("factual",),
        "tasks": ("regression",),
        "trust": {"trusted": trusted},
        "dependencies": (),
    }


def test_mark_plot_renderer_trusted_untrusted():
    # Register a dummy renderer
    descriptor = registry.PlotRendererDescriptor(
        identifier="test.renderer",
        metadata={"name": "Test Renderer", "trust": {"trusted": False}},
        trusted=False,
        renderer=lambda: None,  # Add dummy renderer
        source="manual",
    )

    # Mock the internal registry lists
    set_plot_renderer("test.renderer", descriptor, trusted=False)

    # Mark trusted
    updated_descriptor = registry.mark_plot_renderer_trusted("test.renderer")
    assert updated_descriptor.trusted is True
    assert updated_descriptor.metadata["trust"]["trusted"] is True
    assert registry.find_plot_renderer_trusted("test.renderer") is not None

    # Mark untrusted
    updated_descriptor_2 = registry.mark_plot_renderer_untrusted("test.renderer")
    assert updated_descriptor_2.trusted is False
    assert updated_descriptor_2.metadata["trust"]["trusted"] is False
    trusted_ids = [d.identifier for d in registry.list_plot_renderer_descriptors(trusted_only=True)]
    assert "test.renderer" not in trusted_ids


def test_mark_plot_renderer_trusted_missing():
    with pytest.raises(KeyError, match="Plot renderer 'missing' is not registered"):
        registry.mark_plot_renderer_trusted("missing")


def test_should_verify_trust_invariants_when_debug_checks_enabled(monkeypatch):
    monkeypatch.setattr(registry, "trust_debug_checks_enabled", lambda: True)

    descriptor = registry.PlotRendererDescriptor(
        identifier="test.renderer.debug",
        metadata={"name": "Test Renderer Debug", "trust": {"trusted": False}},
        trusted=False,
        renderer=lambda: None,
        source="manual",
    )
    set_plot_renderer("test.renderer.debug", descriptor, trusted=False)

    updated = registry.mark_plot_renderer_trusted("test.renderer.debug")
    assert updated.trusted is True


def test_should_raise_configuration_error_when_trust_state_is_mismatched(monkeypatch):
    monkeypatch.setattr(registry, "trust_debug_checks_enabled", lambda: True)

    descriptor = registry.PlotRendererDescriptor(
        identifier="test.renderer.mismatch",
        metadata={"name": "Test Renderer Mismatch", "trust": {"trusted": False}},
        trusted=False,
        renderer=lambda: None,
        source="manual",
    )

    with pytest.raises(ConfigurationError, match="trust mismatch"):
        set_plot_renderer("test.renderer.mismatch", descriptor, trusted=True)

    # Disable invariant checks and clear mutated test state to keep teardown stable.
    monkeypatch.setattr(registry, "trust_debug_checks_enabled", lambda: False)
    clear_plot_renderers()


def test_should_synchronise_all_descriptor_kinds_when_legacy_trust_changes():
    class SharedPlugin:
        def __init__(self):
            self.plugin_meta = {
                "schema_version": 1,
                "name": "tests.registry.runtime.shared",
                "version": "0.1",
                "provider": "tests",
                "capabilities": ("explain", "task:classification"),
                "modes": ("factual",),
                "tasks": ("classification",),
                "trust": {"trusted": False},
                "dependencies": (),
            }

        def supports(self, _model):
            return True

        def explain(self, *_args, **_kwargs):
            return {}

        def calibrate(self, *_args, **_kwargs):
            return {}

    plugin = SharedPlugin()
    registry.register_explanation_plugin("tests.registry.runtime.shared.expl", plugin)
    registry.register_interval_plugin(
        "tests.registry.runtime.shared.interval",
        plugin,
        metadata={
            "schema_version": 1,
            "name": "tests.registry.runtime.shared.interval",
            "version": "0.1",
            "provider": "tests",
            "capabilities": ("interval",),
            "modes": ("classification",),
            "fast_compatible": True,
            "requires_bins": False,
            "confidence_source": "posterior",
            "trust": {"trusted": False},
            "dependencies": (),
        },
    )

    set_plot_builder(
        "tests.registry.runtime.shared.builder",
        registry.PlotBuilderDescriptor(
            identifier="tests.registry.runtime.shared.builder",
            builder=plugin,
            metadata={"name": "shared.builder", "trust": {"trusted": False}},
            trusted=False,
            source="manual",
        ),
        trusted=False,
    )
    set_plot_renderer(
        "tests.registry.runtime.shared.renderer",
        registry.PlotRendererDescriptor(
            identifier="tests.registry.runtime.shared.renderer",
            renderer=plugin,
            metadata={"name": "shared.renderer", "trust": {"trusted": False}},
            trusted=False,
            source="manual",
        ),
        trusted=False,
    )

    with pytest.warns(DeprecationWarning, match=r"register\(\) is deprecated"):
        registry.register(plugin)
    with pytest.warns(DeprecationWarning, match=r"trust_plugin\(\) is deprecated"):
        registry.trust_plugin(plugin)
    registry.untrust_plugin(plugin)

    assert registry.find_interval_descriptor("tests.registry.runtime.shared.interval").trusted is False
    assert (
        registry.find_plot_builder_descriptor("tests.registry.runtime.shared.builder").trusted
        is False
    )
    assert (
        registry.find_plot_renderer_descriptor("tests.registry.runtime.shared.renderer").trusted
        is False
    )


def test_should_return_defaults_for_discovery_report_and_support_record_pickling():
    report = registry.get_discovery_report()
    assert report.accepted == []
    assert report.skipped_untrusted == []

    record = registry.PluginDiscoveryRecord(
        identifier="tests.registry.runtime.discovery",
        provider="tests",
        source="entrypoint",
    )
    state = record.__getstate__()
    assert state["identifier"] == "tests.registry.runtime.discovery"


def test_should_return_plot_style_list_and_renderer_none_when_untrusted():
    set_plot_style(
        "tests.registry.runtime.style",
        registry.PlotStyleDescriptor(
            identifier="tests.registry.runtime.style",
            metadata={"builder_id": "b", "renderer_id": "r"},
        ),
    )
    styles = registry.list_plot_descriptors()
    assert len(styles) == 1

    descriptor = registry.PlotRendererDescriptor(
        identifier="tests.registry.runtime.renderer.untrusted",
        metadata={"name": "renderer.untrusted", "trust": {"trusted": False}},
        trusted=False,
        renderer=lambda: None,
        source="manual",
    )
    set_plot_renderer("tests.registry.runtime.renderer.untrusted", descriptor, trusted=False)
    assert registry.find_plot_renderer_trusted("tests.registry.runtime.renderer.untrusted") is None


def test_should_raise_validation_errors_for_invalid_metadata_shapes():
    with pytest.raises(ValidationError, match=r"plugin_meta\['modes'\] must not be empty"):
        registry.validate_explanation_metadata(
            {
                "schema_version": 1,
                "name": "invalid.expl",
                "version": "0.1",
                "provider": "tests",
                "capabilities": ("explain",),
                "modes": (),
                "tasks": ("classification",),
                "trust": {"trusted": False},
                "dependencies": (),
            }
        )

    with pytest.raises(ValidationError, match="must be a non-empty string"):
        registry.validate_interval_metadata(
            {
                "schema_version": 1,
                "name": "invalid.interval",
                "version": "0.1",
                "provider": "tests",
                "capabilities": ("interval",),
                "modes": ("classification",),
                "fast_compatible": True,
                "requires_bins": False,
                "confidence_source": "",
                "trust": {"trusted": False},
                "dependencies": (),
            }
        )


def test_should_untrust_legacy_plugin_when_metadata_is_setitem_mapping():
    class MetaMapping(UserDict):
        pass

    class PluginWithMutableMappingMeta:
        def __init__(self):
            self.plugin_meta = MetaMapping(
                {
                    "schema_version": 1,
                    "name": "tests.registry.runtime.mutable-meta",
                    "version": "0.1",
                    "provider": "tests",
                    "capabilities": ("explain", "task:classification"),
                    "modes": ("factual",),
                    "tasks": ("classification",),
                    "trust": {"trusted": True},
                    "dependencies": (),
                }
            )

        def supports(self, _model):
            return True

        def explain(self, *_args, **_kwargs):
            return {}

    plugin = PluginWithMutableMappingMeta()
    with pytest.warns(DeprecationWarning, match=r"register\(\) is deprecated"):
        registry.register(plugin)
    with pytest.warns(DeprecationWarning, match=r"trust_plugin\(\) is deprecated"):
        registry.trust_plugin(plugin)
    registry.untrust_plugin(plugin)

    assert plugin.plugin_meta["trusted"] is False
