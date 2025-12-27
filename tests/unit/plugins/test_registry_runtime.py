from __future__ import annotations

import sys
import types

import pytest

from calibrated_explanations.plugins import registry


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    """Ensure each test works with a clean registry state."""

    registry.clear_explanation_plugins()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)
    # Also clear plot renderers for the new tests using public helpers
    registry.clear_plot_renderers()
    yield
    registry.clear_explanation_plugins()
    registry.clear_plot_renderers()


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


def test_list_explanation_descriptors_trusted_only_filters():
    trusted_plugin = types.SimpleNamespace(plugin_meta=make_metadata("trusted", True))
    untrusted_plugin = types.SimpleNamespace(plugin_meta=make_metadata("untrusted", False))

    registry.register_explanation_plugin("trusted", trusted_plugin)
    registry.register_explanation_plugin("untrusted", untrusted_plugin)

    all_descriptors = registry.list_explanation_descriptors()
    assert {d.identifier for d in all_descriptors} == {"trusted", "untrusted"}

    trusted_descriptors = registry.list_explanation_descriptors(trusted_only=True)
    assert [d.identifier for d in trusted_descriptors] == ["trusted"]


def test_verify_plugin_checksum_raises_on_mismatch(tmp_path, monkeypatch):
    plugin_file = tmp_path / "fake_plugin.py"
    plugin_file.write_text("VALUE = 'original'\n")

    module = types.ModuleType("tests.fake_plugin")
    module.__file__ = str(plugin_file)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    from calibrated_explanations.utils.exceptions import ValidationError

    class Plugin:
        __module__ = module.__name__
        plugin_meta = {
            "schema_version": 1,
            "name": "checksum",
            "version": "0.0",
            "provider": "tests",
            "capabilities": ("interval:regression",),
            "checksum": {"sha256": "deadbeef"},
        }

    with pytest.raises(ValidationError, match="Checksum mismatch"):
        registry.verify_plugin_checksum(Plugin(), Plugin.plugin_meta)


def test_mark_plot_renderer_trusted_untrusted():
    # Register a dummy renderer
    descriptor = types.SimpleNamespace(
        identifier="test.renderer",
        metadata={"name": "Test Renderer", "trust": {"trusted": False}},
        trusted=False,
        renderer=lambda: None,  # Add dummy renderer
    )

    # Mock the internal registry lists
    registry.set_plot_renderer("test.renderer", descriptor, trusted=False)

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
