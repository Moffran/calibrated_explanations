"""Tests for legacy list-path plugin API deprecations."""

from __future__ import annotations

import pytest

from calibrated_explanations.plugins import registry
from tests.unit.utils.test_deprecations_helper import reset_deprecation_state  # noqa: F401


class _LegacyPlugin:
    plugin_meta = {
        "schema_version": 1,
        "name": "tests.legacy.plugin",
        "version": "1.0.0",
        "provider": "tests",
        "capabilities": ("factual",),
        "trusted": False,
    }

    def supports(self, model):
        return model == "supported"

    def explain(self, model, x, **kwargs):
        return {"model": model, "x": x, "kwargs": kwargs}


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    """Run each test with an isolated legacy/plugin registry state."""
    registry.clear()
    registry.reset_plugin_catalog(kind="all")
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)
    yield
    registry.clear()
    registry.reset_plugin_catalog(kind="all")


def test_should_warn_when_register_list_path_is_used():
    plugin = _LegacyPlugin()

    with pytest.warns(DeprecationWarning, match="v0.11.3"):
        registry.register(plugin)


def test_should_warn_when_trust_plugin_list_path_is_used():
    plugin = _LegacyPlugin()
    registry.register_explanation_plugin(plugin.plugin_meta["name"], plugin, metadata=plugin.plugin_meta)

    with pytest.warns(DeprecationWarning, match=r"metadata=\{'trusted': True\}"):
        registry.trust_plugin(plugin)


def test_should_warn_when_find_for_list_path_is_used():
    plugin = _LegacyPlugin()
    registry.register_explanation_plugin(plugin.plugin_meta["name"], plugin, metadata=plugin.plugin_meta)

    with pytest.warns(DeprecationWarning, match="trusted_only=False"):
        found = registry.find_for("supported")

    assert plugin in found


def test_should_warn_when_find_for_trusted_list_path_is_used():
    plugin = _LegacyPlugin()
    trusted_meta = dict(plugin.plugin_meta)
    trusted_meta["trusted"] = True
    registry.register_explanation_plugin(plugin.plugin_meta["name"], plugin, metadata=trusted_meta)

    with pytest.warns(DeprecationWarning, match="trusted_only=True"):
        found = registry.find_for_trusted("supported")

    assert plugin in found


def test_should_raise_when_ce_deprecations_error_is_set(monkeypatch):
    plugin = _LegacyPlugin()
    trusted_meta = dict(plugin.plugin_meta)
    trusted_meta["trusted"] = True
    registry.register_explanation_plugin(plugin.plugin_meta["name"], plugin, metadata=trusted_meta)

    monkeypatch.setenv("CE_DEPRECATIONS", "error")

    with pytest.raises(DeprecationWarning, match="register\\(\\) is deprecated"):
        registry.register(plugin)

    with pytest.raises(DeprecationWarning, match="trust_plugin\\(\\) is deprecated"):
        registry.trust_plugin(plugin)

    with pytest.raises(DeprecationWarning, match="find_for\\(\\) is deprecated"):
        registry.find_for("supported")

    with pytest.raises(DeprecationWarning, match="find_for_trusted\\(\\) is deprecated"):
        registry.find_for_trusted("supported")
