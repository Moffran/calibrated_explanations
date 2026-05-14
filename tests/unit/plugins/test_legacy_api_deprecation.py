"""Tests for removed legacy list-path plugin APIs."""

from __future__ import annotations

from calibrated_explanations.plugins import registry


class LegacyPlugin:
    plugin_meta = {
        "schema_version": 1,
        "name": "tests.legacy.plugin",
        "version": "1.0.0",
        "provider": "tests",
        "capabilities": ("factual",),
        "modes": ("factual",),
        "tasks": ("classification",),
        "dependencies": (),
        "trusted": False,
    }

    def supports(self, model):
        return model == "supported"

    def explain(self, model, x, **kwargs):
        return {"model": model, "x": x, "kwargs": kwargs}


def test_should_fail_closed_for_removed_list_path_registry_functions() -> None:
    for symbol in ("register", "trust_plugin", "find_for", "find_for_trusted"):
        assert not hasattr(registry, symbol)
        assert symbol not in registry.__all__


def test_should_use_identifier_based_registry_replacement(monkeypatch) -> None:
    plugin = LegacyPlugin()
    trusted_meta = dict(plugin.plugin_meta)
    trusted_meta["trusted"] = True
    trusted_meta["data_modalities"] = ("tabular",)

    registry.clear()
    registry.reset_plugin_catalog(kind="all")
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None)
    try:
        registry.register_explanation_plugin(
            plugin.plugin_meta["name"], plugin, metadata=trusted_meta
        )
        registry.mark_explanation_trusted(plugin.plugin_meta["name"])

        identifier, found = registry.find_explanation_plugin_for(
            "tabular",
            mode="factual",
            task="classification",
            model="supported",
            trusted_only=True,
        )
        assert identifier == plugin.plugin_meta["name"]
        assert found is plugin
    finally:
        registry.clear()
        registry.reset_plugin_catalog(kind="all")
