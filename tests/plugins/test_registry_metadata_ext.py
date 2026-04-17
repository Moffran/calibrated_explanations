from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from calibrated_explanations.plugins import registry
from tests.support.registry_helpers import (
    clear_explanation_plugins,
    clear_env_trust_cache,
    clear_interval_plugins,
    clear_plot_plugins,
    clear_trust_warnings,
    plot_builders,
    plot_renderers,
    plot_styles,
    set_plot_builder,
    update_trust_keys,
)


@pytest.fixture(autouse=True)
def isolate_registry_fixture(monkeypatch):
    # Use public clear helpers rather than patching internals.
    registry.clear()
    clear_explanation_plugins()
    clear_interval_plugins()
    clear_plot_plugins()
    clear_env_trust_cache()
    clear_trust_warnings()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None, raising=False)
    yield


def base_meta(**extra):
    meta = {
        "schema_version": 1,
        "name": "tests.meta",
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["explain"],
        "data_modalities": ("tabular",),
    }
    meta.update(extra)
    return meta


def test_update_trust_keys_synchronises_nested_mapping():
    meta = {"trust": {"trusted": False, "other": "value"}}

    update_trust_keys(meta, True)

    assert meta["trusted"] is True
    assert meta["trust"]["trusted"] is True


def make_module_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    module_name = "tests.plugins.checksum_mod"
    module_path = tmp_path / "checksum_mod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    monkeypatch.setitem(sys.modules, module_name, module)

    return module_name, module_path


def interval_meta(**extra):
    return base_meta(
        capabilities=["interval"],
        modes=("classification",),
        dependencies=(),
        fast_compatible=True,
        requires_bins=False,
        confidence_source="posterior",
        trust=False,
        **extra,
    )


def plot_builder_meta(**extra):
    return base_meta(
        capabilities=["plot"],
        style="fancy",
        dependencies=(),
        legacy_compatible=True,
        output_formats=("png",),
        trust=False,
        **extra,
    )


def test_validate_plot_builder_accepts_default_renderer():
    meta = plot_builder_meta(default_renderer="core.plot.legacy")
    validated = registry.validate_plot_builder_metadata(meta)
    assert validated.get("default_renderer") == "core.plot.legacy"


def test_list_plot_builder_descriptors_respects_trust(monkeypatch):
    set_plot_builder(
        "a", registry.PlotBuilderDescriptor("a", object(), {}, True, "manual"), trusted=True
    )
    set_plot_builder(
        "b", registry.PlotBuilderDescriptor("b", object(), {}, False, "manual"), trusted=False
    )

    all_ids = [descriptor.identifier for descriptor in registry.list_plot_builder_descriptors()]
    trusted_ids = [
        descriptor.identifier
        for descriptor in registry.list_plot_builder_descriptors(trusted_only=True)
    ]

    assert all_ids == ["a", "b"]
    assert trusted_ids == ["a"]


def test_register_plot_builder_renderer_and_style_register_all_components():
    class PlotPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "tests.plot",
            "version": "0.0-test",
            "provider": "tests",
            "capabilities": ["plot"],
            "style": "combo",
            "dependencies": (),
            "legacy_compatible": True,
            "output_formats": ("png",),
            "supports_interactive": False,
            "trust": False,
        }

        def build(self, *args, **kwargs):  # pragma: no cover - behaviour exercised via registry
            return {"built": True}

        def render(self, *args, **kwargs):  # pragma: no cover - behaviour exercised via registry
            return {"rendered": True}

    plugin = PlotPlugin()

    descriptor = registry.register_plot_builder("combo", plugin)
    registry.register_plot_renderer("combo", plugin)
    registry.register_plot_style(
        "combo",
        metadata={
            "style": "combo",
            "builder_id": "combo",
            "renderer_id": "combo",
            "fallbacks": (),
        },
    )

    assert descriptor.identifier == "combo"
    assert "combo" in plot_builders()
    assert "combo" in plot_renderers()
    assert "combo" in plot_styles()


def test_register_emits_governance_event_for_accepted_registration(caplog):
    class Plugin:
        plugin_meta = base_meta()

    with (
        caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"),
        pytest.warns(DeprecationWarning, match="register\\(\\) is deprecated"),
    ):
        registry.register(Plugin(), source="manual")

    matches = [
        record
        for record in caplog.records
        if getattr(record, "decision", None) == "accepted_registration"
    ]
    assert matches
    assert matches[-1].source == "manual"


def test_discover_entrypoint_emits_accepted_registration_event(monkeypatch, caplog):
    class Plugin:
        plugin_meta = base_meta(name="tests.entrypoint.accepted")

    class EntryPoint:
        name = "tests.entrypoint.accepted"

        def load(self):
            return Plugin()

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == "calibrated_explanations.plugins" else []

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: EntryPoints())
    registry.clear()
    clear_env_trust_cache()
    clear_trust_warnings()
    monkeypatch.setenv("CE_TRUST_PLUGIN", "tests.entrypoint.accepted")

    with caplog.at_level("INFO", logger="calibrated_explanations.governance.plugins"):
        loaded = registry.load_entrypoint_plugins(include_untrusted=False)

    assert len(loaded) == 1
    matches = [
        record
        for record in caplog.records
        if getattr(record, "decision", None) == "accepted_registration"
        and getattr(record, "source", None) == "entrypoint"
    ]
    assert matches
