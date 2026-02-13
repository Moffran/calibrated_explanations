from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.plugins import registry


@pytest.fixture(autouse=True)
def isolate_registry_fixture(monkeypatch):
    # Use public clear helpers rather than patching internals.
    registry.clear()
    registry.clear_explanation_plugins()
    registry.clear_interval_plugins()
    registry.clear_plot_plugins()
    registry.clear_env_trust_cache()
    registry.clear_trust_warnings()
    monkeypatch.setattr(registry, "ensure_builtin_plugins", lambda: None, raising=False)
    yield


def base_meta(**extra):
    meta = {
        "schema_version": 1,
        "name": "tests.meta",
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["explain"],
    }
    meta.update(extra)
    return meta




def test_update_trust_keys_synchronises_nested_mapping():
    meta = {"trust": {"trusted": False, "other": "value"}}

    registry.update_trust_keys(meta, True)

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
    registry.set_plot_builder(
        "a", registry.PlotBuilderDescriptor("a", object(), {}, True, "manual"), trusted=True
    )
    registry.set_plot_builder(
        "b", registry.PlotBuilderDescriptor("b", object(), {}, False, "manual"), trusted=False
    )

    all_ids = [descriptor.identifier for descriptor in registry.list_plot_builder_descriptors()]
    trusted_ids = [
        descriptor.identifier
        for descriptor in registry.list_plot_builder_descriptors(trusted_only=True)
    ]

    assert all_ids == ["a", "b"]
    assert trusted_ids == ["a"]


@pytest.mark.filterwarnings("ignore:register_plot_plugin is deprecated")
def test_register_plot_plugin_registers_all_components():
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

    with pytest.warns(DeprecationWarning, match="register_plot_plugin is deprecated"):
        descriptor = registry.register_plot_plugin("combo", plugin)

    assert descriptor.identifier == "combo"
    assert "combo" in registry.plot_builders()
    assert "combo" in registry.plot_renderers()
    assert "combo" in registry.plot_styles()




def test_register_interval_plugin():
    class IntervalPlugin:
        plugin_meta = {
            "schema_version": 1,
            "name": "tests.interval",
            "version": "0.0-test",
            "provider": "tests",
            "capabilities": ["interval"],
            "modes": ("classification",),
            "dependencies": (),
            "fast_compatible": True,
            "requires_bins": False,
            "confidence_source": "test",
            "trust": False,
        }

        def calibrate(self, *args, **kwargs):  # pragma: no cover
            return {"calibrated": True}

    plugin = IntervalPlugin()
    descriptor = registry.register_interval_plugin("test.interval", plugin)
    assert descriptor.identifier == "test.interval"
    assert registry.find_interval_plugin("test.interval") is not None




def test_register_plot_style():
    style_meta = {
        "schema_version": 1,
        "name": "tests.style",
        "version": "0.0-test",
        "provider": "tests",
        "capabilities": ["plot"],
        "dependencies": (),
        "trust": False,
        "style": "test_style",
        "builder_id": "test.builder",
        "renderer_id": "test.renderer",
    }
    descriptor = registry.register_plot_style("test.style", metadata=style_meta)
    assert descriptor.identifier == "test.style"
    assert "test.style" in registry.plot_styles()




def test_list_plot_builder_descriptors_filters_trusted(monkeypatch):
    registry.ensure_builtin_plugins()
    # Clear trusted flags using public setter.
    for d in registry.list_plot_builder_descriptors():
        registry.set_plot_builder(d.identifier, d, trusted=False)

    all_descriptors = list(registry.list_plot_builder_descriptors())
    trusted_descriptors = list(registry.list_plot_builder_descriptors(trusted_only=True))

    assert len(all_descriptors) >= len(trusted_descriptors)




def test_register_explanation_plugin_no_metadata():
    class Plugin:
        pass

    with pytest.raises(ValidationError, match="plugin must expose plugin_meta metadata"):
        registry.register_explanation_plugin("test_no_meta", Plugin())


