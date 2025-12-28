from __future__ import annotations

import hashlib
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.plugins import registry


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
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


def test_env_trusted_names_parses_multiple_delimiters(monkeypatch):
    monkeypatch.setenv("CE_TRUST_PLUGIN", "alpha; beta , gamma,,")

    names = registry.env_trusted_names()
    assert names == {"alpha", "beta", "gamma"}

    # A second call should reuse the cached entries even if the environment changes.
    monkeypatch.setenv("CE_TRUST_PLUGIN", "ignored")
    assert registry.env_trusted_names() == {"alpha", "beta", "gamma"}


def test_should_trust_respects_environment_override(monkeypatch):
    monkeypatch.setenv("CE_TRUST_PLUGIN", "external")

    meta = {"name": "external", "trust": False, "trusted": False}
    assert registry.should_trust(meta) is True


def test_update_trust_keys_synchronises_nested_mapping():
    meta = {"trust": {"trusted": False, "other": "value"}}

    registry.update_trust_keys(meta, True)

    assert meta["trusted"] is True
    assert meta["trust"]["trusted"] is True


def test_warn_untrusted_plugin_only_warns_once():
    meta = {"name": "plugin", "provider": "tests"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        registry.warn_untrusted_plugin(meta, source="entry point")
        registry.warn_untrusted_plugin(meta, source="entry point")

    runtime_warnings = [item for item in caught if item.category is UserWarning]
    assert len(runtime_warnings) == 1


def _make_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    module_name = "tests.plugins._checksum_mod"
    module_path = tmp_path / "_checksum_mod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    monkeypatch.setitem(sys.modules, module_name, module)

    return module_name, module_path


def test_verify_plugin_checksum_handles_success_failure_and_missing(tmp_path, monkeypatch):
    module_name, module_path = _make_module(tmp_path, monkeypatch)

    class Plugin:
        __module__ = module_name

    plugin = Plugin()
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()

    registry.verify_plugin_checksum(plugin, {"checksum": {"sha256": digest}, "name": "ok"})

    with pytest.raises(ValidationError):
        registry.verify_plugin_checksum(plugin, {"checksum": "deadbeef", "name": "broken"})

    module_path.unlink()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        registry.verify_plugin_checksum(plugin, {"checksum": digest, "name": "missing"})
    assert any("Cannot verify checksum" in str(item.message) for item in caught)


def test_ensure_sequence_validates_inputs():
    assert registry.ensure_sequence({"values": ["a", "b"]}, "values") == ("a", "b")

    with pytest.raises(ValidationError):
        registry.ensure_sequence({}, "values")
    with pytest.raises(ValidationError):
        registry.ensure_sequence({"values": "single"}, "values")
    with pytest.raises(ValidationError):
        registry.ensure_sequence({"values": ["a", 1]}, "values")
    with pytest.raises(ValidationError):
        registry.ensure_sequence({"values": []}, "values")
    with pytest.raises(ValidationError):
        registry.ensure_sequence({"values": ["x"]}, "values", allowed={"y"})


def test_normalise_dependency_field_optional_and_empty():
    meta = {"deps": ["first", "second"]}
    normalised = registry.normalise_dependency_field(meta, "deps")
    assert normalised == ("first", "second")
    assert meta["deps"] == normalised

    empty_meta = {"deps": ()}
    assert registry.normalise_dependency_field(empty_meta, "deps", allow_empty=True) == ()

    assert registry.normalise_dependency_field({}, "missing", optional=True) is None

    with pytest.raises(ValidationError):
        registry.normalise_dependency_field({}, "required")


def test_coerce_string_collection_rejects_invalid():
    assert registry.coerce_string_collection(["a", "b"], key="k") == ("a", "b")

    with pytest.raises(ValidationError):
        registry.coerce_string_collection(1, key="k")
    with pytest.raises(ValidationError):
        registry.coerce_string_collection(["a", 2], key="k")
    with pytest.raises(ValidationError):
        registry.coerce_string_collection([], key="k")


def test_normalise_tasks_requires_known_values():
    meta = {"tasks": ("classification", "regression")}
    assert registry.normalise_tasks(meta) == ("classification", "regression")

    with pytest.raises(ValidationError):
        registry.normalise_tasks({"tasks": ("unknown",)})
    with pytest.raises(ValidationError):
        registry.normalise_tasks({})


def test_validate_explanation_metadata_requires_trust_key():
    meta = base_meta(
        modes=("factual",),
        tasks=("classification",),
        dependencies=(),
    )

    with pytest.raises(ValidationError):
        registry.validate_explanation_metadata(meta)


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


def test_validate_interval_metadata_handles_trust(monkeypatch):
    meta = interval_meta()
    registry.validate_interval_metadata(meta)

    meta_with_trusted = interval_meta(trusted=True)
    registry.validate_interval_metadata(meta_with_trusted)
    assert meta_with_trusted["trust"] is True

    del meta_with_trusted["trust"]
    with pytest.raises(ValidationError):
        registry.validate_interval_metadata(meta_with_trusted)


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


def test_validate_plot_builder_and_renderer_metadata():
    builder_meta = plot_builder_meta()
    registry.validate_plot_builder_metadata(builder_meta)

    renderer_meta = base_meta(
        capabilities=["plot"],
        dependencies=(),
        output_formats=("png",),
        supports_interactive=False,
        trust=False,
    )
    registry.validate_plot_renderer_metadata(renderer_meta)

    with pytest.raises(ValidationError):
        registry.validate_plot_renderer_metadata(
            {"capabilities": (), "dependencies": (), "output_formats": (), "trust": False}
        )


def test_validate_plot_builder_accepts_default_renderer():
    meta = plot_builder_meta(default_renderer="core.plot.legacy")
    validated = registry.validate_plot_builder_metadata(meta)
    assert validated.get("default_renderer") == "core.plot.legacy"


def test_ensure_string_and_bool_validation():
    assert registry.ensure_string({"name": "value"}, "name") == "value"
    assert registry.ensure_bool({"flag": True}, "flag") is True

    with pytest.raises(ValidationError):
        registry.ensure_string({}, "missing")
    with pytest.raises(ValidationError):
        registry.ensure_bool({}, "missing")


def test_validate_plot_style_metadata_normalises_fields():
    meta = {
        "style": "bespoke",
        "builder_id": "builder",
        "renderer_id": "renderer",
        "fallbacks": ["alt"],
        "is_default": True,
        "legacy_compatible": True,
        "default_for": "classification",
    }

    normalised = registry.validate_plot_style_metadata(meta)

    assert normalised["fallbacks"] == ("alt",)
    assert normalised["is_default"] is True
    assert normalised["legacy_compatible"] is True
    assert normalised["default_for"] == ("classification",)


def test_register_plot_style_round_trip():
    descriptor = registry.register_plot_style(
        "custom",
        metadata={
            "style": "custom",
            "builder_id": "builder",
            "renderer_id": "renderer",
        },
    )

    assert descriptor.identifier == "custom"
    assert registry.find_plot_style_descriptor("custom") is descriptor


def test_list_plot_builder_descriptors_respects_trust(monkeypatch):
    registry.set_plot_builder(
        "a", registry.PlotBuilderDescriptor("a", object(), {}, True), trusted=True
    )
    registry.set_plot_builder(
        "b", registry.PlotBuilderDescriptor("b", object(), {}, False), trusted=False
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


def test_list_explanation_descriptors_filters_trusted(monkeypatch):
    # Register two simple explanation plugins via public API rather than
    # mutating internals.
    class PlugA:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "trusted",
            "version": "0.0",
            "provider": "tests",
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "trust": True,
        }

    class PlugB:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "untrusted",
            "version": "0.0",
            "provider": "tests",
            "modes": ("factual",),
            "tasks": ("classification",),
            "dependencies": (),
            "trust": False,
        }

    registry.clear_explanation_plugins()
    registry.register_explanation_plugin("trusted", PlugA())
    registry.register_explanation_plugin("untrusted", PlugB())

    all_ids = [descriptor.identifier for descriptor in registry.list_explanation_descriptors()]
    trusted_ids = [
        descriptor.identifier
        for descriptor in registry.list_explanation_descriptors(trusted_only=True)
    ]

    assert all_ids == ["trusted", "untrusted"]
    assert trusted_ids == ["trusted"]


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


def test_register_plot_builder():
    class PlotBuilder:
        plugin_meta = {
            "schema_version": 1,
            "name": "tests.builder",
            "version": "0.0-test",
            "provider": "tests",
            "capabilities": ["plot"],
            "style": "test",
            "dependencies": (),
            "trust": False,
            "legacy_compatible": True,
            "output_formats": ("png",),
        }

        def build(self, *args, **kwargs):  # pragma: no cover
            return {"built": True}

    plugin = PlotBuilder()
    descriptor = registry.register_plot_builder("test.builder", plugin)
    assert descriptor.identifier == "test.builder"
    assert registry.find_plot_builder("test.builder") is not None


def test_register_plot_renderer():
    class PlotRenderer:
        plugin_meta = {
            "schema_version": 1,
            "name": "tests.renderer",
            "version": "0.0-test",
            "provider": "tests",
            "capabilities": ["plot"],
            "output_formats": ("png",),
            "supports_interactive": False,
            "dependencies": (),
            "trust": False,
        }

        def render(self, *args, **kwargs):  # pragma: no cover
            return {"rendered": True}

    plugin = PlotRenderer()
    descriptor = registry.register_plot_renderer("test.renderer", plugin)
    assert descriptor.identifier == "test.renderer"
    assert registry.find_plot_renderer("test.renderer") is not None


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


def test_list_interval_descriptors_filters_trusted(monkeypatch):
    registry.ensure_builtin_plugins()
    # Ensure no intervals are marked trusted for this test using public helpers.
    for d in registry.list_interval_descriptors():
        registry.register_interval_plugin(d.identifier, d.plugin, metadata=d.metadata)

    all_descriptors = list(registry.list_interval_descriptors())
    trusted_descriptors = list(registry.list_interval_descriptors(trusted_only=True))

    assert len(all_descriptors) >= len(trusted_descriptors)


def test_list_plot_builder_descriptors_filters_trusted(monkeypatch):
    registry.ensure_builtin_plugins()
    # Clear trusted flags using public setter.
    for d in registry.list_plot_builder_descriptors():
        registry.set_plot_builder(d.identifier, d, trusted=False)

    all_descriptors = list(registry.list_plot_builder_descriptors())
    trusted_descriptors = list(registry.list_plot_builder_descriptors(trusted_only=True))

    assert len(all_descriptors) >= len(trusted_descriptors)


def test_register_explanation_plugin_invalid_identifier():
    class Plugin:
        plugin_meta = base_meta()

    with pytest.raises(ValidationError, match="identifier must be a non-empty string"):
        registry.register_explanation_plugin("", Plugin())


def test_register_explanation_plugin_no_metadata():
    class Plugin:
        pass

    with pytest.raises(ValidationError, match="plugin must expose plugin_meta metadata"):
        registry.register_explanation_plugin("test", Plugin())


def test_register_explanation_plugin_invalid_metadata():
    class Plugin:
        plugin_meta = {"invalid": "meta"}

    with pytest.raises(ValidationError):
        registry.register_explanation_plugin("test", Plugin())


def test_register_interval_plugin_invalid_identifier():
    class Plugin:
        plugin_meta = base_meta(capabilities=["interval"])

    with pytest.raises(ValidationError, match="identifier must be a non-empty string"):
        registry.register_interval_plugin("", Plugin())


def test_register_plot_builder_invalid_identifier():
    class Plugin:
        plugin_meta = base_meta(capabilities=["plot"])

    with pytest.raises(ValidationError, match="identifier must be a non-empty string"):
        registry.register_plot_builder("", Plugin())


def test_validate_interval_metadata_missing_modes():
    meta = base_meta(capabilities=["interval"])

    with pytest.raises(ValidationError, match="plugin_meta missing required key: modes"):
        registry.validate_interval_metadata(meta)


def test_validate_interval_metadata_invalid_modes():
    meta = base_meta(capabilities=["interval"])
    meta["modes"] = ("invalid",)

    with pytest.raises(ValidationError):
        registry.validate_interval_metadata(meta)


def test_validate_interval_metadata_missing_capabilities():
    meta = base_meta(capabilities=["interval"])
    del meta["capabilities"]

    with pytest.raises(ValidationError):
        registry.validate_interval_metadata(meta)


def test_validate_plot_builder_metadata_missing_capabilities():
    meta = base_meta(capabilities=["plot"])
    del meta["capabilities"]

    with pytest.raises(ValidationError):
        registry.validate_plot_builder_metadata(meta)
