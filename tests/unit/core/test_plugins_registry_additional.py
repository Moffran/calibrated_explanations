import hashlib
from collections.abc import MutableMapping
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

from calibrated_explanations.plugins import registry


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    registry.clear_explanation_plugins()
    registry.clear_interval_plugins()
    registry.clear_plot_plugins()
    registry._ENV_TRUST_CACHE = None
    registry._WARNED_UNTRUSTED.clear()
    yield
    registry.clear()
    registry.clear_explanation_plugins()
    registry.clear_interval_plugins()
    registry.clear_plot_plugins()
    registry._ENV_TRUST_CACHE = None
    registry._WARNED_UNTRUSTED.clear()


def test_normalise_trust_and_env_cache(monkeypatch):
    assert registry._normalise_trust({"trust": {"trusted": "yes"}})
    assert registry._normalise_trust({"trust": {"default": "1"}})

    monkeypatch.setenv("CE_TRUST_PLUGIN", "alpha;beta")
    registry._ENV_TRUST_CACHE = None
    first = registry._env_trusted_names()
    monkeypatch.setenv("CE_TRUST_PLUGIN", "")
    second = registry._env_trusted_names()
    assert first == {"alpha", "beta"}
    assert second == first

    meta = {"trust": False, "name": "alpha"}
    assert registry._should_trust(meta)


def test_propagate_trust_metadata_variants():
    registry._propagate_trust_metadata(object(), {"trusted": True, "trust": True})

    plugin_dict = SimpleNamespace(plugin_meta={})
    registry._propagate_trust_metadata(plugin_dict, {"trusted": True, "trust": True})
    assert plugin_dict.plugin_meta["trusted"] is True
    assert plugin_dict.plugin_meta["trust"] is True

    plugin_no_setter = SimpleNamespace(plugin_meta=SimpleNamespace())
    registry._propagate_trust_metadata(plugin_no_setter, {"trusted": False, "trust": False})

    class CustomMeta(MutableMapping):
        def __init__(self):
            self._data = {}

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def __delitem__(self, key):  # pragma: no cover - protocol requirement
            del self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    custom_meta = CustomMeta()
    plugin_custom = SimpleNamespace(plugin_meta=custom_meta)
    registry._propagate_trust_metadata(plugin_custom, {"trusted": False, "trust": False})
    assert custom_meta["trusted"] is False
    assert custom_meta["trust"] is False

    class ExplodingMeta(MutableMapping):
        def __init__(self):
            self._data = {}

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            raise RuntimeError("boom")

        def __delitem__(self, key):  # pragma: no cover - protocol requirement
            del self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    exploding_meta = ExplodingMeta()
    plugin_exploding = SimpleNamespace(plugin_meta=exploding_meta)
    registry._propagate_trust_metadata(plugin_exploding, {"trusted": True, "trust": True})


def test_resolve_plugin_module_file_branches(tmp_path):
    plugin_no_module = SimpleNamespace(__module__=None)
    assert registry._resolve_plugin_module_file(plugin_no_module) is None

    plugin_missing_module = SimpleNamespace(__module__="tests.missing.module")
    assert registry._resolve_plugin_module_file(plugin_missing_module) is None

    module = ModuleType("tests.fake_no_file")
    module.__file__ = None
    sys_modules = registry.sys.modules
    sys_modules[module.__name__] = module
    try:
        plugin_no_file = SimpleNamespace(__module__=module.__name__)
        assert registry._resolve_plugin_module_file(plugin_no_file) is None
    finally:
        sys_modules.pop(module.__name__, None)

    real_module = ModuleType("tests.fake_with_file")
    module_path = tmp_path / "module.py"
    module_path.write_text("pass")
    real_module.__file__ = str(module_path)
    sys_modules[real_module.__name__] = real_module
    try:
        resolved = registry._resolve_plugin_module_file(real_module)
        assert resolved == module_path
    finally:
        sys_modules.pop(real_module.__name__, None)


def test_verify_plugin_checksum_warnings_and_errors(monkeypatch, tmp_path):
    plugin = SimpleNamespace()

    with pytest.raises(ValueError):
        registry._verify_plugin_checksum(plugin, {"checksum": {"sha256": 123}})

    missing_path = tmp_path / "missing.py"
    monkeypatch.setattr(registry, "_resolve_plugin_module_file", lambda p: missing_path)
    with pytest.warns(RuntimeWarning):
        registry._verify_plugin_checksum(plugin, {"checksum": "deadbeef"})

    class FailingPath:
        def exists(self):
            return True

        def read_bytes(self):
            raise OSError("boom")

    monkeypatch.setattr(registry, "_resolve_plugin_module_file", lambda p: FailingPath())
    with pytest.warns(RuntimeWarning):
        registry._verify_plugin_checksum(plugin, {"checksum": "deadbeef"})

    module_file = tmp_path / "module.py"
    module_file.write_text("payload")
    digest = hashlib.sha256(module_file.read_bytes()).hexdigest()
    monkeypatch.setattr(registry, "_resolve_plugin_module_file", lambda p: module_file)
    registry._verify_plugin_checksum(plugin, {"checksum": {"sha256": digest}})

    monkeypatch.setattr(registry, "_resolve_plugin_module_file", lambda p: module_file)
    with pytest.raises(ValueError):
        registry._verify_plugin_checksum(plugin, {"checksum": "not-the-digest"})


def test_validate_explanation_metadata_from_mapping():
    source = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "mapping",
        "version": "0",
        "provider": "tests",
        "modes": ("factual",),
        "tasks": ("classification",),
        "dependencies": (),
        "trusted": False,
    }
    mapping = MappingProxyType(source)
    result = registry.validate_explanation_metadata(mapping)
    assert result["modes"] == ("factual",)
    assert result["trust"] is False


def test_validate_interval_metadata_requires_trust():
    meta = {
        "modes": ("classification",),
        "capabilities": ["interval:classification"],
        "dependencies": [],
        "fast_compatible": True,
        "requires_bins": False,
        "confidence_source": "legacy",
    }
    with pytest.raises(ValueError):
        registry.validate_interval_metadata(meta)


def test_sequence_and_collection_validation_errors():
    with pytest.raises(ValueError):
        registry._ensure_sequence({"values": "single"}, "values")

    with pytest.raises(ValueError):
        registry._ensure_sequence({"values": [1]}, "values")

    with pytest.raises(ValueError):
        registry._ensure_sequence({"values": []}, "values")

    with pytest.raises(ValueError):
        registry._ensure_sequence({"values": ["foo"]}, "values", allowed={"bar"})

    with pytest.raises(ValueError):
        registry._coerce_string_collection([1], key="items")

    with pytest.raises(ValueError):
        registry._coerce_string_collection(42, key="items")

    with pytest.raises(ValueError):
        registry._coerce_string_collection((), key="items")

    with pytest.raises(ValueError):
        registry._normalise_dependency_field({}, "missing")

    with pytest.raises(ValueError):
        registry._normalise_tasks({"tasks": ["invalid"]})

    bad_meta = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "bad",
        "version": "0",
        "provider": "tests",
        "modes": [],
        "tasks": ["classification"],
        "dependencies": [],
    }
    with pytest.raises(ValueError):
        registry.validate_explanation_metadata(bad_meta)

    missing_trust = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "bad2",
        "version": "0",
        "provider": "tests",
        "modes": ["factual"],
        "tasks": ["classification"],
        "dependencies": [],
    }
    with pytest.raises(ValueError):
        registry.validate_explanation_metadata(missing_trust)

    with pytest.raises(ValueError):
        registry._ensure_bool({}, "flag")

    with pytest.raises(ValueError):
        registry._ensure_bool({"flag": "yes"}, "flag")

    with pytest.raises(ValueError):
        registry._ensure_string({}, "name")

    with pytest.raises(ValueError):
        registry._ensure_string({"name": ""}, "name")

    interval_meta = {
        "modes": [],
        "capabilities": ["interval:classification"],
        "dependencies": [],
        "fast_compatible": True,
        "requires_bins": False,
        "confidence_source": "legacy",
        "trust": False,
    }
    with pytest.raises(ValueError):
        registry.validate_interval_metadata(interval_meta)

    proxy_meta = MappingProxyType(
        {
            "modes": ["classification"],
            "capabilities": ["interval:classification"],
            "dependencies": [],
            "fast_compatible": False,
            "requires_bins": False,
            "confidence_source": "legacy",
            "trust": False,
        }
    )
    result = registry.validate_interval_metadata(proxy_meta)
    assert isinstance(result, dict)

    proxy_builder = MappingProxyType(
        {
            "style": "s",
            "capabilities": ["plot:builder"],
            "dependencies": [],
            "legacy_compatible": True,
            "output_formats": ["png"],
            "trust": False,
        }
    )
    assert isinstance(registry.validate_plot_builder_metadata(proxy_builder), dict)

    proxy_renderer = MappingProxyType(
        {
            "capabilities": ["plot:renderer"],
            "dependencies": [],
            "output_formats": ["png"],
            "supports_interactive": False,
            "trust": False,
        }
    )
    assert isinstance(registry.validate_plot_renderer_metadata(proxy_renderer), dict)

    with pytest.raises(ValueError):
        registry.validate_plot_style_metadata(
            {
                "style": "s",
                "builder_id": "b",
                "renderer_id": "r",
                "fallbacks": ["ok", 1],
            }
        )

    with pytest.raises(ValueError):
        registry.validate_plot_style_metadata(
            {
                "style": "s",
                "builder_id": "b",
                "renderer_id": "r",
                "fallbacks": object(),
            }
        )


class _SimpleExplanationPlugin:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["explain"],
        "name": "simple.explanation",
        "version": "0",
        "provider": "tests",
        "modes": ["factual"],
        "tasks": ["classification"],
        "dependencies": [],
        "trust": False,
    }

    def supports(self, model):  # pragma: no cover - not needed
        return True

    def explain(self, model, x, **kwargs):  # pragma: no cover - not needed
        return {}


def test_register_explanation_plugin_updates_raw_meta():
    registry.clear_explanation_plugins()
    plugin = _SimpleExplanationPlugin()
    descriptor = registry.register_explanation_plugin("simple", plugin)
    assert descriptor.metadata["trusted"] is False
    assert plugin.plugin_meta["trusted"] is False
    assert plugin.plugin_meta["trust"] is False


def test_register_interval_plugin_requires_metadata():
    registry.clear_interval_plugins()

    class MissingMetaInterval:
        plugin_meta = None

    with pytest.raises(ValueError):
        registry.register_interval_plugin("missing", MissingMetaInterval())


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
        "trust": False,
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "legacy",
    }


def test_register_plot_builder_renderer_require_metadata():
    registry.clear_plot_plugins()

    class NoMetaBuilder:
        plugin_meta = None

    class NoMetaRenderer:
        plugin_meta = None

    with pytest.raises(ValueError):
        registry.register_plot_builder("builder", NoMetaBuilder())

    with pytest.raises(ValueError):
        registry.register_plot_renderer("renderer", NoMetaRenderer())


def test_register_helpers_validate_identifiers():
    registry.clear_explanation_plugins()
    with pytest.raises(ValueError):
        registry.register_explanation_plugin("", _SimpleExplanationPlugin())

    class NoMetaPlugin:
        plugin_meta = None

    with pytest.raises(ValueError):
        registry.register_explanation_plugin("ok", NoMetaPlugin())

    registry.clear_interval_plugins()
    with pytest.raises(ValueError):
        registry.register_interval_plugin("", ExampleIntervalPlugin())

    registry.clear_plot_plugins()
    with pytest.raises(ValueError):
        registry.register_plot_builder("", _Builder())

    with pytest.raises(ValueError):
        registry.register_plot_renderer("", _Renderer())

    with pytest.raises(ValueError):
        registry.register_plot_style(
            "", metadata={"style": "", "builder_id": "b", "renderer_id": "r"}
        )


def test_register_plot_style_validation():
    registry.clear_plot_plugins()
    with pytest.raises(ValueError):
        registry.register_plot_style("style", metadata=None)  # type: ignore[arg-type]

    meta = {
        "style": "other",
        "builder_id": "builder.id",
        "renderer_id": "renderer.id",
        "fallbacks": "legacy",
        "is_default": True,
        "legacy_compatible": True,
        "default_for": "ce",
    }
    descriptor = registry.register_plot_style("style", metadata=meta)
    assert descriptor.metadata["style"] == "other"
    assert descriptor.metadata["fallbacks"] == "legacy"
    assert descriptor.metadata["default_for"] == "ce"


class _Builder:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["plot:builder"],
        "name": "builder",
        "version": "0",
        "provider": "tests",
        "style": "style",
        "dependencies": [],
        "legacy_compatible": False,
        "output_formats": ["png"],
        "trust": False,
    }

    def build(self, *args, **kwargs):  # pragma: no cover - not exercised directly
        return {}


class _Renderer:
    plugin_meta = {
        "schema_version": 1,
        "capabilities": ["plot:renderer"],
        "name": "renderer",
        "version": "0",
        "provider": "tests",
        "dependencies": [],
        "output_formats": ["png"],
        "supports_interactive": False,
        "trust": False,
    }

    def render(self, *args, **kwargs):  # pragma: no cover - not exercised directly
        return {}


def test_find_plot_plugin_variants():
    assert registry.find_plot_plugin("missing") is None

    empty_descriptor = registry.PlotStyleDescriptor("nosteps", {"style": "nosteps"})
    registry._PLOT_STYLES["nosteps"] = empty_descriptor
    assert registry.find_plot_plugin("nosteps") is None

    registry.clear_plot_plugins()
    registry._PLOT_STYLES["broken"] = registry.PlotStyleDescriptor(
        "broken", {"style": "broken", "builder_id": "", "renderer_id": ""}
    )
    assert registry.find_plot_plugin_trusted("broken") is None
    registry.clear_plot_plugins()
    registry.register_plot_style(
        "style-only",
        metadata={
            "style": "style-only",
            "builder_id": "builder.missing",
            "renderer_id": "renderer.missing",
        },
    )
    assert registry.find_plot_plugin("style-only") is None

    registry.clear_plot_plugins()
    registry.register_plot_style(
        "style",
        metadata={
            "style": "style",
            "builder_id": "builder",
            "renderer_id": "renderer",
        },
    )
    registry.register_plot_builder("builder", _Builder())
    registry.register_plot_renderer("renderer", _Renderer())
    plugin = registry.find_plot_plugin("style")
    assert plugin is not None
    assert plugin.build({}, {}) == {}
    assert plugin.render({}, {}) == {}


def test_find_plot_plugin_trusted_requires_trust():
    assert registry.find_plot_plugin_trusted("unknown") is None

    registry.register_plot_style(
        "style",
        metadata={
            "style": "style",
            "builder_id": "builder",
            "renderer_id": "renderer",
        },
    )
    registry.register_plot_builder("builder", _Builder())
    registry.register_plot_renderer("renderer", _Renderer())
    assert registry.find_plot_plugin_trusted("style") is None

    registry.mark_plot_builder_trusted("builder")
    registry.mark_plot_renderer_trusted("renderer")
    trusted = registry.find_plot_plugin_trusted("style")
    assert trusted is not None
    assert trusted.build({}, {}) == {}
    assert trusted.render({}, {}) == {}


def test_find_interval_trusted_and_builtin_helpers(monkeypatch):
    registry.clear_interval_plugins()
    descriptor = registry.register_interval_plugin("interval", ExampleIntervalPlugin())
    assert registry.find_interval_plugin_trusted("interval") is None
    registry.mark_interval_trusted("interval")
    assert registry.find_interval_plugin_trusted("interval") is descriptor.plugin
    registry.mark_interval_untrusted("interval")
    assert registry.find_interval_plugin_trusted("interval") is None

    registry.ensure_builtin_plugins()
    registry.ensure_builtin_plugins()  # second call should hit early return


class _EntryPoint:
    def __init__(self, plugin):
        self.name = plugin.plugin_meta["name"]
        self.module = "tests.entry"
        self.attr = None
        self.group = registry._ENTRYPOINT_GROUP
        self._plugin = plugin

    def load(self):
        return self._plugin


def test_load_entrypoint_plugins_include_untrusted(monkeypatch):
    plugin = _SimpleExplanationPlugin()
    entries = [_EntryPoint(plugin)]

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: entries)

    loaded = registry.load_entrypoint_plugins(include_untrusted=True)
    assert loaded == (plugin,)
    assert plugin in registry.list_plugins()


def test_load_entrypoint_plugins_trusted_flow(monkeypatch):
    class TrustedPlugin(_SimpleExplanationPlugin):
        plugin_meta = {**_SimpleExplanationPlugin.plugin_meta, "trust": True, "trusted": True}

    plugin = TrustedPlugin()
    entries = [_EntryPoint(plugin)]

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", lambda: entries)

    loaded = registry.load_entrypoint_plugins()
    assert loaded == (plugin,)
    assert plugin in registry.list_plugins(include_untrusted=False)


def test_load_entrypoint_plugins_errors(monkeypatch):
    registry.clear()

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(registry.importlib_metadata, "entry_points", boom)
    with pytest.warns(RuntimeWarning):
        assert registry.load_entrypoint_plugins() == ()

    class LegacyEntryPoints(list):
        pass

    class LegacyEntryPoint:
        def __init__(self):
            self.name = "legacy"
            self.module = "legacy"
            self.attr = None
            self.group = registry._ENTRYPOINT_GROUP

        def load(self):
            class Bad:
                plugin_meta = None

            return Bad()

    monkeypatch.setattr(
        registry.importlib_metadata, "entry_points", lambda: LegacyEntryPoints([LegacyEntryPoint()])
    )
    with pytest.warns(RuntimeWarning):
        registry.load_entrypoint_plugins()


def test_refresh_interval_descriptor_trust():
    interval_meta = {
        "schema_version": 1,
        "capabilities": ["interval:classification"],
        "name": "interval",
        "version": "0",
        "provider": "tests",
        "modes": ["classification"],
        "dependencies": [],
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "legacy",
        "trust": False,
    }

    class IntervalPlugin:
        plugin_meta = dict(interval_meta)

    registry.register_interval_plugin("interval", IntervalPlugin())

    with pytest.raises(KeyError):
        registry._refresh_interval_descriptor_trust("missing", trusted=True)

    updated = registry._refresh_interval_descriptor_trust("interval", trusted=True)
    assert updated.trusted is True
    assert updated.metadata["trusted"] is True


def test_refresh_plot_builder_and_renderer_trust():
    registry.register_plot_builder("builder", _Builder())
    registry.register_plot_renderer("renderer", _Renderer())

    with pytest.raises(KeyError):
        registry._refresh_plot_builder_trust("unknown", trusted=True)

    builder = registry._refresh_plot_builder_trust("builder", trusted=True)
    assert builder.trusted is True
    renderer = registry._refresh_plot_renderer_trust("renderer", trusted=True)
    assert renderer.trusted is True
    builder_untrusted = registry._refresh_plot_builder_trust("builder", trusted=False)
    assert builder_untrusted.trusted is False
    renderer_untrusted = registry._refresh_plot_renderer_trust("renderer", trusted=False)
    assert renderer_untrusted.trusted is False


class _MutableMeta(MutableMapping):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):  # pragma: no cover - protocol requirement
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class _MutablePlugin:
    def __init__(self, *, trusted: bool):
        self.plugin_meta = _MutableMeta(
            {
                "schema_version": 1,
                "capabilities": ["explain"],
                "name": "mutable",
                "version": "0",
                "provider": "tests",
                "trust": trusted,
            }
        )

    def supports(self, model):  # pragma: no cover - not used
        return True

    def explain(self, model, x, **kwargs):  # pragma: no cover - not used
        return {}


def test_register_existing_plugin_updates_trust_list():
    plugin = _MutablePlugin(trusted=False)
    registry.register(plugin)
    assert plugin not in registry.list_plugins(include_untrusted=False)

    plugin.plugin_meta["trust"] = True
    plugin.plugin_meta["trusted"] = True
    registry.register(plugin)
    assert plugin in registry.list_plugins(include_untrusted=False)


def test_resolve_plugin_from_name_and_safe_supports():
    bad_plugin = SimpleNamespace(plugin_meta=object())
    registry._REGISTRY.append(bad_plugin)
    with pytest.raises(KeyError):
        registry._resolve_plugin_from_name("missing")
    registry._REGISTRY.remove(bad_plugin)

    class RaisingMeta:
        def get(self, key, default=None):
            raise RuntimeError("boom")

    raising_plugin = SimpleNamespace(plugin_meta=RaisingMeta())
    registry._REGISTRY.append(raising_plugin)
    with pytest.raises(KeyError):
        registry._resolve_plugin_from_name("missing")
    registry._REGISTRY.remove(raising_plugin)

    plugin = _SimpleExplanationPlugin()
    registry.register(plugin)
    assert registry._resolve_plugin_from_name("simple.explanation") is plugin

    class BrokenPlugin(_SimpleExplanationPlugin):
        def supports(self, model):
            raise RuntimeError("boom")

    broken = BrokenPlugin()
    registry.register(broken)
    assert registry._safe_supports(broken, object()) is False


def test_refresh_descriptor_and_register_errors():
    registry.clear_explanation_plugins()
    plugin = _SimpleExplanationPlugin()
    descriptor = registry.register_explanation_plugin("simple.extra", plugin)
    registry.unregister(plugin)
    assert registry._resolve_plugin_from_name("simple.explanation") is descriptor.plugin

    with pytest.raises(KeyError):
        registry._refresh_descriptor_trust("missing", trusted=True)

    registry.clear()

    class NoMeta:
        plugin_meta = None

    with pytest.raises(ValueError):
        registry.register(NoMeta())

    class FailingMeta(MutableMapping):
        def __init__(self):
            self._data = {
                "schema_version": 1,
                "capabilities": ["explain"],
                "name": "fail",
                "version": "0",
                "provider": "tests",
                "trust": False,
            }

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            raise RuntimeError("nope")

        def __delitem__(self, key):  # pragma: no cover - protocol requirement
            del self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class FailingPlugin:
        def __init__(self):
            self.plugin_meta = FailingMeta()

        def supports(self, model):  # pragma: no cover - unused
            return False

        def explain(self, model, x, **kwargs):  # pragma: no cover - unused
            return {}

    registry.register(FailingPlugin())

    registry.clear()
    plugin2 = _SimpleExplanationPlugin()
    registry.register(plugin2)
    registry.trust_plugin("simple.explanation")
    registry.untrust_plugin("simple.explanation")
