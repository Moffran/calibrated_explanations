import types
import importlib.metadata as importlib_metadata

import pytest

from calibrated_explanations.plugins import registry


class FakeEntryPoint:
    def __init__(self, name, plugin, dist_name=None):
        self.name = name
        self._plugin = plugin
        self.dist = types.SimpleNamespace(name=dist_name) if dist_name else None
        self.module = None
        self.attr = None

    def load(self):
        return self._plugin


class FakeEntryPoints:
    def __init__(self, eps):
        self._eps = list(eps)

    def select(self, group=None):
        return list(self._eps)


def make_plugin(name):
    class P:
        pass

    plugin = P()
    plugin.plugin_meta = {
        "schema_version": 1,
        "name": name,
        "version": "0.0.0",
        "provider": "vendor",
        "capabilities": ("classification",),
        "trusted": True,
    }
    return plugin


@pytest.fixture(autouse=True)
def cleanup(monkeypatch):
    # Reset registry and trust caches before each test
    registry.clear()
    registry.clear_env_trust_cache()
    registry.clear_trust_warnings()
    registry.clear_explanation_plugins()
    monkeypatch.delenv("CE_TRUST_PLUGIN", raising=False)
    # Default to no entrypoints unless a test monkeypatches it
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: FakeEntryPoints([]))
    yield
    registry.clear()
    registry.clear_env_trust_cache()
    registry.clear_trust_warnings()
    registry.clear_explanation_plugins()
    monkeypatch.delenv("CE_TRUST_PLUGIN", raising=False)


def test_entrypoint_skipped_by_default(monkeypatch):
    name = "third.party.plugin"
    plugin = make_plugin(name)
    ep = FakeEntryPoint(name, plugin)
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: FakeEntryPoints([ep]))

    loaded = registry.load_entrypoint_plugins()
    assert loaded == ()

    report = registry.get_discovery_report()
    assert any(rec.identifier == name for rec in report.skipped_untrusted)


def test_entrypoint_trusted_via_env(monkeypatch):
    name = "third.party.plugin"
    plugin = make_plugin(name)
    ep = FakeEntryPoint(name, plugin)
    monkeypatch.setenv("CE_TRUST_PLUGIN", name)
    registry.clear_env_trust_cache()
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: FakeEntryPoints([ep]))

    loaded = registry.load_entrypoint_plugins()
    assert len(loaded) == 1

    report = registry.get_discovery_report()
    assert any(rec.identifier == name for rec in report.accepted)


def test_entrypoint_trusted_via_pyproject(monkeypatch):
    name = "third.party.plugin"
    plugin = make_plugin(name)
    ep = FakeEntryPoint(name, plugin)
    # Monkeypatch pyproject read to include the trusted identifier
    from calibrated_explanations.core import config_helpers

    monkeypatch.setattr(
        config_helpers, "read_pyproject_section", lambda *args, **kwargs: {"trusted": (name,)}
    )
    # Clear env cache to ensure only pyproject is considered
    registry.clear_env_trust_cache()
    # Prime the pyproject trust cache to the expected value
    registry._PYPROJECT_TRUST_CACHE = {name}
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: FakeEntryPoints([ep]))

    loaded = registry.load_entrypoint_plugins()
    assert len(loaded) == 1

    report = registry.get_discovery_report()
    assert any(rec.identifier == name for rec in report.accepted)
