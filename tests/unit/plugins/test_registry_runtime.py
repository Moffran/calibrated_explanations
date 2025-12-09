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
    yield
    registry.clear_explanation_plugins()


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

    from calibrated_explanations.core.exceptions import ValidationError

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
        registry._verify_plugin_checksum(Plugin(), Plugin.plugin_meta)
