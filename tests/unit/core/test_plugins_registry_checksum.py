"""Additional coverage for plugin checksum validation logic."""

from __future__ import annotations

import importlib
import sys
import types
import textwrap
from pathlib import Path

import pytest

from calibrated_explanations.plugins import registry


def _write_checksum_plugin(tmp_path: Path) -> str:
    """Write a temporary plugin module that self-reports its checksum."""

    module_name = "checksum_test_plugin"
    module_path = tmp_path / f"{module_name}.py"
    module_code = textwrap.dedent(
        """
        from __future__ import annotations

        import hashlib
        from pathlib import Path


        def _module_sha256() -> str:
            return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


        class Plugin:
            plugin_meta = {
                "schema_version": 1,
                "capabilities": ["explain"],
                "name": "tests.checksum.valid",
                "version": "0.0-test",
                "provider": "tests",
                "modes": ["factual"],
                "tasks": ["classification"],
                "dependencies": [],
                "trust": True,
                "trusted": True,
                "checksum": {"sha256": _module_sha256()},
            }

            def supports(self, model):  # pragma: no cover - exercised via registry
                return True

            def explain(self, model, x, **kwargs):  # pragma: no cover - exercised via registry
                return {}


        PLUGIN = Plugin()
        """
    )
    module_path.write_text(module_code)
    return module_name


def test_register_explanation_plugin_verifies_checksum(tmp_path, monkeypatch):
    """A correct checksum allows registration without raising warnings."""

    module_name = _write_checksum_plugin(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    registry.clear()
    registry.clear_explanation_plugins()

    try:
        module = importlib.import_module(module_name)
        plugin = module.PLUGIN

        descriptor = registry.register_explanation_plugin("checksum.valid", plugin)

        # Metadata should reflect that trust fields were harmonised after verification.
        assert descriptor.metadata["trusted"] is True
        assert descriptor.metadata["trust"] is True
    finally:
        registry.clear()
        registry.clear_explanation_plugins()
        sys.modules.pop(module_name, None)


def test_register_explanation_plugin_rejects_checksum_mismatch(tmp_path, monkeypatch):
    """An incorrect checksum raises a ValidationError to prevent tampering."""
    from calibrated_explanations.core.exceptions import ValidationError

    module_name = _write_checksum_plugin(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    registry.clear()
    registry.clear_explanation_plugins()

    try:
        module = importlib.import_module(module_name)
        plugin = module.Plugin()
        plugin.plugin_meta = dict(module.PLUGIN.plugin_meta)
        plugin.plugin_meta["checksum"] = "0" * 64

        with pytest.raises(ValidationError, match="Checksum mismatch"):
            registry.register_explanation_plugin("checksum.invalid", plugin)
    finally:
        registry.clear()
        registry.clear_explanation_plugins()
        sys.modules.pop(module_name, None)


def test_register_explanation_plugin_warns_when_module_missing(tmp_path):
    """Missing module files emit a warning instead of crashing verification."""

    registry.clear()
    registry.clear_explanation_plugins()

    module_name = "tests.plugins.missing_checksum"
    synthetic_module = types.ModuleType(module_name)
    synthetic_module.__file__ = str(tmp_path / "missing_checksum.py")
    sys.modules[module_name] = synthetic_module

    class MissingFilePlugin:
        plugin_meta = {
            "schema_version": 1,
            "capabilities": ["explain"],
            "name": "tests.checksum.missing",
            "version": "0.0-test",
            "provider": "tests",
            "modes": ["factual"],
            "tasks": ["classification"],
            "dependencies": [],
            "trust": False,
            "trusted": False,
            "checksum": {"sha256": "deadbeef" * 8},
        }

        def supports(self, model):  # pragma: no cover - exercised via registry
            return True

        def explain(self, model, x, **kwargs):  # pragma: no cover - exercised via registry
            return {}

    MissingFilePlugin.__module__ = module_name

    try:
        with pytest.warns(UserWarning, match="Cannot verify checksum"):
            registry.register_explanation_plugin("checksum.missing", MissingFilePlugin())
    finally:
        registry.clear()
        registry.clear_explanation_plugins()
        sys.modules.pop(module_name, None)
