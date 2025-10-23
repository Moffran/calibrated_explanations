"""Smoke tests for the ``ce.plugins`` console entry point.

These guard against packaging regressions by exercising the CLI commands
registered in ``pyproject.toml`` and ensure trust toggles round-trip through the
registry at runtime. They intentionally avoid spawning a new interpreter so the
output remains deterministic and fast during CI runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin
from calibrated_explanations.plugins.cli import main
from calibrated_explanations.plugins.registry import (
    clear_explanation_plugins,
    ensure_builtin_plugins,
    find_explanation_descriptor,
    register_explanation_plugin,
    unregister,
)


@pytest.fixture(autouse=True)
def _ensure_builtins() -> None:
    """Make sure built-in plugins are available for every test."""

    ensure_builtin_plugins()


class _TemporaryFactualPlugin(LegacyFactualExplanationPlugin):
    """Legacy factual plugin variant that starts untrusted."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.cli.untrusted",
        "trusted": False,
        "trust": {"trusted": False},
    }


def _cleanup_plugin(plugin: LegacyFactualExplanationPlugin) -> None:
    """Remove the plugin and restore built-in registry state."""

    unregister(plugin)
    clear_explanation_plugins()
    ensure_builtin_plugins()


def test_console_script_entrypoint_exposed() -> None:
    """The ``ce.plugins`` console script should be published for packaging."""

    contents = Path("pyproject.toml").read_text("utf-8")
    assert (
        '"ce.plugins" = "calibrated_explanations.plugins.cli:main"' in contents
    ), "Console script declaration missing from pyproject.toml"


def test_list_all_prints_registered_plugins(capfd: pytest.CaptureFixture[str]) -> None:
    """Listing all plugin categories should succeed and include built-ins."""

    exit_code = main(["list", "all"])
    assert exit_code == 0
    stdout, stderr = capfd.readouterr()
    assert "Explanation plugins" in stdout
    assert "core.explanation.factual" in stdout
    assert "Interval calibrators" in stdout
    assert "core.interval.legacy" in stdout
    assert "Plot styles" in stdout
    assert stderr == ""


def test_show_interval_descriptor(capfd: pytest.CaptureFixture[str]) -> None:
    """Showing metadata for a registered interval plugin should succeed."""

    exit_code = main(["show", "core.interval.legacy", "--kind", "intervals"])
    assert exit_code == 0
    stdout, stderr = capfd.readouterr()
    assert "Identifier : core.interval.legacy" in stdout
    assert "Trusted" in stdout
    assert stderr == ""


def test_show_missing_descriptor(capfd: pytest.CaptureFixture[str]) -> None:
    """Unknown identifiers should return a non-zero exit code."""

    exit_code = main(["show", "tests.missing.cli", "--kind", "intervals"])
    assert exit_code == 1
    stdout, stderr = capfd.readouterr()
    assert "not registered" in stdout
    assert stderr == ""


def test_trust_and_untrust_commands(capfd: pytest.CaptureFixture[str]) -> None:
    """Trusted state changes should flow through the registry."""

    plugin = _TemporaryFactualPlugin()
    register_explanation_plugin("tests.cli.untrusted", plugin)
    try:
        descriptor = find_explanation_descriptor("tests.cli.untrusted")
        assert descriptor is not None
        assert not descriptor.trusted

        exit_code = main(["trust", "tests.cli.untrusted"])
        assert exit_code == 0
        stdout, stderr = capfd.readouterr()
        assert "Marked 'tests.cli.untrusted' as trusted" in stdout
        assert stderr == ""
        descriptor = find_explanation_descriptor("tests.cli.untrusted")
        assert descriptor is not None and descriptor.trusted

        exit_code = main(["untrust", "tests.cli.untrusted"])
        assert exit_code == 0
        stdout, stderr = capfd.readouterr()
        assert "Marked 'tests.cli.untrusted' as untrusted" in stdout
        assert stderr == ""
        descriptor = find_explanation_descriptor("tests.cli.untrusted")
        assert descriptor is not None and not descriptor.trusted
    finally:
        _cleanup_plugin(plugin)
