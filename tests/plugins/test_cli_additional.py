from __future__ import annotations

from types import SimpleNamespace

import pytest

from calibrated_explanations.plugins import cli


class DummyDescriptor:
    def __init__(
        self, identifier: str, metadata: dict[str, object] | None = None, *, trusted: bool = True
    ):
        self.identifier = identifier
        self.metadata = metadata or {}
        self.trusted = trusted


class DummySimpleDescriptor:
    def __init__(self, identifier: str, metadata: dict[str, object] | None = None):
        self.identifier = identifier
        self.metadata = metadata or {}


def test_emit_explanation_descriptor_reports_fallbacks(monkeypatch, capsys):
    descriptor = DummyDescriptor(
        "demo.explainer",
        metadata={
            "name": "Demo",
            "schema_version": 2,
            "modes": ["fast"],
            "tasks": ["demo"],
            "interval_dependency": ["interval"],
            "plot_dependency": ["plot"],
            "fallbacks": ["core"],
        },
        trusted=False,
    )
    monkeypatch.setattr(
        cli, "is_identifier_denied", lambda identifier: identifier == "demo.explainer"
    )

    cli.emit_explanation_descriptor(descriptor)

    out = capsys.readouterr().out
    assert "fallbacks=core" in out
    assert "denied via CE_DENY_PLUGIN" in out






def test_emit_plot_builder_descriptor_reports_legacy(monkeypatch, capsys):
    descriptor = DummyDescriptor(
        "demo.builder",
        metadata={
            "name": "Builder",
            "schema_version": 1,
            "style": "chart",
            "capabilities": ["plots"],
            "output_formats": ["json"],
            "dependencies": [],
            "legacy_compatible": True,
        },
    )

    cli.emit_plot_builder_descriptor(descriptor)

    out = capsys.readouterr().out
    assert "legacy_compatible=yes" in out


def test_cmd_list_handles_empty_sections(monkeypatch, capsys):
    args = SimpleNamespace(kind="all", trusted_only=False, verbose=False)
    monkeypatch.setattr(cli, "list_explanation_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_interval_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_plot_builder_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_plot_renderer_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_plot_style_descriptors", lambda: [])

    exit_code = cli.cmd_list(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.count("<none>") >= 4


def test_cmd_show_for_plot_styles(monkeypatch, capsys):
    descriptor = DummyDescriptor("demo.plot", metadata={"name": "Plot"}, trusted=True)
    monkeypatch.setattr(cli, "find_plot_style_descriptor", lambda identifier: descriptor)

    args = SimpleNamespace(identifier="demo.plot", kind="plots")
    exit_code = cli.cmd_show(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Identifier : demo.plot" in out
    assert "Trusted    : yes" in out










