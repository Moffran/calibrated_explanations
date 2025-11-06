import argparse
import runpy
import sys
from types import SimpleNamespace

import pytest

from calibrated_explanations.plugins import cli


class DummyDescriptor(SimpleNamespace):
    """Simple helper for constructing descriptor objects."""

    metadata: dict
    identifier: str
    trusted: bool = True


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ()),
        ("", ()),
        ("single", ("single",)),
        (["a", "", 0, "b"], ("a", "b")),
        (123, ()),
    ],
)
def test_string_tuple_variants(value, expected):
    assert cli._string_tuple(value) == expected


def test_emit_descriptor_helpers_cover_branches(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "is_identifier_denied", lambda identifier: identifier.endswith("-denied")
    )

    explanation = DummyDescriptor(
        identifier="explainer-denied",
        metadata={
            "name": "explainer",
            "schema_version": 1,
            "modes": ["fast", "accurate"],
            "tasks": ("classification", None, ""),
            "interval_dependency": ["interval-a"],
            "plot_dependency": ("plot-a",),
            "fallbacks": ("fallback-1", "fallback-2"),
        },
        trusted=False,
    )

    interval = DummyDescriptor(
        identifier="interval-denied",
        metadata={
            "name": "interval",
            "schema_version": 2,
            "modes": ["batch"],
            "dependencies": ("dep-a",),
        },
        trusted=True,
    )

    plot_style = SimpleNamespace(
        identifier="plot-style",
        metadata={
            "builder_id": "builder-x",
            "renderer_id": "renderer-x",
            "fallbacks": ["fallback"],
            "is_default": True,
            "legacy_compatible": False,
            "default_for": ["task-a", "task-b"],
        },
    )

    plot_builder = DummyDescriptor(
        identifier="builder",
        metadata={
            "name": "builder",
            "schema_version": 3,
            "style": "static",
            "capabilities": ["html"],
            "output_formats": ("png",),
            "dependencies": (),
            "legacy_compatible": True,
        },
        trusted=True,
    )

    plot_renderer = DummyDescriptor(
        identifier="renderer",
        metadata={
            "name": "renderer",
            "schema_version": 4,
            "capabilities": ["interactive"],
            "output_formats": ["svg"],
            "dependencies": ["builder"],
            "supports_interactive": True,
        },
        trusted=False,
    )

    cli._emit_explanation_descriptor(explanation)
    cli._emit_interval_descriptor(interval)
    cli._emit_plot_descriptor(plot_style)
    cli._emit_plot_builder_descriptor(plot_builder)
    cli._emit_plot_renderer_descriptor(plot_renderer)

    output = capsys.readouterr().out
    assert "denied via CE_DENY_PLUGIN" in output
    assert "fallbacks=fallback-1, fallback-2" in output
    assert "default_for=task-a, task-b" in output
    assert "legacy_compatible=yes" in output
    assert "supports_interactive=yes" in output


def test_cmd_list_all_branches(monkeypatch, capsys):
    calls: dict[str, bool] = {}

    def stub_explanations(*, trusted_only: bool):
        calls["explanations"] = trusted_only
        return []

    def stub_intervals(*, trusted_only: bool):
        calls["intervals"] = trusted_only
        return [DummyDescriptor(identifier="interval", metadata={}, trusted=True)]

    def stub_plot_builders(*, trusted_only: bool):
        calls["plot-builders"] = trusted_only
        return []

    def stub_plot_renderers(*, trusted_only: bool):
        calls["plot-renderers"] = trusted_only
        return []

    def stub_plot_styles():
        return [SimpleNamespace(identifier="plot", metadata={})]

    monkeypatch.setattr(cli, "list_explanation_descriptors", stub_explanations)
    monkeypatch.setattr(cli, "list_interval_descriptors", stub_intervals)
    monkeypatch.setattr(cli, "list_plot_builder_descriptors", stub_plot_builders)
    monkeypatch.setattr(cli, "list_plot_renderer_descriptors", stub_plot_renderers)
    monkeypatch.setattr(cli, "list_plot_style_descriptors", stub_plot_styles)
    monkeypatch.setattr(cli, "is_identifier_denied", lambda _identifier: False)

    args = argparse.Namespace(kind="all", trusted_only=True)
    exit_code = cli._cmd_list(args)

    assert exit_code == 0
    assert calls == {
        "explanations": True,
        "intervals": True,
        "plot-builders": True,
        "plot-renderers": True,
    }

    output = capsys.readouterr().out
    assert "Explanation plugins" in output
    assert "<none>" in output
    assert "Plot styles" in output


def test_cmd_list_explanations_descriptor(monkeypatch, capsys):
    descriptor = DummyDescriptor(identifier="expl", metadata={}, trusted=True)

    monkeypatch.setattr(cli, "list_explanation_descriptors", lambda **kwargs: [descriptor])
    monkeypatch.setattr(cli, "is_identifier_denied", lambda _identifier: False)

    args = argparse.Namespace(kind="explanations", trusted_only=False)
    exit_code = cli._cmd_list(args)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "expl" in output


def test_cmd_list_intervals_empty(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_interval_descriptors", lambda **kwargs: [])
    monkeypatch.setattr(cli, "is_identifier_denied", lambda _identifier: False)

    args = argparse.Namespace(kind="intervals", trusted_only=False)
    exit_code = cli._cmd_list(args)

    assert exit_code == 0
    assert "<none>" in capsys.readouterr().out


def test_cmd_list_plot_renderers_descriptor(monkeypatch, capsys):
    descriptor = DummyDescriptor(identifier="renderer", metadata={}, trusted=True)

    def stub(**kwargs):
        assert kwargs == {"trusted_only": True}
        return [descriptor]

    monkeypatch.setattr(cli, "list_plot_renderer_descriptors", stub)
    monkeypatch.setattr(cli, "is_identifier_denied", lambda _identifier: False)

    args = argparse.Namespace(kind="plot-renderers", trusted_only=True)
    exit_code = cli._cmd_list(args)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "renderer" in output


def test_cmd_list_plot_styles_none(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_plot_style_descriptors", lambda: [])

    args = argparse.Namespace(kind="plots", trusted_only=False)
    exit_code = cli._cmd_list(args)

    assert exit_code == 0
    assert "<none>" in capsys.readouterr().out


def test_cmd_show_not_registered(monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_plot_style_descriptor", lambda identifier: None)
    args = argparse.Namespace(identifier="missing", kind="plots")
    exit_code = cli._cmd_show(args)
    assert exit_code == 1
    assert "Plot style 'missing' is not registered" in capsys.readouterr().out


def test_cmd_show_registered_with_trust_flag(monkeypatch, capsys):
    descriptor = DummyDescriptor(identifier="plugin", metadata={"foo": "bar"}, trusted=False)
    monkeypatch.setattr(cli, "find_explanation_descriptor", lambda identifier: descriptor)
    args = argparse.Namespace(identifier="plugin", kind="explanations")
    exit_code = cli._cmd_show(args)
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Identifier : plugin" in output
    assert "Trusted    : no" in output


@pytest.mark.parametrize(
    "kind, finder_name",
    [
        ("intervals", "find_interval_descriptor"),
        ("plot-builders", "find_plot_builder_descriptor"),
        ("plot-renderers", "find_plot_renderer_descriptor"),
    ],
)
def test_cmd_show_other_kinds(monkeypatch, capsys, kind, finder_name):
    descriptor = DummyDescriptor(identifier="plugin", metadata={"foo": "bar"}, trusted=True)
    monkeypatch.setattr(cli, finder_name, lambda identifier: descriptor)
    args = argparse.Namespace(identifier="plugin", kind=kind)

    exit_code = cli._cmd_show(args)
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Identifier : plugin" in output


def test_cmd_trust_plot_renderer(monkeypatch, capsys):
    captured = {}

    def stub(identifier: str):
        captured["identifier"] = identifier
        return SimpleNamespace(identifier=identifier)

    monkeypatch.setattr(cli, "mark_plot_renderer_trusted", stub)
    monkeypatch.setattr(cli, "mark_plot_renderer_untrusted", stub)

    args = argparse.Namespace(identifier="renderer", kind="plot-renderers", action="trust")
    exit_code = cli._cmd_trust(args)
    assert exit_code == 0
    assert captured["identifier"] == "renderer"
    assert "Marked 'renderer' as trusted" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kind, marker_name",
    [
        ("explanations", "mark_explanation_trusted"),
        ("intervals", "mark_interval_trusted"),
        ("plot-builders", "mark_plot_builder_trusted"),
    ],
)
def test_cmd_trust_other_kinds(monkeypatch, capsys, kind, marker_name):
    captured: dict[str, str] = {}

    def stub(identifier: str):
        captured[kind] = identifier
        return SimpleNamespace(identifier=identifier)

    monkeypatch.setattr(cli, marker_name, stub)

    args = argparse.Namespace(identifier="target", kind=kind, action="trust")
    exit_code = cli._cmd_trust(args)
    assert exit_code == 0
    assert captured[kind] == "target"
    assert "Marked 'target' as trusted" in capsys.readouterr().out


def test_cmd_trust_keyerror(monkeypatch, capsys):
    def stub(identifier: str):
        raise KeyError("missing")

    monkeypatch.setattr(cli, "mark_explanation_trusted", stub)
    monkeypatch.setattr(cli, "mark_explanation_untrusted", stub)

    args = argparse.Namespace(identifier="missing", kind="explanations", action="trust")
    exit_code = cli._cmd_trust(args)
    assert exit_code == 1
    assert "missing" in capsys.readouterr().out


def test_main_invalid_arguments(capsys):
    exit_code = cli.main(["--not-a-real-arg"])
    assert exit_code == 2
    output = capsys.readouterr().out
    assert "usage:" in output


def test_main_no_command(capsys):
    exit_code = cli.main([])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "usage:" in output


def test_main_list_flow(monkeypatch, capsys):
    def stub_explanations(*, trusted_only: bool):
        return []

    def stub_intervals(*, trusted_only: bool):
        return [DummyDescriptor(identifier="interval", metadata={}, trusted=True)]

    def stub_plot_builders(*, trusted_only: bool):
        return [DummyDescriptor(identifier="builder", metadata={}, trusted=True)]

    def stub_plot_renderers(*, trusted_only: bool):
        return []

    def stub_plot_styles():
        return [SimpleNamespace(identifier="plot", metadata={})]

    monkeypatch.setattr(cli, "list_explanation_descriptors", stub_explanations)
    monkeypatch.setattr(cli, "list_interval_descriptors", stub_intervals)
    monkeypatch.setattr(cli, "list_plot_builder_descriptors", stub_plot_builders)
    monkeypatch.setattr(cli, "list_plot_renderer_descriptors", stub_plot_renderers)
    monkeypatch.setattr(cli, "list_plot_style_descriptors", stub_plot_styles)
    monkeypatch.setattr(cli, "is_identifier_denied", lambda _identifier: False)

    exit_code = cli.main(["list", "all", "--trusted-only"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Optional tooling" in output
    assert "Plot builders" in output


def test_cli_module_main_entry(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["calibrated_explanations.plugins.cli"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "calibrated_explanations.plugins.cli",
            run_name="__main__",
            alter_sys=True,
        )

    assert exc.value.code == 0
