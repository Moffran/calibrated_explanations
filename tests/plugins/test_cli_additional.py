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


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ()),
        ("", ()),
        ("plugin", ("plugin",)),
        (["a", "", 1, None, "b"], ("a", "1", "b")),
        (42, ()),
    ],
)
def testcoerce_string_tuple_normalizes_values(value, expected):
    assert cli.coerce_string_tuple(value) == expected


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

    cli._emit_explanation_descriptor(descriptor)

    out = capsys.readouterr().out
    assert "fallbacks=core" in out
    assert "denied via CE_DENY_PLUGIN" in out


def test_emit_interval_descriptor_marks_denied_plugins(monkeypatch, capsys):
    descriptor = DummyDescriptor(
        "demo.interval",
        metadata={"name": "Interval", "schema_version": 1, "modes": ["x"], "dependencies": []},
        trusted=True,
    )
    monkeypatch.setattr(cli, "is_identifier_denied", lambda identifier: True)

    cli._emit_interval_descriptor(descriptor)

    out = capsys.readouterr().out
    assert "denied via CE_DENY_PLUGIN" in out


def test_emit_plot_descriptor_includes_extras(capsys):
    descriptor = DummySimpleDescriptor(
        "demo.plot",
        metadata={
            "builder_id": "builder",
            "renderer_id": "renderer",
            "fallbacks": ["other"],
            "is_default": True,
            "legacy_compatible": False,
            "default_for": ["explanation"],
        },
    )

    cli._emit_plot_descriptor(descriptor)

    out = capsys.readouterr().out
    assert "is_default=yes" in out
    assert "legacy_compatible=no" in out
    assert "default_for=explanation" in out


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

    cli._emit_plot_builder_descriptor(descriptor)

    out = capsys.readouterr().out
    assert "legacy_compatible=yes" in out


def test_cmd_list_handles_empty_sections(monkeypatch, capsys):
    args = SimpleNamespace(kind="all", trusted_only=False, verbose=False)
    monkeypatch.setattr(cli, "list_explanation_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_interval_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_plot_builder_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_plot_renderer_descriptors", lambda trusted_only: [])
    monkeypatch.setattr(cli, "list_plot_style_descriptors", lambda: [])

    exit_code = cli._cmd_list(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.count("<none>") >= 4


def test_cmd_show_for_plot_styles(monkeypatch, capsys):
    descriptor = DummyDescriptor("demo.plot", metadata={"name": "Plot"}, trusted=True)
    monkeypatch.setattr(cli, "find_plot_style_descriptor", lambda identifier: descriptor)

    args = SimpleNamespace(identifier="demo.plot", kind="plots")
    exit_code = cli._cmd_show(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Identifier : demo.plot" in out
    assert "Trusted    : yes" in out


def test_cmd_trust_uses_renderer_marker(monkeypatch, capsys):
    descriptor = DummyDescriptor("demo.renderer")

    def mark(identifier: str) -> DummyDescriptor:
        assert identifier == "demo.renderer"
        return descriptor

    monkeypatch.setattr(cli, "mark_plot_renderer_trusted", mark)
    args = SimpleNamespace(identifier="demo.renderer", kind="plot-renderers", action="trust")

    exit_code = cli._cmd_trust(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Marked 'demo.renderer' as trusted" in out


def test_main_invalid_arguments_prints_help(capsys):
    exit_code = cli.main(["--unknown"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "usage:" in out


def test_main_without_command_prints_help(monkeypatch, capsys):
    exit_code = cli.main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "usage:" in out

    # banner should not be printed without a command being executed
    assert "Optional tooling" not in out


def test_main_list_plots_flag(capsys):
    """`ce.plugins list --plots` should behave like `list plots` and print styles."""
    exit_code = cli.main(["list", "--plots"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Plot styles" in out


def test_set_default_plot_style_alias(monkeypatch, capsys):
    """`set-default --plot-style` should be accepted as an alias for --style."""
    descriptor = DummyDescriptor("demo.plot", metadata={"builder_id": "b", "renderer_id": "r"})

    # Patch the registry functions that are imported inside the command
    import calibrated_explanations.plugins.registry as registry

    monkeypatch.setattr(
        registry,
        "find_plot_style_descriptor",
        lambda identifier: descriptor if identifier == "demo.plot" else None,
    )
    monkeypatch.setattr(registry, "list_plot_style_descriptors", lambda: [descriptor])
    # Register_plot_style should be callable; capture calls via a simple wrapper
    calls: list[tuple[str, dict]] = []

    def fake_register_plot_style(identifier: str, metadata: dict):
        calls.append((identifier, dict(metadata)))
        return None

    monkeypatch.setattr(cli, "register_plot_style", fake_register_plot_style)

    exit_code = cli.main(["set-default", "--plot-style", "demo.plot"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Set 'demo.plot' as default plot style" in out
    assert calls, "register_plot_style should have been invoked"
    assert calls[0][0] == "demo.plot"
