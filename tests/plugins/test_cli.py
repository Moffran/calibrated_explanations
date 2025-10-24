from __future__ import annotations

from calibrated_explanations.plugins import cli
from calibrated_explanations.plugins.registry import ensure_builtin_plugins

OPTIONAL_BANNER = "Optional tooling: the plugin CLI is opt-in."


def _assert_banner(output: str) -> None:
    assert OPTIONAL_BANNER in output


def test_cli_list_explanations(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Explanation plugins" in out
    assert "core.explanation.factual" in out


def test_cli_list_marks_denied_plugins(monkeypatch, capsys):
    ensure_builtin_plugins()
    monkeypatch.setenv("CE_DENY_PLUGIN", "core.explanation.factual")
    exit_code = cli.main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "denied via CE_DENY_PLUGIN" in out


def test_cli_trust_roundtrip(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["untrust", "core.explanation.factual"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Marked 'core.explanation.factual' as untrusted" in out

    exit_code = cli.main(["list", "explanations", "--trusted-only"])
    assert exit_code == 0
    trusted_out = capsys.readouterr().out
    _assert_banner(trusted_out)
    assert "core.explanation.factual" not in trusted_out

    try:
        exit_code = cli.main(["trust", "core.explanation.factual"])
        assert exit_code == 0
        restore = capsys.readouterr().out
        _assert_banner(restore)
        assert "Marked 'core.explanation.factual' as trusted" in restore
    finally:
        cli.main(["trust", "core.explanation.factual"])
        capsys.readouterr()


def test_cli_show_outputs_metadata(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "core.explanation.factual"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Identifier : core.explanation.factual" in out
    assert "Metadata   :" in out


def test_cli_list_intervals(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list", "intervals"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Interval calibrators" in out
    assert "core.interval.legacy" in out


def test_cli_show_interval_descriptor(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "core.interval.legacy", "--kind", "intervals"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Identifier : core.interval.legacy" in out
    assert "Metadata   :" in out


def test_cli_list_plot_builders(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list", "plot-builders"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Plot builders" in out


def test_cli_list_plot_renderers(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list", "plot-renderers"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Plot renderers" in out


def test_cli_trust_interval(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["untrust", "core.interval.legacy", "--kind", "intervals"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Marked 'core.interval.legacy' as untrusted" in out

    try:
        exit_code = cli.main(["trust", "core.interval.legacy", "--kind", "intervals"])
        assert exit_code == 0
        restore = capsys.readouterr().out
        _assert_banner(restore)
        assert "Marked 'core.interval.legacy' as trusted" in restore
    finally:
        cli.main(["trust", "core.interval.legacy", "--kind", "intervals"])
        capsys.readouterr()


def test_cli_trust_plot_builder(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["untrust", "core.plot.legacy", "--kind", "plot-builders"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Marked 'core.plot.legacy' as untrusted" in out

    try:
        exit_code = cli.main(["trust", "core.plot.legacy", "--kind", "plot-builders"])
        assert exit_code == 0
        restore = capsys.readouterr().out
        _assert_banner(restore)
        assert "Marked 'core.plot.legacy' as trusted" in restore
    finally:
        cli.main(["trust", "core.plot.legacy", "--kind", "plot-builders"])
        capsys.readouterr()


def test_cli_list_all(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list", "all"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Explanation plugins" in out
    assert "Interval calibrators" in out
    assert "Plot builders" in out


def test_cli_show_plot_builder(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "core.plot.legacy", "--kind", "plot-builders"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Identifier : core.plot.legacy" in out


def test_cli_show_plot_renderer(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "core.plot.legacy", "--kind", "plot-renderers"])
    assert exit_code == 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "Identifier : core.plot.legacy" in out


def test_cli_invalid_command(capsys):
    exit_code = cli.main(["invalid"])
    assert exit_code != 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "usage:" in out


def test_cli_show_unknown_plugin(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "unknown"])
    assert exit_code != 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "not registered" in out.lower()


def test_cli_trust_unknown_plugin(capsys):
    exit_code = cli.main(["trust", "unknown"])
    assert exit_code != 0
    out = capsys.readouterr().out
    _assert_banner(out)
    assert "not registered" in out.lower()
