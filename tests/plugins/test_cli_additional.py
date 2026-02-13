from __future__ import annotations

from types import SimpleNamespace

from calibrated_explanations.plugins import cli

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




