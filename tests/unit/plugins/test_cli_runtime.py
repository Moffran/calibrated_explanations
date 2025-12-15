from __future__ import annotations

import types

from calibrated_explanations.plugins import cli as cli_module


def test_cli_list_trusted_only(capsys, monkeypatch):
    descriptor = types.SimpleNamespace(
        identifier="plugin.one",
        metadata={"modes": ("regression",), "tasks": ("predict",)},
        trusted=True,
    )

    called: dict[str, bool] = {"explanations": False, "intervals": False, "plots": False}

    monkeypatch.setattr(
        cli_module,
        "list_explanation_descriptors",
        lambda trusted_only=False: called.__setitem__("explanations", trusted_only)
        or (descriptor,),
    )
    monkeypatch.setattr(
        cli_module,
        "list_interval_descriptors",
        lambda trusted_only=False: called.__setitem__("intervals", trusted_only) or (),
    )
    monkeypatch.setattr(
        cli_module,
        "list_plot_style_descriptors",
        lambda: called.__setitem__("plots", True) or (),
    )

    exit_code = cli_module.main(["list", "all", "--trusted-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Explanation plugins" in captured.out
    assert "plugin.one" in captured.out
    assert called["explanations"] is True
    assert called["intervals"] is True
    assert called["plots"] is True


def test_cli_show_missing_descriptor(capsys, monkeypatch):
    monkeypatch.setattr(cli_module, "find_interval_descriptor", lambda identifier: None)

    exit_code = cli_module.main(["show", "missing", "--kind", "intervals"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Interval plugin 'missing' is not registered" in output


def test_cli_trust_command(monkeypatch, capsys):
    descriptor = types.SimpleNamespace(identifier="plugin.trusted")
    called = {"trusted": False}

    def mark(identifier):
        called["trusted"] = identifier
        return descriptor

    monkeypatch.setattr(cli_module, "mark_explanation_trusted", mark)

    exit_code = cli_module.main(["trust", "plugin.trusted"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert called["trusted"] == "plugin.trusted"
    assert "Marked 'plugin.trusted' as trusted" in output


def test_cli_untrust_command(monkeypatch, capsys):
    descriptor = types.SimpleNamespace(identifier="plugin.untrusted")
    called = {"untrusted": False}

    def mark(identifier):
        called["untrusted"] = identifier
        return descriptor

    monkeypatch.setattr(cli_module, "mark_explanation_untrusted", mark)

    exit_code = cli_module.main(["untrust", "plugin.untrusted"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert called["untrusted"] == "plugin.untrusted"
    assert "Marked 'plugin.untrusted' as untrusted" in output


def test_cli_trust_missing(monkeypatch, capsys):
    
    def mark(identifier):
        raise KeyError(f"Plugin '{identifier}' not found")

    monkeypatch.setattr(cli_module, "mark_explanation_trusted", mark)

    exit_code = cli_module.main(["trust", "missing"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Plugin 'missing' not found" in output
