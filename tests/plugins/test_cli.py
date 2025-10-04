from __future__ import annotations

from calibrated_explanations.plugins import cli
from calibrated_explanations.plugins.registry import ensure_builtin_plugins


def test_cli_list_explanations(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Explanation plugins" in out
    assert "core.explanation.factual" in out


def test_cli_trust_roundtrip(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["untrust", "core.explanation.factual"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Marked 'core.explanation.factual' as untrusted" in out

    exit_code = cli.main(["list", "explanations", "--trusted-only"])
    assert exit_code == 0
    trusted_out = capsys.readouterr().out
    assert "core.explanation.factual" not in trusted_out

    try:
        exit_code = cli.main(["trust", "core.explanation.factual"])
        assert exit_code == 0
        restore = capsys.readouterr().out
        assert "Marked 'core.explanation.factual' as trusted" in restore
    finally:
        cli.main(["trust", "core.explanation.factual"])
        capsys.readouterr()


def test_cli_show_outputs_metadata(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "core.explanation.factual"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Identifier : core.explanation.factual" in out
    assert "Metadata   :" in out
