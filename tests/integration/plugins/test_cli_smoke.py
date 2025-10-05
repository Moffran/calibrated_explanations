from calibrated_explanations.plugins import cli
from calibrated_explanations.plugins.registry import ensure_builtin_plugins


def test_cli_list_all_smoke(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["list", "all"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Explanation plugins" in captured.out
    assert "Interval calibrators" in captured.out


def test_cli_show_interval_smoke(capsys):
    ensure_builtin_plugins()
    exit_code = cli.main(["show", "core.interval.legacy", "--kind", "intervals"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Identifier" in captured.out
    assert "core.interval.legacy" in captured.out
