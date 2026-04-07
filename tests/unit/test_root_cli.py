from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from calibrated_explanations.cli import main


def test_should_show_config_snapshot_when_running_ce_config_show(capsys) -> None:
    with patch("calibrated_explanations.cli.ConfigManager.from_sources") as mock_from_sources:
        mock_from_sources.return_value.export_effective.return_value = SimpleNamespace(
            profile_id="default",
            schema_version="1",
            values={"env.CE_PLOT_STYLE": "legacy"},
            sources={"env.CE_PLOT_STYLE": "env"},
        )
        exit_code = main(["config", "show"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "profile_id=default" in captured.out
    assert "schema_version=1" in captured.out


def test_should_export_config_snapshot_when_running_ce_config_export(capsys) -> None:
    with patch("calibrated_explanations.cli.ConfigManager.from_sources") as mock_from_sources:
        mock_from_sources.return_value.export_effective.return_value = SimpleNamespace(
            profile_id="default",
            schema_version="1",
            values={"env.CE_PLOT_STYLE": "legacy"},
            sources={"env.CE_PLOT_STYLE": "env"},
        )
        exit_code = main(["config", "export"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "profile_id" in captured.out
    assert "values" in captured.out
    assert "sources" in captured.out


def test_should_delegate_plugins_command_when_running_ce_plugins_subcommand() -> None:
    with patch("calibrated_explanations.cli.plugins_main", return_value=0) as mock_plugins:
        exit_code = main(["plugins", "list", "--kind", "all"])

    assert exit_code == 0
    mock_plugins.assert_called_once_with(["list", "--kind", "all"])
