from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path("scripts/quality/check_config_manager_usage.py")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def run_checker(
    root: Path,
    *,
    check: bool = True,
    scope: str = "targeted",
) -> subprocess.CompletedProcess[str]:
    report = root / "report.json"
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--root",
        str(root / "src"),
        "--report",
        str(report),
        "--scope",
        scope,
    ]
    if check:
        command.append("--check")
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_checker_fails_for_os_getenv_in_migrated_module(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/logging.py",
        """
        import os
        value = os.getenv("CE_TELEMETRY_DIAGNOSTIC_MODE")
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "ADR-034 §6 boundary violations" in result.stdout


def test_checker_passes_when_no_direct_reads(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/plugins/manager.py",
        """
        from calibrated_explanations.core.config_manager import ConfigManager
        class Manager:
            def __init__(self):
                self.cfg = ConfigManager.from_sources()
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0
    assert "ConfigManager usage check passed" in result.stdout
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["package_root"] == "src/calibrated_explanations"


def test_checker_allows_pytest_current_test_probe(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/plugins/registry.py",
        """
        import os
        active = os.getenv("PYTEST_CURRENT_TEST")
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["package_root"] == "src/calibrated_explanations"


def test_checker_blocks_from_sources_in_non_init_method(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/core/prediction/orchestrator.py",
        """
        from calibrated_explanations.core.config_manager import ConfigManager
        class Orchestrator:
            def _fetch_env(self, key):
                return ConfigManager.from_sources().env(key)
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "ADR-034 §3 lifecycle violations" in result.stdout


def test_checker_allows_module_level_singleton(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/logging.py",
        """
        from calibrated_explanations.core.config_manager import ConfigManager
        _cfg = ConfigManager.from_sources()
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0


def test_checker_allows_from_env_when_manager_is_built_in_singleton_helper(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/cache/cache.py",
        """
        from calibrated_explanations.core.config_manager import ConfigManager
        _cfg = None
        def _get_cache_config_manager():
            global _cfg
            if _cfg is None:
                _cfg = ConfigManager.from_sources()
            return _cfg
        class CacheConfig:
            @classmethod
            def from_env(cls, base=None, *, config_manager=None):
                mgr = config_manager or _get_cache_config_manager()
                return cls()
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0


def test_checker_allows_root_cli_handler_from_sources(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/cli.py",
        """
        from calibrated_explanations.core.config_manager import ConfigManager
        def _cmd_config_show(args):
            snap = ConfigManager.from_sources().export_effective()
            print(snap)
            return 0
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0


def test_checker_runtime_scope_flags_non_target_runtime_file(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/runtime_module.py",
        """
        import os
        value = os.getenv("CE_PLOT_STYLE")
        """,
    )
    result = run_checker(tmp_path, scope="runtime")
    assert result.returncode == 1
    assert "ADR-034 §6 boundary violations" in result.stdout


def test_checker_targeted_scope_ignores_non_target_runtime_file(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/runtime_module.py",
        """
        import os
        value = os.getenv("CE_PLOT_STYLE")
        """,
    )
    result = run_checker(tmp_path, scope="targeted")
    assert result.returncode == 0
