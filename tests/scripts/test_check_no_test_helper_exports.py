from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path("scripts/quality/check_no_test_helper_exports.py")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    report = root / "report.json"
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--root",
        str(root / "src"),
        "--report",
        str(report),
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_checker_blocks_banned_registry_exports(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/calibrated_explanations/plugins/registry.py",
        """
        __all__ = ["mark_plot_builder_trusted"]

        def mark_plot_builder_trusted(identifier):
            return identifier
        """,
    )
    result = _run_checker(tmp_path)
    assert result.returncode == 1
    assert "mark_plot_builder_trusted" in result.stdout
    assert "explicitly banned export" in result.stdout


def test_checker_blocks_docstring_labeled_testing_helpers(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/calibrated_explanations/plugins/example.py",
        """
        __all__ = ["normalise_trust"]

        def normalise_trust(meta):
            \"\"\"Public wrapper around internal trust normalisation used by tests.\"\"\"
            return _normalise_trust(meta)

        def _normalise_trust(meta):
            return bool(meta)
        """,
    )
    result = _run_checker(tmp_path)
    assert result.returncode == 1
    assert "docstring labels symbol as testing helper" in result.stdout


def test_checker_passes_for_clean_exports(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/calibrated_explanations/plugins/example.py",
        """
        __all__ = ["validate_config"]

        def validate_config(value):
            \"\"\"Validate plugin config value.\"\"\"
            return bool(value)
        """,
    )
    result = _run_checker(tmp_path)
    assert result.returncode == 0
    assert "No prohibited test-helper exports detected." in result.stdout

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["total_violations"] == 0
