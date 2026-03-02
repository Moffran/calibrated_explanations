from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/quality/check_logging_domains.py")


def write_module(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_check_logging_domains_passes_for_dunder_name_and_project_literal(tmp_path: Path) -> None:
    package = tmp_path / "src" / "calibrated_explanations"
    write_module(
        package / "good_module.py",
        (
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "gov = logging.getLogger('calibrated_explanations.governance.feature_filter')\n"
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(package),
            "--report",
            str(tmp_path / "report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Logger domain check passed" in result.stdout


def test_check_logging_domains_fails_for_non_project_literal(tmp_path: Path) -> None:
    package = tmp_path / "src" / "calibrated_explanations"
    write_module(
        package / "bad_module.py",
        (
            "import logging\n"
            "logger = logging.getLogger('external.service')\n"
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(package),
            "--report",
            str(tmp_path / "report.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Logger domain violations detected" in result.stdout
    assert "external.service" in result.stdout
