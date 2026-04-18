from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path("scripts/quality/check_trust_mutation_primitive.py")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def run_checker(root: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    report = root / "report.json"
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--root",
        str(root / "src"),
        "--report",
        str(report),
    ]
    if check:
        command.append("--check")
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_checker_blocks_direct_trusted_set_mutation(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/plugins/registry.py",
        """
        _TRUSTED_INTERVALS = set()

        def mark_interval(identifier):
            _TRUSTED_INTERVALS.add(identifier)
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "Disallowed trust mutation sites detected" in result.stdout
    assert "trusted_set.add" in result.stdout


def test_checker_allows_mutation_in_primitive_module(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/plugins/_trust.py",
        """
        _TRUSTED_INTERVALS = set()

        def mutate(identifier):
            _TRUSTED_INTERVALS.add(identifier)
        """,
    )
    result = run_checker(tmp_path)
    assert result.returncode == 0
    assert "Trust mutation primitive check passed" in result.stdout

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["total_violations"] == 0
    assert report["total_records"] == 1
    assert report["package_root"] == "src/calibrated_explanations"
