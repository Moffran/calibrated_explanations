from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path("scripts/quality/check_std001_nomenclature.py")


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


def test_should_fail_when_non_legacy_dunder_exists_outside_allowlist(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/core/random_module.py",
        """
        class RandomModule:
            def __hidden(self):
                return 1
        """,
    )

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "STD-001 nomenclature violations detected" in result.stdout
    assert "non_legacy_dunder_definition" in result.stdout


def test_should_fail_when_expired_bridge_symbol_is_present(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/core/calibrated_explainer.py",
        """
        class CalibratedExplainer:
            def __init__(self):
                self.__initialized = True
        """,
    )

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "STD-001 nomenclature violations detected" in result.stdout
    assert "__initialized" in result.stdout


def test_should_report_utility_import_bridge_records(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/explanations/explanation.py",
        """
        from ..utils.helper import assign_threshold as normalize_threshold

        class Explanation:
            def run(self):
                return normalize_threshold(0.5)
        """,
    )

    result = run_checker(tmp_path)

    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert any(row["violation_kind"] == "utility_import_bridge" for row in report["records"])
    assert report["total_violations"] == 0


def test_should_fail_when_mangled_private_symbol_is_not_allowlisted(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/core/unknown.py",
        """
        class Unknown:
            def run(self):
                return self._Unknown__secret
        """,
    )

    result = run_checker(tmp_path)

    assert result.returncode == 1
    assert "__secret" in result.stdout


def test_should_not_detect_removed_shim_surfaces_for_serialization_and_builders(tmp_path: Path) -> None:
    write(
        tmp_path / "src/calibrated_explanations/serialization.py",
        """
        def _schema_validate_payload(obj):
            pass

        def to_json(exp):
            _schema_validate_payload(exp)
        """,
    )
    write(
        tmp_path / "src/calibrated_explanations/viz/builders.py",
        """
        def _legacy_get_fill_color(x):
            return x
        """,
    )

    result = run_checker(tmp_path)

    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    shim_records = [row for row in report["records"] if row["violation_kind"] == "non_legacy_transitional_shim"]
    assert not any("validate_payload" in row["symbol"] for row in shim_records)
    assert not any("legacy_get_fill_color" in row["symbol"] for row in shim_records)
