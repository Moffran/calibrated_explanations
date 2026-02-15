from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/anti-pattern-analysis/detect_test_anti_patterns.py")


def run_detector(tmp_path: Path, tests_content: str, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_sample.py").write_text(tests_content, encoding="utf-8")
    report_csv = tmp_path / "report.csv"
    report_json = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--tests-dir",
        str(tests_dir),
        "--output",
        str(report_csv),
        "--report",
        str(report_json),
        "--baseline",
        str(baseline),
        *extra_args,
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_detector_flags_new_assertion_and_determinism_patterns(tmp_path: Path) -> None:
    content = """
import random
import time
import pytest

def test_no_assertion():
    value = 1

def test_raises_is_assertion():
    with pytest.raises(ValueError):
        raise ValueError("boom")

def test_random_unseeded():
    random.random()
    assert True

def test_time_without_patch():
    time.time()
    assert True
"""
    result = run_detector(tmp_path, content, [])
    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    patterns = {entry["pattern"] for entry in report["findings"]}
    assert "test without assertion" in patterns
    assert "random usage without explicit seeding" in patterns
    assert "time/network usage without patching" in patterns
    assert "pytest.raises(FrozenInstanceError)" not in patterns


def test_detector_check_mode_enforces_no_new_violations(tmp_path: Path) -> None:
    initial = """
def test_no_assertion():
    value = 1
"""
    first = run_detector(tmp_path, initial, ["--rebaseline"])
    assert first.returncode == 0

    second = run_detector(tmp_path, initial, ["--check"])
    assert second.returncode == 0

    updated = """
import random

def test_no_assertion():
    value = 1

def test_random_unseeded():
    random.random()
    assert True
"""
    third = run_detector(tmp_path, updated, ["--check"])
    assert third.returncode == 1
    assert "New violations versus baseline" in third.stdout


def test_detector_flags_excessive_mocking_without_outcome_assertions(tmp_path: Path) -> None:
    content = """
from unittest.mock import patch

def test_over_mocked_only_mock_assertions():
    with patch("os.getcwd") as getcwd:
        pass
    with patch("os.listdir") as listdir:
        pass
    with patch("os.path.exists") as exists:
        pass

    getcwd.assert_not_called()
    listdir.assert_not_called()
    exists.assert_not_called()
"""
    result = run_detector(tmp_path, content, [])
    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    patterns = [entry["pattern"] for entry in report["findings"]]
    assert "excessive mocking without outcome assertions" in patterns


def test_detector_does_not_flag_excessive_mocking_when_outcome_asserted(tmp_path: Path) -> None:
    content = """
from unittest.mock import patch

def test_over_mocked_with_outcome_assertion():
    with patch("os.getcwd") as getcwd:
        pass
    with patch("os.listdir") as listdir:
        pass
    with patch("os.path.exists") as exists:
        pass

    getcwd.assert_not_called()
    listdir.assert_not_called()
    exists.assert_not_called()
    assert True
"""
    result = run_detector(tmp_path, content, [])
    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    patterns = [entry["pattern"] for entry in report["findings"]]
    assert "excessive mocking without outcome assertions" not in patterns
