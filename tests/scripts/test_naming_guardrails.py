"""Regression tests for Standard-001 naming guardrails."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ruff_available = importlib.util.find_spec("ruff") is not None


@pytest.mark.skipif(not ruff_available, reason="ruff is not installed in this environment")
def test_should_fail_when_naming_violation_exists(tmp_path: Path):
    """Verify Ruff naming checks fail on non-compliant identifiers."""
    bad_file = tmp_path / "bad_naming.py"
    bad_file.write_text(
        "def BadFunction():\n    return 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", "--select", "N", str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "N802" in combined_output


@pytest.mark.skipif(not ruff_available, reason="ruff is not installed in this environment")
def test_should_pass_when_names_are_compliant(tmp_path: Path):
    """Verify Ruff naming checks pass for compliant identifiers."""
    good_file = tmp_path / "good_naming.py"
    good_file.write_text(
        "def good_function():\n    return 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", "--select", "N", str(good_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
