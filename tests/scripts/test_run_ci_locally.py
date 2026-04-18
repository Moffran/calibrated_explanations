from __future__ import annotations

from scripts.run_ci_locally import summary_cwd


def test_summary_cwd_returns_relative_path() -> None:
    assert summary_cwd(".") == "."


def test_summary_cwd_does_not_force_absolute_path_for_external_value() -> None:
    assert summary_cwd("reports") == "reports"
