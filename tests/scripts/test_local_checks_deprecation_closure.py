"""Tests for the deprecation-closure lane and the ADR-011 deprecation ledger gate."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import scripts.local_checks as local_checks


def write_deprecation_ledger(tmp_path: Path, active_row: str = "") -> None:
    """Write a minimal deprecation ledger for parser tests."""
    path = tmp_path / "docs" / "migration" / "deprecations.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "# Deprecation & Migration Guide",
                "",
                "### Active deprecations",
                "",
                "| Deprecated symbol | Replacement | Deprecated since | Removal ETA | Notes |",
                "|---|---|---:|---:|---|",
                active_row,
                "",
                "### Removed deprecations (history)",
                "",
                "| Deprecated symbol | Replacement | Deprecated since | Removed in | Notes |",
                "|---|---|---:|---:|---|",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_should_define_deprecation_closure_steps_in_expected_order() -> None:
    """The lane should encode the focused deprecation-closure sequence."""
    steps = local_checks.deprecation_closure_steps()

    assert len(steps) == 2
    assert steps[0].name == "Focused deprecation closure tests"
    assert steps[1].name == "ADR-030 ratification lane"
    assert steps[0].command[0] == local_checks.sys.executable
    assert steps[0].command[1] == "-m"
    assert steps[0].command[2] == "pytest"
    assert steps[0].command[3] == "tests/"
    assert steps[1].command[-1] == "--adr030-ratification"


def test_should_write_reports_when_deprecation_closure_passes(monkeypatch, tmp_path: Path) -> None:
    """The deprecation-closure lane should emit ledger and timing artifacts."""
    monotonic_values = count(100)
    calls: list[str] = []
    write_deprecation_ledger(tmp_path)

    def fake_run_step(step: local_checks.Step) -> int:
        calls.append(step.name)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-05-14T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    rc = local_checks.run_deprecation_closure()

    ledger = json.loads(
        Path("reports/deprecations/active_deprecations_check.json").read_text(encoding="utf-8")
    )
    timing = json.loads(
        Path("reports/deprecations/deprecation_closure_timing.json").read_text(encoding="utf-8")
    )
    assert rc == 0
    assert ledger["status"] == "pass"
    assert ledger["active_rows_count"] == 0
    assert ledger["blocking_rows_count"] == 0
    assert calls == [
        "Focused deprecation closure tests",
        "ADR-030 ratification lane",
    ]
    assert timing["schema_version"] == 1
    assert timing["generated_at"] == "2026-05-14T00:00:00+00:00"
    assert [step["exit_code"] for step in timing["steps"]] == [0, 0, 0]


def test_should_report_active_deprecation_rows_as_blocking_when_eta_targets_v1_0_0(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The lane should stop at the ledger gate when Active rows remain."""
    write_deprecation_ledger(
        tmp_path,
        "| `old_api()` | `new_api()` | v0.1.0 | v1.0.0 | Still active. |",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        local_checks,
        "_run_step",
        lambda step: (_ for _ in ()).throw(AssertionError("should not run commands")),
    )
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-05-14T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: 1.0)

    rc = local_checks.run_deprecation_closure()

    ledger = json.loads(
        Path("reports/deprecations/active_deprecations_check.json").read_text(encoding="utf-8")
    )
    assert rc == 1
    assert ledger["status"] == "fail"
    assert ledger["active_rows_count"] == 1
    assert ledger["blocking_rows_count"] == 1
    assert ledger["blocking_symbols"] == ["`old_api()`"]


def _run_ledger_gate(monkeypatch, tmp_path: Path, row: str, current: str) -> dict:
    write_deprecation_ledger(tmp_path, row)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{current}"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-11-23T00:00:00+00:00")
    rc = local_checks.run_deprecation_ledger_gate()
    ledger = json.loads(
        Path("reports/deprecations/active_deprecations_check.json").read_text(encoding="utf-8")
    )
    ledger["rc"] = rc
    return ledger


def test_should_pass_post_v1_deprecation_with_two_minor_removal_eta(
    monkeypatch, tmp_path: Path
) -> None:
    """ADR-011 default policy: an active row with a future two-minor ETA does not block."""
    ledger = _run_ledger_gate(
        monkeypatch, tmp_path, "| `old()` | `new()` | v1.1.0 | v1.3.0 | Active. |", "1.1.0-dev"
    )

    assert ledger["rc"] == 0
    assert ledger["status"] == "pass"
    assert ledger["active_symbols"] == ["`old()`"]
    assert ledger["blocking_rows_count"] == 0


def test_should_block_removal_eta_inside_two_minor_window(monkeypatch, tmp_path: Path) -> None:
    ledger = _run_ledger_gate(
        monkeypatch, tmp_path, "| `old()` | `new()` | v1.1.0 | v1.2.0 | Too soon. |", "1.1.0-dev"
    )

    assert ledger["rc"] == 1
    assert ledger["blocking_symbols"] == ["`old()`"]


def test_should_block_deprecation_once_its_removal_eta_is_reached(
    monkeypatch, tmp_path: Path
) -> None:
    ledger = _run_ledger_gate(
        monkeypatch, tmp_path, "| `old()` | `new()` | v1.1.0 | v1.3.0 | Overdue. |", "1.3.0-dev"
    )

    assert ledger["rc"] == 1
    assert "reached" in ledger["blocking_reasons"]["`old()`"]
