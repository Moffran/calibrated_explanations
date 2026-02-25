from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/quality/check_agent_instruction_consistency.py")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_valid_instruction_tree(root: Path) -> None:
    write(
        root / "CONTRIBUTOR_INSTRUCTIONS.md",
        """
# Agent Instructions

- Naming: `test_should_<behavior>_when_<condition>`.
- Coverage: `pytest --cov-config=pyproject.toml`.
""".strip(),
    )
    write(
        root / "AGENTS.md",
        """
> canonical `CONTRIBUTOR_INSTRUCTIONS.md` is the single source of truth.
> this file adds only codex-specific context.
""".strip(),
    )
    write(
        root / "CLAUDE.md",
        """
> canonical `CONTRIBUTOR_INSTRUCTIONS.md` is the single source of truth.
> this file adds only claude-specific context.
""".strip(),
    )
    write(
        root / "GEMINI.md",
        """
> canonical `CONTRIBUTOR_INSTRUCTIONS.md` is the single source of truth.
> this file adds only gemini-specific context.
""".strip(),
    )
    write(
        root / ".github/copilot-instructions.md",
        """
> canonical `CONTRIBUTOR_INSTRUCTIONS.md` is the single source of truth.
> this file adds only github copilot-specific context.
""".strip(),
    )
    write(
        root / ".github/pull_request_template.md",
        "- [ ] Coverage gate uses `--cov-config=pyproject.toml`.",
    )
    write(
        root / ".github/copilot-feedback-log.md",
        """
## YYYY-MM-DD – Example
**Feedback:** bad guidance
**Root cause:** stale instruction
**Durable fix:** CONTRIBUTOR_INSTRUCTIONS.md
**Verification:** python scripts/quality/check_agent_instruction_consistency.py
**Status:** ✅ incorporated
""".strip(),
    )
    write(
        root / "docs/get-started/copilot-setup.md",
        """
See `CONTRIBUTOR_INSTRUCTIONS.md`, `AGENTS.md`, and `tests/README.md`.
""".strip(),
    )
    write(
        root / "tests/README.md",
        """
Naming: `test_should_<behavior>_when_<condition>`.
Coverage: `--cov-config=pyproject.toml`.
""".strip(),
    )


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT_PATH), "--root", str(root)]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_checker_passes_on_valid_instruction_tree(tmp_path: Path) -> None:
    create_valid_instruction_tree(tmp_path)
    result = run_checker(tmp_path)
    assert result.returncode == 0
    assert "passed" in result.stdout.lower()


def test_checker_fails_on_coveragerc_and_bare_should(tmp_path: Path) -> None:
    create_valid_instruction_tree(tmp_path)
    write(
        tmp_path / "CONTRIBUTOR_INSTRUCTIONS.md",
        "Coverage: `--cov-config=.coveragerc`",
    )
    write(
        tmp_path / "tests/README.md",
        "Naming: `should_<behavior>_when_<condition>`.",
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert ".coveragerc" in result.stdout
    assert "bare should_<behavior>_when_<condition>" in result.stdout


def test_checker_fails_when_overlay_misses_canonical_reference(tmp_path: Path) -> None:
    create_valid_instruction_tree(tmp_path)
    write(
        tmp_path / "CLAUDE.md",
        "This file adds only claude-specific context without canonical reference.",
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "CLAUDE.md: missing canonical reference" in result.stdout


def test_checker_fails_when_feedback_schema_is_incomplete(tmp_path: Path) -> None:
    create_valid_instruction_tree(tmp_path)
    write(
        tmp_path / ".github/copilot-feedback-log.md",
        """
## YYYY-MM-DD – Missing fields
**Feedback:** something
**Status:** open
""".strip(),
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "missing schema field **Root cause:**" in result.stdout
    assert "missing schema field **Durable fix:**" in result.stdout
    assert "missing schema field **Verification:**" in result.stdout


def test_checker_fails_when_instruction_path_reference_is_missing(tmp_path: Path) -> None:
    create_valid_instruction_tree(tmp_path)
    write(
        tmp_path / "AGENTS.md",
        """
> canonical `CONTRIBUTOR_INSTRUCTIONS.md` is the single source of truth.
> this file adds only codex-specific context.
Missing path: `docs/missing.md`.
""".strip(),
    )
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "referenced path does not exist -> docs/missing.md" in result.stdout
