"""Regression tests ensuring agent-facing docs do not recommend ce_agent_utils.

These tests fail if any agent-facing or CE-first guide starts recommending
calibrated_explanations.ce_agent_utils as the preferred agent interface.

Allowed mentions are restricted to:
- The source file itself (src/calibrated_explanations/ce_agent_utils.py)
- Backward-compatibility tests (tests/unit/test_ce_agent_utils.py)
- Changelog / release notes (CHANGELOG.md)
- The explicit legacy-compatibility notes added to the guide (which must contain
  the words "legacy" or "backward" in the same line or nearby context)

Files checked here are exclusively agent-facing instructions and the CE-First
guide that agents are directed to read.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

AGENT_FACING_DOCS = [
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
    ROOT / "GEMINI.md",
    ROOT / "CONTRIBUTOR_INSTRUCTIONS.md",
    ROOT / "docs" / "get-started" / "ce_first_agent_guide.md",
    ROOT / "docs" / "get-started" / "copilot-setup.md",
]

FORBIDDEN_RECOMMENDATION_PATTERNS = [
    r"Use helpers from `src/calibrated_explanations/ce_agent_utils\.py`",
    r"Use `src/calibrated_explanations/ce_agent_utils\.py` helpers",
    r"Use WrapCalibratedExplainer and ce_agent_utils helpers",
    r"Register `calibrated_explanations\.ce_agent_utils` as a canonical",
    r"canonical helper module",
    r"validated helpers? from.*ce_agent_utils",
    r"CE-First runtime helpers",
    r"pipeline helpers? for agents",
    r"one-line CE-first flow",
    r"agent helper",
    r"Prefer these helpers over ad-hoc code",
]

_ALLOWED_CONTEXT_WORDS = {
    "legacy",
    "backward",
    "backward-compat",
    "retained",
    "not the recommended",
    "do not",
    "must not",
    "agents must not",
    "not use",
    "not recommend",
}


def line_is_in_allowed_context(line: str) -> bool:
    lower = line.lower()
    return any(word in lower for word in _ALLOWED_CONTEXT_WORDS)


def test_no_forbidden_recommendation_patterns_in_agent_docs() -> None:
    violations: list[str] = []
    for doc_path in AGENT_FACING_DOCS:
        if not doc_path.exists():
            continue
        for lineno, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in FORBIDDEN_RECOMMENDATION_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE) and not line_is_in_allowed_context(line):
                    violations.append(
                        f"{doc_path.relative_to(ROOT)}:{lineno}: "
                        f"forbidden pattern «{pattern}» found in: {line.strip()}"
                    )
    assert not violations, (
        "Agent-facing docs must not recommend ce_agent_utils usage.\n" + "\n".join(violations)
    )


def test_no_import_from_ce_agent_utils_in_agent_guide() -> None:
    guide = ROOT / "docs" / "get-started" / "ce_first_agent_guide.md"
    if not guide.exists():
        return
    text = guide.read_text(encoding="utf-8")
    lines = text.splitlines()
    violations: list[str] = []
    for lineno, line in enumerate(lines, 1):
        if "from calibrated_explanations.ce_agent_utils import" in line:
            # Allow only on lines that are inside a clearly-labelled legacy section
            # (the line itself or the preceding 5 lines contain "legacy" or "backward")
            context_lines = lines[max(0, lineno - 6) : lineno]
            context_text = " ".join(context_lines).lower()
            if not any(w in context_text for w in _ALLOWED_CONTEXT_WORDS):
                violations.append(f"ce_first_agent_guide.md:{lineno}: {line.strip()}")
    assert not violations, (
        "ce_first_agent_guide.md must not import from ce_agent_utils outside "
        "explicitly labelled legacy-compatibility sections.\n" + "\n".join(violations)
    )
