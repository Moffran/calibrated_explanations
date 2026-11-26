"""Regression test: active guidance and normal workflows never mention ``ce_agent_utils``.

``calibrated_explanations.ce_agent_utils`` is deprecated since v1.1.0 and is
removed in v1.3.0 (ADR-011). Active agent instructions, skills, user
documentation, notebooks, examples and CE source code must not reference it at
all, not even to discourage its use. The only permitted references are formal
deprecation and migration records, version-specific historical records, and the
temporary compatibility infrastructure listed in ``ALLOWED`` below; each entry
goes away with the module.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NEEDLE = "ce_agent_utils"

SCANNED_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "CONTRIBUTOR_INSTRUCTIONS.md",
    "CONTRIBUTING.md",
    "PROMPTS.md",
    "QUICK_API.md",
    "README.md",
    ".github/copilot-instructions.md",
]
SCANNED_DIRS = [
    "skills",
    ".agents",
    ".ai",
    ".claude/skills",
    ".github/instructions",
    ".github/prompts",
    "docs",
    "examples",
    "notebooks",
    "scripts",
    "src",
]
SCANNED_SUFFIXES = {".md", ".rst", ".txt", ".py", ".ipynb", ".yaml", ".yml", ".toml", ".json"}
SKIPPED_DIR_NAMES = {"_build", "__pycache__", ".ipynb_checkpoints"}

ALLOWED = {
    # The deprecated module itself (removed in v1.3.0).
    "src/calibrated_explanations/ce_agent_utils.py",
    # Formal ADR-011 deprecation ledger and migration guidance.
    "docs/migration/deprecations.md",
    # Version-specific upgrade record describing v0.11.6 behavior.
    "docs/upgrade/v0.11.6-upgrade-checklist.md",
    # ADR-001 import-graph exception for the module (removed with the module).
    "scripts/quality/check_import_graph.py",
}


def _scanned_paths() -> list[Path]:
    paths = [ROOT / name for name in SCANNED_FILES]
    for directory in SCANNED_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        paths.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix in SCANNED_SUFFIXES
            and not any(
                part in SKIPPED_DIR_NAMES or part.endswith(".egg-info")
                for part in path.relative_to(ROOT).parts
            )
        )
    return [path for path in paths if path.is_file()]


def test_active_guidance_and_workflows_do_not_reference_deprecated_module() -> None:
    # Arrange
    paths = _scanned_paths()

    # Act
    offenders = sorted(
        relative
        for path in paths
        if (relative := path.relative_to(ROOT).as_posix()) not in ALLOWED
        and NEEDLE in path.read_text(encoding="utf-8", errors="replace")
    )

    # Assert
    assert paths, "no files were scanned; check SCANNED_FILES and SCANNED_DIRS"
    assert not offenders, (
        "Remove references to the deprecated ce_agent_utils module; use "
        "WrapCalibratedExplainer and the documented public CE API directly:\n"
        + "\n".join(offenders)
    )


def test_allowed_references_still_exist() -> None:
    # Keeps ALLOWED from silently outliving the references it permits.
    stale = sorted(
        name
        for name in ALLOWED
        if not (ROOT / name).is_file()
        or NEEDLE not in (ROOT / name).read_text(encoding="utf-8", errors="replace")
    )
    assert not stale, "Drop stale ALLOWED entries: " + ", ".join(stale)
