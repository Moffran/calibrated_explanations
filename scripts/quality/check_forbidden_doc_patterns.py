"""Named forbidden-pattern gates for docs, skills, ADRs, and notebooks.

Each named check pairs regex patterns with an include/exclude file scope and
fails when any current-source file still contains the forbidden text. The
checks back the focused ``make local-checks-task TASK=<n>`` mappings for the
v0.11.6 documentation-alignment tasks (12, 14, 15, 17, 18, 19, 29); see
``development/current-work/v0.11.6_plan.md`` and
``development/current-work/pre-v2-gaps.md``.

Historical records (CHANGELOG, migration ledger, finished-work plans) are
excluded per check so the gates target current guidance only.

Usage
-----
    python scripts/quality/check_forbidden_doc_patterns.py --list-checks
    python scripts/quality/check_forbidden_doc_patterns.py --check narrative-format
    python scripts/quality/check_forbidden_doc_patterns.py --all-checks
    python scripts/quality/check_forbidden_doc_patterns.py --check guarded-stale-wording --check template-json-references
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_COMMON_EXCLUDES = (
    "docs/_build/**",
    "reports/**",
    "development/current-work/**",
    "development/finished-work/**",
    "CHANGELOG.md",
)


@dataclass(frozen=True)
class PatternCheck:
    """One named forbidden-pattern gate."""

    name: str
    description: str
    patterns: tuple[str, ...]
    includes: tuple[str, ...]
    excludes: tuple[str, ...] = field(default=_COMMON_EXCLUDES)


CHECKS: dict[str, PatternCheck] = {
    check.name: check
    for check in (
        PatternCheck(
            name="narrative-format",
            description=(
                "Current docs/skills must not teach to_narrative(format=...); "
                "the runtime rejects it (v0.11.6 Task 12)."
            ),
            patterns=(r"to_narrative\(\s*format\s*=",),
            includes=(
                "*.md",
                "docs/**/*.md",
                ".github/**/*.md",
                "skills/**/*.md",
                "notebooks/**/*.ipynb",
            ),
        ),
        PatternCheck(
            name="guarded-stale-wording",
            description=(
                "Current docs/skills/ADRs must not describe guarded=True as "
                "deprecated/current; it was removed in v0.11.5 (Task 14)."
            ),
            patterns=(
                r"guarded=True`?\s+(?:kwarg\s+|boolean\s+kwarg\s+)?is\s+deprecated",
                r"deprecated\s+`?guarded=True",
                r"guarded=True`?.{0,40}\(deprecated\)",
            ),
            includes=(
                "docs/**/*.md",
                "skills/**/*.md",
                "development/adrs/**/*.md",
            ),
            excludes=_COMMON_EXCLUDES + ("docs/migration/**",),
        ),
        PatternCheck(
            name="tests-helpers-imports",
            description=(
                "Public docs must not import repository-only tests.helpers modules (Task 15)."
            ),
            patterns=(r"(?:from|import)\s+tests(?:\.|\s+import\b)",),
            includes=("docs/**/*.md",),
        ),
        PatternCheck(
            name="stale-notebook-warnings",
            description=(
                "Notebook outputs must not show the retired v0.11.4 "
                "'forwarded for compatibility' unknown-kwarg warning (Task 17)."
            ),
            patterns=(r"forwarded for compatibility",),
            includes=("notebooks/**/*.ipynb",),
        ),
        PatternCheck(
            name="dead-legacy-contract-path",
            description=(
                "Contributor guidance must not reference the removed "
                "development/current-work legacy API contract path (Task 18)."
            ),
            patterns=(r"current-work/legacy_user_api_contract",),
            includes=("*.md", "docs/**/*.md", ".github/**/*.md"),
        ),
        PatternCheck(
            name="template-json-references",
            description=(
                "Docs and source must reference the packaged "
                "explain_template.yaml, not the non-existent .json (Task 19)."
            ),
            patterns=(r"explain_template\.json",),
            includes=("docs/**/*.md", "src/**/*.py"),
        ),
        PatternCheck(
            name="ledger-active-framing",
            description=(
                "The deprecation ledger's Active section must not claim the "
                "removals happened in 1.0.0 (Task 28)."
            ),
            patterns=(r"removed in 1\.0\.0",),
            includes=("docs/migration/deprecations.md",),
            excludes=(),
        ),
        PatternCheck(
            name="removed-legacy-locations",
            description=(
                "Live skill/agent/docs surfaces must not reference the removed "
                "docs/improvement/ or docs/standards/ directories (Task 59 L4)."
            ),
            patterns=(r"docs/improvement/", r"docs/standards/"),
            includes=(
                "skills/**/*.md",
                ".github/agents/**/*.md",
                "docs/**/*.md",
            ),
            excludes=_COMMON_EXCLUDES + ("docs/foundations/governance/release_notes.md",),
        ),
        PatternCheck(
            name="utf8-mojibake-needle",
            description=(
                "Tracked text files must not contain the U+00E2 U+20AC "
                "mojibake needle or a UTF-8 BOM (Task 35)."
            ),
            patterns=(r"\u00e2\u20ac", r"^\ufeff"),
            includes=(
                "*.md",
                "*.toml",
                "*.yml",
                "*.yaml",
                "docs/**/*.md",
                "development/**/*.md",
                "tests/**/*.py",
                "scripts/**/*.py",
                ".github/**/*.md",
                ".github/**/*.yml",
                ".github/**/*.yaml",
            ),
        ),
    )
}


def _iter_files(check: PatternCheck) -> list[Path]:
    """Return the de-duplicated, sorted file scope for a check."""
    excluded = {
        path.resolve()
        for pattern in check.excludes
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    }
    excluded_dirs = {
        path.resolve()
        for pattern in check.excludes
        for path in REPO_ROOT.glob(pattern.removesuffix("/**"))
        if path.is_dir()
    }

    def _is_excluded(candidate: Path) -> bool:
        resolved = candidate.resolve()
        if resolved in excluded:
            return True
        return any(parent in excluded_dirs for parent in resolved.parents)

    files: set[Path] = set()
    for pattern in check.includes:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and not _is_excluded(path):
                files.add(path)
    return sorted(files)


def run_check(check: PatternCheck) -> list[str]:
    """Return formatted violation lines for one check."""
    compiled = [re.compile(pattern) for pattern in check.patterns]
    violations: list[str] = []
    for path in _iter_files(check):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            violations.append(f"{path}: unreadable ({exc})")
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in compiled:
                if pattern.search(line):
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{lineno}: {line.strip()[:160]}")
                    break
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        choices=sorted(CHECKS),
        help="Named check to run (repeatable).",
    )
    parser.add_argument(
        "--list-checks", action="store_true", help="List available checks and exit."
    )
    parser.add_argument(
        "--all-checks",
        action="store_true",
        help="Run every named check.",
    )
    args = parser.parse_args()

    if args.list_checks:
        for name in sorted(CHECKS):
            print(f"{name}: {CHECKS[name].description}")
        return 0
    if args.all_checks and args.check:
        parser.error("Use either --all-checks or one or more --check values, not both.")

    selected_checks = list(CHECKS) if args.all_checks else args.check
    if not selected_checks:
        parser.error("Provide at least one --check name (or --list-checks).")

    exit_code = 0
    for name in selected_checks:
        check = CHECKS[name]
        violations = run_check(check)
        if violations:
            exit_code = 1
            print(f"FAIL [{name}] {check.description}")
            for violation in violations:
                print(f"  {violation}")
        else:
            print(f"PASS [{name}]")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
