"""Validate shared agent-instruction consistency rules.

This checker enforces deterministic guardrails for instruction drift:
1. no `.coveragerc` usage in agent docs/templates
2. no bare `should_<behavior>_when_<condition>` naming guidance
3. platform overlays must reference canonical `AGENT_INSTRUCTIONS.md`
4. repo-local path references in instruction docs must exist
5. feedback log schema fields must be present
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

REQUIRED_OVERLAYS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("GEMINI.md"),
    Path(".github/copilot-instructions.md"),
)

REQUIRED_DOCS = (
    Path("AGENT_INSTRUCTIONS.md"),
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("GEMINI.md"),
    Path(".github/copilot-instructions.md"),
    Path(".github/pull_request_template.md"),
    Path(".github/copilot-feedback-log.md"),
    Path("docs/get-started/copilot-setup.md"),
    Path("tests/README.md"),
)

FEEDBACK_LOG = Path(".github/copilot-feedback-log.md")

PATH_PREFIXES = (
    ".github/",
    ".claude/",
    ".ai/",
    "docs/",
    "src/",
    "tests/",
    "scripts/",
    "AGENT_INSTRUCTIONS.md",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "PROMPTS.md",
    "README.md",
    "CHANGELOG.md",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "constraints.txt",
    "GOVERNANCE.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
)

BARE_SHOULD_PATTERN = re.compile(r"(?<!test_)should_<behavior>_when_<condition>")
BACKTICK_PATTERN = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)\n]+)\)")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _collect_instruction_files(root: Path) -> list[Path]:
    prompt_files = sorted((root / ".github/prompts").glob("*.prompt.md"))
    instruction_files = sorted((root / ".github/instructions").glob("*.instructions.md"))
    files = [root / rel for rel in REQUIRED_DOCS]
    files.extend(prompt_files)
    files.extend(instruction_files)
    return files


def _is_http_reference(reference: str) -> bool:
    lowered = reference.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _normalize_reference(raw: str) -> str:
    reference = raw.strip().strip(".,);:")
    if "#" in reference:
        reference = reference.split("#", maxsplit=1)[0]
    return reference


def _is_path_candidate(reference: str) -> bool:
    if not reference or reference in {".", ".."}:
        return False
    if any(marker in reference for marker in ("<", ">", "|")):
        return False
    if _is_http_reference(reference):
        return False
    if reference.startswith(("mailto:", "#", "{doc}`")):
        return False
    if reference.startswith(("./", "../")):
        return True
    return reference.startswith(PATH_PREFIXES)


def _resolve_reference(source_file: Path, reference: str, root: Path) -> Path:
    if reference.startswith(("./", "../")):
        return (source_file.parent / reference).resolve()
    return (root / reference).resolve()


def _iter_reference_candidates(source_text: str) -> Iterable[str]:
    for match in BACKTICK_PATTERN.finditer(source_text):
        yield match.group(1)
    for match in MARKDOWN_LINK_PATTERN.finditer(source_text):
        yield match.group(1)


def _check_required_files_exist(root: Path) -> list[str]:
    errors: list[str] = []
    for rel_path in REQUIRED_DOCS:
        full_path = root / rel_path
        if not full_path.exists():
            errors.append(f"Missing required instruction file: {rel_path.as_posix()}")
    return errors


def _check_no_coveragerc(files: Iterable[Path], root: Path) -> list[str]:
    errors: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        if ".coveragerc" in _read_text(path):
            errors.append(
                f"{path.relative_to(root).as_posix()}: forbidden reference to .coveragerc "
                "(use pyproject.toml)."
            )
    return errors


def _check_no_bare_should(files: Iterable[Path], root: Path) -> list[str]:
    errors: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        text = _read_text(path)
        if BARE_SHOULD_PATTERN.search(text):
            errors.append(
                f"{path.relative_to(root).as_posix()}: bare should_<behavior>_when_<condition> "
                "found (use test_should_<behavior>_when_<condition>)."
            )
    return errors


def _check_overlay_contract(root: Path) -> list[str]:
    errors: list[str] = []
    for rel_path in REQUIRED_OVERLAYS:
        full_path = root / rel_path
        if not full_path.is_file():
            errors.append(f"Missing platform overlay file: {rel_path.as_posix()}")
            continue
        text = _read_text(full_path)
        lowered = text.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
        if "AGENT_INSTRUCTIONS.md" not in text:
            errors.append(
                f"{rel_path.as_posix()}: missing canonical reference to AGENT_INSTRUCTIONS.md."
            )
        if "single source of truth" not in normalized:
            errors.append(
                f"{rel_path.as_posix()}: missing 'single source of truth' overlay wording."
            )
        if "adds only" not in normalized:
            errors.append(
                f"{rel_path.as_posix()}: missing overlay scope wording ('adds only ...')."
            )
    return errors


def _check_feedback_schema(root: Path) -> list[str]:
    errors: list[str] = []
    full_path = root / FEEDBACK_LOG
    if not full_path.is_file():
        return [f"Missing feedback log file: {FEEDBACK_LOG.as_posix()}"]

    text = _read_text(full_path)
    required_fields = (
        "**Feedback:**",
        "**Root cause:**",
        "**Durable fix:**",
        "**Verification:**",
        "**Status:**",
    )
    for field in required_fields:
        if field not in text:
            errors.append(f"{FEEDBACK_LOG.as_posix()}: missing schema field {field}")
    return errors


def _check_path_references(root: Path, files: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for source_file in files:
        if not source_file.is_file():
            continue
        text = _read_text(source_file)
        for raw_reference in _iter_reference_candidates(text):
            reference = _normalize_reference(raw_reference)
            if not _is_path_candidate(reference):
                continue
            if "*" in reference:
                continue
            resolved = _resolve_reference(source_file, reference, root)
            if not resolved.exists():
                errors.append(
                    f"{source_file.relative_to(root).as_posix()}: referenced path does not exist -> {reference}"
                )
    return errors


def run_checks(root: Path) -> list[str]:
    errors: list[str] = []
    files = _collect_instruction_files(root)

    errors.extend(_check_required_files_exist(root))
    errors.extend(_check_no_coveragerc(files, root))
    errors.extend(_check_no_bare_should(files, root))
    errors.extend(_check_overlay_contract(root))
    errors.extend(_check_feedback_schema(root))
    errors.extend(_check_path_references(root, files))

    # Stable output ordering for deterministic CI logs.
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check agent instruction consistency.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root path (defaults to current working directory).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    errors = run_checks(root)
    if errors:
        print("Agent instruction consistency check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Agent instruction consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
