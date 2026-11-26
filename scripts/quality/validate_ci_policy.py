"""Validate the complete CI workflow inventory against ADR-035 v1 rules.

v0.11.6 Task 60 replaced the earlier diff-based, advisory validator with a
blocking full-inventory validator. The same command is executed by three
layers so they cannot drift apart:

1. the always-run ``policy`` job in ``.github/workflows/ci.yml``,
2. the ``local-checks-pr`` profile (``scripts/local_checks.py``),
3. the ``.github/``-scoped pre-commit hook (``.pre-commit-config.yaml``).

The validator is intentionally stdlib-only (regex/text based) so the CI
policy job can run before any dependency installation.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: The only top-level workflows permitted under ``.github/workflows/``.
#: Expanding this set is a deliberate governance decision (ADR-035): update
#: the set here together with ADR-035 and the release plan, never ad hoc.
APPROVED_WORKFLOWS = frozenset({"ci.yml", "scheduled.yml", "maintenance.yml"})

#: The only composite actions permitted under ``.github/actions/``.
APPROVED_ACTIONS = frozenset({"setup-ce-python"})

#: The single workflow allowed to request write scopes.
MAINTENANCE_WORKFLOW = "maintenance.yml"

#: The identical command every inventory-guard layer must execute.
FULL_INVENTORY_COMMAND = "python scripts/quality/validate_ci_policy.py --full-inventory"

FULL_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
USES_PATTERN = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>[^\s#]+)")
WRITE_SCOPE_PATTERN = re.compile(r"^\s*[\w-]+:\s*write\b", re.MULTILINE)
RUNS_ON_PATTERN = re.compile(r"^\s*runs-on:", re.MULTILINE)
TIMEOUT_PATTERN = re.compile(r"^\s*timeout-minutes:", re.MULTILINE)
WORKFLOW_NAME_PATTERN = re.compile(r"^name:\s*(?P<name>.+?)\s*$", re.MULTILINE)
MAKE_PROFILE_PATTERN = re.compile(r"\bmake\s+local-checks-([\w-]+)")
SCRIPT_PROFILE_PATTERN = re.compile(r"local_checks\.py\s+--profile\s+([\w-]+)")
LOCAL_CHECKS_WIRING_PATTERN = re.compile(
    r"scripts/quality/validate_ci_policy\.py.{0,80}--full-inventory", re.DOTALL
)

#: References that must never appear in OSS CI (private skill repositories,
#: automated skill synchronization). Regression guard for v0.11.6 Task 60
#: step 2 (`sync-skills.yml` removal).
SKILL_SYNC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"generic-skill-library"), "reference to private skill repository"),
    (re.compile(r"\.github/skills/"), "automated write to agent skill directory"),
    (re.compile(r"\.codex/skills/"), "automated write to agent skill directory"),
    (re.compile(r"\.claude/skills/"), "automated write to agent skill directory"),
    (re.compile(r"\.agents/skills/"), "automated write to agent skill directory"),
    (re.compile(r"(?<![\w./-])skills/"), "automated write to canonical skill directory"),
)

#: Publishing/deployment mechanics forbidden on this development mirror.
PUBLISH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"twine\s+upload"), "package publication"),
    (re.compile(r"pypi-publish"), "package publication"),
    (re.compile(r"\bgh\s+release\b"), "GitHub release creation"),
    (re.compile(r"\bgit\s+push\b"), "direct git push"),
    (re.compile(r"\bgit\s+tag\b"), "tag creation"),
    (re.compile(r"deploy-pages"), "documentation deployment"),
    (re.compile(r"gh-pages"), "documentation deployment"),
    (re.compile(r"softprops/action-gh-release"), "GitHub release creation"),
)

#: The only workflow allowed to use the PR-opening write mechanism.
WRITE_MECHANISM_PATTERN = re.compile(r"peter-evans/create-pull-request")


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]


def _workflow_files(repo_root: Path) -> list[Path]:
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return []
    return sorted(path for path in workflows_dir.iterdir() if path.suffix in {".yml", ".yaml"})


def _action_files(repo_root: Path) -> list[Path]:
    actions_dir = repo_root / ".github" / "actions"
    if not actions_dir.is_dir():
        return []
    return sorted(actions_dir.glob("*/*.yml")) + sorted(actions_dir.glob("*/*.yaml"))


def _rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _check_inventory(repo_root: Path, errors: list[str]) -> None:
    workflow_names = {path.name for path in _workflow_files(repo_root)}
    for unexpected in sorted(workflow_names - APPROVED_WORKFLOWS):
        errors.append(
            f".github/workflows/{unexpected}: not in the approved inventory "
            f"{sorted(APPROVED_WORKFLOWS)}. Adding a workflow is a deliberate "
            "governance decision: update APPROVED_WORKFLOWS in "
            "scripts/quality/validate_ci_policy.py together with ADR-035."
        )
    for missing in sorted(APPROVED_WORKFLOWS - workflow_names):
        errors.append(f".github/workflows/{missing}: approved workflow is missing.")

    actions_dir = repo_root / ".github" / "actions"
    if actions_dir.is_dir():
        action_names = {path.name for path in actions_dir.iterdir() if path.is_dir()}
        for unexpected in sorted(action_names - APPROVED_ACTIONS):
            errors.append(
                f".github/actions/{unexpected}: not in the approved composite-action "
                f"inventory {sorted(APPROVED_ACTIONS)}."
            )


def _check_responsibility_header(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    head = "\n".join(text.splitlines()[:10])
    if "# Responsibility:" not in head:
        errors.append(
            f"{_rel(path, repo_root)}: missing a leading '# Responsibility:' comment "
            "stating the workflow's single responsibility."
        )


def _check_permissions(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    if not re.search(r"^permissions:", text, re.MULTILINE):
        errors.append(
            f"{_rel(path, repo_root)}: top-level permissions block is required "
            "(default contents: read)."
        )
    if "contents: read" not in text:
        errors.append(
            f"{_rel(path, repo_root)}: contents: read must be present for least privilege."
        )
    if path.name != MAINTENANCE_WORKFLOW:
        for line in text.splitlines():
            if re.match(r"^\s*[\w-]+:\s*write\b", line):
                errors.append(
                    f"{_rel(path, repo_root)}: write permissions are only allowed in "
                    f"{MAINTENANCE_WORKFLOW} -> '{line.strip()}'."
                )


def _check_pip_constraints(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"\bpip install\s+--upgrade\s+pip\b", stripped):
            continue
        if "pip install" in stripped and "-c constraints.txt" not in stripped:
            errors.append(
                f"{_rel(path, repo_root)}: pip install must include -c constraints.txt "
                f"-> '{stripped}'."
            )


def _check_action_sha_pins(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    for line_num, line in enumerate(text.splitlines(), start=1):
        match = USES_PATTERN.match(line)
        if not match:
            continue
        ref = match.group("ref")
        if ref.startswith(("./", "../")):
            continue
        if "@" not in ref:
            errors.append(
                f"{_rel(path, repo_root)}:{line_num}: external action '{ref}' must be "
                "pinned to a full commit SHA."
            )
            continue
        owner_repo, version = ref.rsplit("@", 1)
        if not FULL_SHA_PATTERN.fullmatch(version):
            errors.append(
                f"{_rel(path, repo_root)}:{line_num}: external action '{owner_repo}' uses "
                f"'{version}' but must use a full 40-character commit SHA."
            )


def _check_forbidden_references(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    for pattern, description in SKILL_SYNC_PATTERNS + PUBLISH_PATTERNS:
        if pattern.search(text):
            errors.append(
                f"{_rel(path, repo_root)}: forbidden {description} (pattern: {pattern.pattern})."
            )
    if path.name != MAINTENANCE_WORKFLOW and WRITE_MECHANISM_PATTERN.search(text):
        errors.append(
            f"{_rel(path, repo_root)}: PR-opening write mechanism is only allowed in "
            f"{MAINTENANCE_WORKFLOW}."
        )


def _check_reliability(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    if "continue-on-error" in text:
        errors.append(
            f"{_rel(path, repo_root)}: continue-on-error is forbidden; make the check "
            "reliable, classify it advisory-with-artifact, or delete it."
        )
    runs_on_count = len(RUNS_ON_PATTERN.findall(text))
    timeout_count = len(TIMEOUT_PATTERN.findall(text))
    if timeout_count < runs_on_count:
        errors.append(
            f"{_rel(path, repo_root)}: every job needs timeout-minutes "
            f"({runs_on_count} jobs, {timeout_count} timeouts)."
        )


def _check_triggers(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    """Expensive assurance stays scheduled/manual; maintenance stays manual-only."""
    if path.name == "scheduled.yml":
        if "schedule:" not in text:
            errors.append(f"{_rel(path, repo_root)}: scheduled.yml must run on a schedule.")
        if "pull_request:" in text or re.search(r"^\s*push:", text, re.MULTILINE):
            errors.append(f"{_rel(path, repo_root)}: heavy assurance must not trigger on PR/push.")
    if path.name == MAINTENANCE_WORKFLOW:
        if "workflow_dispatch:" not in text:
            errors.append(f"{_rel(path, repo_root)}: maintenance must be workflow_dispatch only.")
        if any(trigger in text for trigger in ("pull_request:", "schedule:")) or re.search(
            r"^\s*push:", text, re.MULTILINE
        ):
            errors.append(f"{_rel(path, repo_root)}: maintenance must be workflow_dispatch only.")


def _check_concurrency(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    if path.name == "ci.yml":
        if "concurrency:" not in text or "cancel-in-progress:" not in text:
            errors.append(
                f"{_rel(path, repo_root)}: ci.yml must cancel superseded PR runs via a "
                "concurrency group."
            )
    if path.name == MAINTENANCE_WORKFLOW and "cancel-in-progress: true" in text:
        errors.append(
            f"{_rel(path, repo_root)}: active maintenance runs must never be cancelled "
            "(cancel-in-progress: true is forbidden here)."
        )


def _check_unique_names(
    workflow_texts: dict[Path, str], repo_root: Path, errors: list[str]
) -> None:
    seen: dict[str, str] = {}
    for path, text in workflow_texts.items():
        match = WORKFLOW_NAME_PATTERN.search(text)
        if match is None:
            errors.append(f"{_rel(path, repo_root)}: workflow must declare a stable name.")
            continue
        name = match.group("name")
        if name in seen:
            errors.append(
                f"{_rel(path, repo_root)}: workflow name '{name}' duplicates {seen[name]}."
            )
        seen[name] = _rel(path, repo_root)

    ci_path = repo_root / ".github" / "workflows" / "ci.yml"
    ci_text = workflow_texts.get(ci_path, "")
    if ci_text:
        if not re.search(r"^name:\s*CI\s*$", ci_text, re.MULTILINE):
            errors.append(
                ".github/workflows/ci.yml: workflow name must be 'CI' so the required "
                "branch-protection check stays 'CI / required'."
            )
        if not re.search(r"^  required:\s*$", ci_text, re.MULTILINE):
            errors.append(
                ".github/workflows/ci.yml: an aggregate 'required' job must exist as the "
                "single stable required PR check."
            )


def _check_guard_wiring(
    repo_root: Path, workflow_texts: dict[Path, str], errors: list[str]
) -> None:
    """All three inventory-guard layers must execute the identical command."""
    ci_path = repo_root / ".github" / "workflows" / "ci.yml"
    ci_text = workflow_texts.get(ci_path, "")
    if FULL_INVENTORY_COMMAND not in ci_text:
        errors.append(
            f".github/workflows/ci.yml: the always-run policy job must execute "
            f"'{FULL_INVENTORY_COMMAND}'."
        )

    pre_commit = repo_root / ".pre-commit-config.yaml"
    pre_commit_text = pre_commit.read_text(encoding="utf-8") if pre_commit.exists() else ""
    if FULL_INVENTORY_COMMAND not in pre_commit_text or "^\\.github/" not in pre_commit_text:
        errors.append(
            ".pre-commit-config.yaml: a hook scoped to files ^\\.github/ must execute "
            f"'{FULL_INVENTORY_COMMAND}'."
        )

    local_checks = repo_root / "scripts" / "local_checks.py"
    local_checks_text = local_checks.read_text(encoding="utf-8") if local_checks.exists() else ""
    if not LOCAL_CHECKS_WIRING_PATTERN.search(local_checks_text):
        errors.append(
            f"scripts/local_checks.py: the PR profile must execute '{FULL_INVENTORY_COMMAND}'."
        )


def _local_profiles(repo_root: Path) -> set[str]:
    local_checks = repo_root / "scripts" / "local_checks.py"
    if not local_checks.exists():
        return set()
    text = local_checks.read_text(encoding="utf-8")
    match = re.search(r"\"--profile\",\s*choices=\(([^)]*)\)", text, re.DOTALL)
    if match is None:
        return set()
    return set(re.findall(r"[\"']([\w-]+)[\"']", match.group(1)))


def _check_profile_mappings(
    repo_root: Path, workflow_texts: dict[Path, str], errors: list[str]
) -> None:
    """Local-profile commands referenced from CI must exist locally."""
    profiles = _local_profiles(repo_root)
    makefile = repo_root / "Makefile"
    makefile_text = makefile.read_text(encoding="utf-8") if makefile.exists() else ""
    for path, text in workflow_texts.items():
        referenced = set(MAKE_PROFILE_PATTERN.findall(text)) | set(
            SCRIPT_PROFILE_PATTERN.findall(text)
        )
        for profile in sorted(referenced):
            if profile not in profiles:
                errors.append(
                    f"{_rel(path, repo_root)}: references local profile '{profile}' that "
                    "scripts/local_checks.py does not define."
                )
        for make_target in sorted(set(MAKE_PROFILE_PATTERN.findall(text))):
            if f"local-checks-{make_target}:" not in makefile_text:
                errors.append(
                    f"{_rel(path, repo_root)}: references 'make local-checks-{make_target}' "
                    "but the Makefile does not define that target."
                )


def validate_full_inventory(repo_root: Path) -> ValidationResult:
    """Validate the complete workflow inventory (blocking)."""
    errors: list[str] = []
    warnings: list[str] = []

    _check_inventory(repo_root, errors)

    workflow_texts: dict[Path, str] = {}
    for path in _workflow_files(repo_root):
        text = path.read_text(encoding="utf-8")
        workflow_texts[path] = text
        _check_responsibility_header(path, text, repo_root, errors)
        _check_permissions(path, text, repo_root, errors)
        _check_pip_constraints(path, text, repo_root, errors)
        _check_action_sha_pins(path, text, repo_root, errors)
        _check_forbidden_references(path, text, repo_root, errors)
        _check_reliability(path, text, repo_root, errors)
        _check_triggers(path, text, repo_root, errors)
        _check_concurrency(path, text, repo_root, errors)

    for path in _action_files(repo_root):
        text = path.read_text(encoding="utf-8")
        _check_pip_constraints(path, text, repo_root, errors)
        _check_action_sha_pins(path, text, repo_root, errors)
        _check_forbidden_references(path, text, repo_root, errors)

    _check_unique_names(workflow_texts, repo_root, errors)
    _check_guard_wiring(repo_root, workflow_texts, errors)
    _check_profile_mappings(repo_root, workflow_texts, errors)

    return ValidationResult(errors=errors, warnings=warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-inventory",
        action="store_true",
        help="Validate the complete workflow inventory (the only supported mode).",
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    if not args.full_inventory:
        print(
            "[ci-policy] The diff-based mode was removed by v0.11.6 Task 60; "
            f"run '{FULL_INVENTORY_COMMAND}'."
        )
        return 2

    result = validate_full_inventory(Path(args.repo_root).resolve())

    for warning in result.warnings:
        print(f"[ci-policy][warning] {warning}")
    for error in result.errors:
        print(f"[ci-policy][error] {error}")

    if result.errors:
        print(
            "[ci-policy] Failed. The workflow inventory violates ADR-035; see errors "
            "above. Inventory changes require a deliberate APPROVED_WORKFLOWS update."
        )
        return 1
    print("[ci-policy] Passed (full inventory).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
