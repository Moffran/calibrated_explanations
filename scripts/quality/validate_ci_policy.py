"""Validate CI workflow changes against ADR-035 governance rules."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

APPROVED_REUSABLES = (
    "./.github/workflows/reusable-python-test.yml",
    "./.github/workflows/reusable-run-make.yml",
    "./.github/workflows/reusable-build-docs.yml",
)
HEAVY_KEYWORDS = ("parity", "perf", "notebook-audit", "docs")
REQUIRED_CHECKLIST_ITEMS = (
    "I used one of the approved reusable workflows OR marked this workflow `experimental: true`",
    "Pip installs in workflow use `-c constraints.txt`",
    "Job `permissions` default to `contents: read`",
    "Heavy job(s) are path-gated, scheduled, or `workflow_dispatch` present",
    "`scripts/local_checks.py` / `Makefile` updated for local reproduction",
)
REPORT_PATH_GUARD_SCRIPT = "scripts/quality/check_no_local_paths_in_reports.py"
REPORT_PATH_GUARD_REPORT = "reports/quality/no_local_paths_report.json"


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]


def _run_git(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, text=True, check=True, capture_output=True)
    return result.stdout.strip()


def _changed_files(base_sha: str, head_sha: str, repo_root: Path) -> list[Path]:
    output = _run_git(["diff", "--name-only", f"{base_sha}..{head_sha}"], repo_root)
    return [Path(line) for line in output.splitlines() if line]


def _diff_is_metadata_only(base_sha: str, head_sha: str, file_path: Path, repo_root: Path) -> bool:
    diff = _run_git(["diff", "--unified=0", f"{base_sha}..{head_sha}", "--", file_path.as_posix()], repo_root)
    changed_lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            changed_lines.append(line[1:].strip())

    if not changed_lines:
        return True

    allowed_patterns = (r"^$", r"^#", r"^name:\s*", r"^run-name:\s*")
    return all(any(re.match(pattern, line) for pattern in allowed_patterns) for line in changed_lines)


def _is_experimental(file_path: Path, text: str) -> bool:
    return "experimental" in file_path.parts or "experimental: true" in text


def _check_reusables(file_path: Path, text: str, errors: list[str]) -> None:
    if file_path.name.startswith("reusable-") or file_path.name == "ci-policy.yml":
        return
    if _is_experimental(file_path, text):
        return
    if "jobs:" not in text:
        return
    if not any(f"uses: {reusable}" in text for reusable in APPROVED_REUSABLES):
        errors.append(
            f"{file_path.as_posix()}: must call an approved reusable workflow or be explicitly experimental."
        )


def _check_permissions(file_path: Path, text: str, errors: list[str]) -> None:
    if "permissions:" not in text:
        errors.append(f"{file_path.as_posix()}: permissions block is required (default contents: read).")
    if "contents: read" not in text:
        errors.append(f"{file_path.as_posix()}: contents: read must be present for least privilege.")
    if file_path.name != "maintenance.yml" and re.search(r"\b\w+:\s*write\b", text):
        errors.append(f"{file_path.as_posix()}: write permissions only allowed in maintenance.yml.")


def _check_pip_constraints(file_path: Path, text: str, errors: list[str]) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if "pip install" in stripped and "-c constraints.txt" not in stripped:
            errors.append(f"{file_path.as_posix()}: pip install must include -c constraints.txt -> '{stripped}'.")


def _check_heavy_gating(file_path: Path, text: str, errors: list[str]) -> None:
    lowered = (file_path.as_posix() + "\n" + text).lower()
    if not any(keyword in lowered for keyword in HEAVY_KEYWORDS):
        return
    has_manual_or_schedule = "workflow_dispatch:" in text or "schedule:" in text
    has_paths = "paths:" in text or "paths-ignore:" in text
    if not has_manual_or_schedule and not has_paths:
        errors.append(f"{file_path.as_posix()}: heavy workflow requires schedule/workflow_dispatch or path filters.")


def _check_report_path_guard_sync(
    changed_files: list[Path],
    repo_root: Path,
    errors: list[str],
) -> None:
    """Require local reproduction updates when the report-path guard changes in CI."""
    workflow_files = [
        path
        for path in changed_files
        if path.as_posix().startswith(".github/workflows/") and path.suffix in {".yml", ".yaml"}
    ]
    if not workflow_files:
        return

    guard_changed = False
    for rel_path in workflow_files:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        text = abs_path.read_text(encoding="utf-8")
        if REPORT_PATH_GUARD_SCRIPT in text or REPORT_PATH_GUARD_REPORT in text:
            guard_changed = True
            break

    if not guard_changed:
        return

    local_checks = repo_root / "scripts/local_checks.py"
    makefile = repo_root / "Makefile"
    local_text = local_checks.read_text(encoding="utf-8") if local_checks.exists() else ""
    makefile_text = makefile.read_text(encoding="utf-8") if makefile.exists() else ""

    if REPORT_PATH_GUARD_SCRIPT not in local_text or REPORT_PATH_GUARD_REPORT not in local_text:
        errors.append(
            "Workflow changes that add/remove the no-local-path report guard must update scripts/local_checks.py."
        )
    if REPORT_PATH_GUARD_SCRIPT not in makefile_text:
        errors.append(
            "Workflow changes that add/remove the no-local-path report guard must update Makefile."
        )


def _load_event(event_path: Path | None) -> dict:
    if event_path is None or not event_path.is_file():
        return {}
    return json.loads(event_path.read_text(encoding="utf-8"))


def _check_pr_metadata(changed_files: list[Path], event_payload: dict, errors: list[str], warnings: list[str]) -> None:
    if not any(path.as_posix().startswith(".github/workflows/") or path.as_posix() == "scripts/local_checks.py" for path in changed_files):
        return

    pr_payload = event_payload.get("pull_request")
    if not isinstance(pr_payload, dict):
        warnings.append("PR metadata checks skipped (pull_request payload unavailable).")
        return

    body = str(pr_payload.get("body") or "")
    for item in REQUIRED_CHECKLIST_ITEMS:
        if item not in body:
            errors.append(f"PR body missing required CI checklist item text: '{item}'.")

    labels = {label.get("name", "") for label in pr_payload.get("labels", []) if isinstance(label, dict)}
    if not labels.intersection({"ci:workflow", "ci:cleanup", "ci-experimental"}):
        errors.append("PR must include one of labels: ci:workflow, ci:cleanup, ci-experimental.")


def validate_policy(base_sha: str, head_sha: str, repo_root: Path, event_path: Path | None = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    changed_files = _changed_files(base_sha, head_sha, repo_root)
    workflow_files = [p for p in changed_files if p.as_posix().startswith(".github/workflows/") and p.suffix in {".yml", ".yaml"}]

    if not workflow_files and Path("scripts/local_checks.py") not in changed_files:
        return ValidationResult(errors=[], warnings=["No CI-governed files changed; policy checks skipped."])

    strict_workflow_change_detected = False

    for rel_path in workflow_files:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        if _diff_is_metadata_only(base_sha, head_sha, rel_path, repo_root):
            warnings.append(f"{rel_path.as_posix()}: metadata-only diff detected; strict workflow checks skipped.")
            continue

        strict_workflow_change_detected = True
        text = abs_path.read_text(encoding="utf-8")
        _check_reusables(rel_path, text, errors)
        _check_permissions(rel_path, text, errors)
        _check_pip_constraints(rel_path, text, errors)
        _check_heavy_gating(rel_path, text, errors)

    changed = {p.as_posix() for p in changed_files}
    if strict_workflow_change_detected and "scripts/local_checks.py" not in changed:
        errors.append("CI workflow files changed without updating scripts/local_checks.py.")
    if strict_workflow_change_detected and "Makefile" not in changed:
        errors.append("CI workflow files changed without updating Makefile.")
    _check_report_path_guard_sync(changed_files, repo_root, errors)

    _check_pr_metadata(changed_files, _load_event(event_path), errors, warnings)
    return ValidationResult(errors=errors, warnings=warnings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--event-path", default=None)
    parser.add_argument("--advisory", action="store_true")
    args = parser.parse_args()

    result = validate_policy(
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        repo_root=Path(args.repo_root).resolve(),
        event_path=Path(args.event_path).resolve() if args.event_path else None,
    )

    for warning in result.warnings:
        print(f"[ci-policy][warning] {warning}")
    for error in result.errors:
        print(f"[ci-policy][error] {error}")

    if result.errors and not args.advisory:
        print("[ci-policy] Failed. Remediation: use approved reusables, enforce constraints, and complete PR checklist.")
        return 1
    if result.errors and args.advisory:
        print("[ci-policy] Advisory mode: violations detected but not blocking.")
    else:
        print("[ci-policy] Passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
