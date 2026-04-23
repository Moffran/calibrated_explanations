from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.quality.validate_ci_policy import _REUSABLE_FIRST_ALLOWLIST, validate_policy


def init_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=tmp_path, check=True)


def commit_all(tmp_path: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=tmp_path, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, text=True, capture_output=True)
    return sha.stdout.strip()


def test_should_fail_when_pip_install_missing_constraints(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-pr.yml").write_text(
        """
name: CI
on: [pull_request]
jobs:
  lint:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: pip install -e .[dev]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-pr.yml").write_text(
        """
name: CI
on: [pull_request]
jobs:
  lint:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: pip install -e .[dev]
      - run: echo verify
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert any("pip install must include -c constraints.txt" in error for error in result.errors)


def test_should_pass_for_metadata_only_workflow_changes(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        """
name: CI Main
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        """
name: CI Main renamed
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert result.errors == []
    assert any("metadata-only diff detected" in warning for warning in result.warnings)


def test_should_fail_when_report_path_guard_is_added_without_local_reproduction_updates(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-pr.yml").write_text(
        """
name: CI
on: [pull_request]
jobs:
  lint:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-pr.yml").write_text(
        """
name: CI
on: [pull_request]
jobs:
  lint:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
    steps:
      - run: python scripts/quality/check_no_local_paths_in_reports.py --check --report reports/quality/no_local_paths_report.json
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert any("scripts/local_checks.py" in error for error in result.errors)
    assert any("Makefile" in error for error in result.errors)
    assert any("no-local-path report guard" in error for error in result.errors)


def test_should_skip_reusable_check_for_allowlisted_workflow(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        """
name: CI Main
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo base
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        """
name: CI Main
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo changed
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts/local_checks.py").write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n\nstatus:\n\techo ok\n", encoding="utf-8")
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert not any("must call an approved reusable workflow" in error for error in result.errors)


def test_should_flag_reusable_check_for_non_allowlisted_new_workflow(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-new.yml").write_text(
        """
name: CI New
on: [push]
jobs:
  check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo base
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-new.yml").write_text(
        """
name: CI New
on: [push]
jobs:
  check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo changed
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts/local_checks.py").write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n\nstatus:\n\techo ok\n", encoding="utf-8")
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert any("must call an approved reusable workflow" in error for error in result.errors)


def test_should_not_suppress_strict_change_detection_for_allowlisted_workflow(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        """
name: CI Main
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo base
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        """
name: CI Main
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo changed
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert not any("must call an approved reusable workflow" in error for error in result.errors)
    assert any("scripts/local_checks.py" in error for error in result.errors)
    assert any("Makefile" in error for error in result.errors)


def test_should_have_dated_rationale_in_allowlist_entries() -> None:
    expected_entries = {
        ".github/workflows/ci-main.yml",
        ".github/workflows/ci-nightly.yml",
        ".github/workflows/deprecation-check.yml",
        ".github/workflows/maintenance.yml",
        ".github/workflows/update_baseline.yml",
    }

    assert expected_entries.issubset(set(_REUSABLE_FIRST_ALLOWLIST))

    validator_text = Path("scripts/quality/validate_ci_policy.py").read_text(encoding="utf-8")
    for entry in expected_entries:
        assert f'"{entry}",  # ' in validator_text
    assert validator_text.count("review-by: v0.11.3") >= len(expected_entries)


def test_should_cover_scripts_local_checks_path_in_codeowners() -> None:
    codeowners_text = Path(".github/CODEOWNERS").read_text(encoding="utf-8")
    assert "/scripts/local_checks.py @loftuw" in codeowners_text
