from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.quality.validate_ci_policy import _REUSABLE_FIRST_ALLOWLIST, validate_policy


def run_git(args: list[str], tmp_path: Path, *, capture_output: bool = False, text: bool = False) -> subprocess.CompletedProcess:
    """Execute a git command in a test repository."""
    git_bin = shutil.which("git")
    if git_bin is None:
        pytest.skip("git executable is not available in PATH")
    try:
        return subprocess.run(
            [git_bin, *args],
            cwd=tmp_path,
            check=True,
            capture_output=capture_output,
            text=text,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git command is unavailable in this environment: {exc}")


def init_repo(tmp_path: Path) -> None:
    """Initialize a test git repository."""
    run_git(["init"], tmp_path, capture_output=True)
    if not (tmp_path / ".git").is_dir():
        pytest.skip("git init did not create a repository in temporary test path")
    run_git(["config", "user.email", "ci@example.com"], tmp_path)
    run_git(["config", "user.name", "CI"], tmp_path)


def commit_all(tmp_path: Path, message: str) -> str:
    """Commit all changes in a test repository and return commit SHA."""
    run_git(["add", "."], tmp_path)
    run_git(["commit", "-m", message], tmp_path)
    sha = run_git(["rev-parse", "HEAD"], tmp_path, text=True, capture_output=True)
    return sha.stdout.strip()


CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"


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


def test_should_allow_pip_bootstrap_upgrade_without_constraints(tmp_path: Path) -> None:
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
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: python -m pip install --upgrade pip
      - run: pip install -e .[dev] -c constraints.txt
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
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: python -m pip install --upgrade pip
      - run: pip install -e .[dev] -c constraints.txt
      - run: echo verify
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert not any("pip install must include -c constraints.txt" in error for error in result.errors)


def test_should_fail_when_external_action_is_major_tag(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        f"""
name: CI Main
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA}
      - uses: actions/setup-python@v6
        with:
          python-version: '3.11'
          cache: pip
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        f"""
name: CI Main
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@{SETUP_PYTHON_SHA}
        with:
          python-version: '3.11'
          cache: pip
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert any("must use a full 40-character commit SHA" in error for error in result.errors)


def test_should_pass_when_external_actions_use_full_sha(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        f"""
name: CI Main
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA}
      - uses: actions/setup-python@{SETUP_PYTHON_SHA}
        with:
          python-version: '3.11'
          cache: pip
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        f"""
name: CI Main Updated
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA}
      - uses: actions/setup-python@{SETUP_PYTHON_SHA}
        with:
          python-version: '3.11'
          cache: pip
      - run: echo verify
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts/local_checks.py").write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text(
        "# includes full-SHA pin enforcement parity\nlocal-checks:\n\tpython scripts/local_checks.py\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert result.errors == []


def test_should_fail_when_external_action_missing_version(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/workflows/ci-main.yml").write_text(
        f"""
name: CI Main
on:
  pull_request:
    branches: [main]
jobs:
  tests:
    uses: ./.github/workflows/reusable-python-test.yml
    permissions:
      contents: read
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA}
      - uses: actions/setup-python@{SETUP_PYTHON_SHA}
        with:
          python-version: '3.11'
          cache: pip
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

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
  audit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout
      - uses: actions/setup-python@v6
        with:
          python-version: '3.11'
          cache: pip
""".strip()
        + "\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert any("must be pinned to a full commit SHA" in error for error in result.errors)


def test_should_pass_when_local_action_is_unpinned(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/actions/ci-policy").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts/local_checks.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("local-checks:\n\tpython scripts/local_checks.py\n", encoding="utf-8")
    (tmp_path / ".github/actions/ci-policy/action.yml").write_text(
        """
name: ci-policy
runs:
  using: composite
  steps:
    - run: echo ok
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".github/workflows/ci-policy.yml").write_text(
        f"""
name: ci-policy/validate-workflows
on:
  pull_request:
    branches: [main]
permissions:
  contents: read
jobs:
  validate-workflows:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA}
      - uses: actions/setup-python@{SETUP_PYTHON_SHA}
        with:
          python-version: '3.11'
      - uses: ./.github/actions/ci-policy
        with:
          base-sha: abc123
          head-sha: def456
          advisory: 'true'
""".strip()
        + "\n",
        encoding="utf-8",
    )
    base_sha = commit_all(tmp_path, "base")

    (tmp_path / ".github/workflows/ci-policy.yml").write_text(
        f"""
name: ci-policy/validate-workflows
on:
  pull_request:
    branches: [main]
permissions:
  contents: read
jobs:
  validate-workflows:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA}
      - uses: actions/setup-python@{SETUP_PYTHON_SHA}
        with:
          python-version: '3.11'
      - uses: ./.github/actions/ci-policy
        with:
          base-sha: abc123
          head-sha: def456
          advisory: 'false'
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts/local_checks.py").write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text(
        "# includes full-SHA pin enforcement parity\nlocal-checks:\n\tpython scripts/local_checks.py\n",
        encoding="utf-8",
    )
    head_sha = commit_all(tmp_path, "head")

    result = validate_policy(base_sha=base_sha, head_sha=head_sha, repo_root=tmp_path)

    assert result.errors == []


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
