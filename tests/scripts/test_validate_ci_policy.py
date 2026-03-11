from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.quality.validate_ci_policy import validate_policy


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
