"""Changed-file lint targets must exclude files deleted in the working tree."""

from __future__ import annotations

import shutil
import subprocess

import pytest

import scripts.local_checks as local_checks

PLAN_WITHOUT_LINT_TARGETS = """\
```toml ce-task-verification
schema_version = 1

[[task.1.steps]]
name = "noop"
command = "python -c pass"
```
"""


def _git(cwd, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=ce", "-c", "user.email=ce@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_should_exclude_deleted_python_files_from_changed_targets(tmp_path, monkeypatch):
    # Arrange
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "removed.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "kept.py").write_text("y = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    (tmp_path / "src" / "removed.py").unlink()
    (tmp_path / "src" / "kept.py").write_text("y = 2\n", encoding="utf-8")
    plan = tmp_path / "plan.md"
    plan.write_text(PLAN_WITHOUT_LINT_TARGETS, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act: a task without lint_targets falls back to the changed Python files.
    targets = local_checks.task_specific_lint_targets(1, plan_path=plan)

    # Assert
    assert "src/kept.py" in targets
    assert "src/removed.py" not in targets
