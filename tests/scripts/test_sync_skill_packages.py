"""Tests for generated platform skill packages (scripts/quality/sync_skill_packages.py)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.quality.instruction_tooling import ValidationError, validate_relative_path
from scripts.quality.sync_skill_packages import main

CONFIG = """\
[tool.skill-packages]
source = "skills"
targets = [".claude/skills", ".agents/skills"]
"""
TARGETS = (".claude/skills", ".agents/skills")
BINARY = b"\x89PNG\r\n\x1a\n\x00\xff\r\nraw\r\n"
LOCAL_SKILL = "---\nname: local-only\ndescription: Locally owned.\n---\n"


def _skill(root: Path, name: str, body: str = "Body.\n", extra: dict[str, bytes] | None = None):
    skill_dir = root / "skills" / name
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_bytes(
        f"---\nname: {name}\ndescription: Example skill {name}.\n---\n\n{body}".encode()
    )
    (skill_dir / "references" / "guide.md").write_text(f"{name} guide\n", encoding="utf-8")
    for relative, data in (extra or {}).items():
        path = skill_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    path.rmdir()


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(CONFIG, encoding="utf-8")
    _skill(tmp_path, "ce-alpha")
    _skill(tmp_path, "ce-beta")
    return tmp_path


def _run(root: Path, mode: str) -> int:
    return main(["--root", str(root), mode])


def _snapshot(root: Path) -> dict[str, bytes]:
    """Return every file below the targets without following links."""
    files: dict[str, bytes] = {}
    for target in TARGETS:
        for current, _dirs, names in os.walk(root / target, followlinks=False):
            for name in names:
                path = Path(current) / name
                if not path.is_symlink():
                    files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _make_link(link: Path, target: Path, kind: str) -> None:
    """Create a symlink or Windows junction, skipping where unsupported."""
    if kind == "junction":
        if os.name != "nt":
            pytest.skip("junctions are Windows-only")
        result = subprocess.run(  # noqa: S603
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip("cannot create a junction here")
        return
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not permitted here")


# ---------------------------------------------------------------------------
# Generation, determinism and integrity
# ---------------------------------------------------------------------------


def test_should_generate_complete_deterministic_packages(tmp_path):
    # Arrange
    root = _repo(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _repo(other)

    # Act
    first = _run(root, "--write")
    snapshot = _snapshot(root)
    second = _run(root, "--write")
    _run(other, "--write")

    # Assert
    assert first == 0
    assert second == 0
    assert _snapshot(root) == snapshot
    assert _snapshot(other) == snapshot
    assert _run(root, "--check") == 0
    for target in TARGETS:
        skill_md = (root / target / "ce-alpha" / "SKILL.md").read_text(encoding="utf-8")
        assert skill_md.startswith("---\nname: ce-alpha\ndescription: Example skill ce-alpha.\n")
        assert "Generated from skills/ce-alpha/SKILL.md" in skill_md
        assert skill_md.endswith("Body.\n")
        guide = root / target / "ce-alpha" / "references" / "guide.md"
        assert guide.read_bytes() == b"ce-alpha guide\n"


def test_should_copy_binary_byte_for_byte_and_normalize_text_to_lf(tmp_path, capsys):
    # Arrange
    root = _repo(tmp_path)
    _skill(
        root,
        "ce-alpha",
        extra={"assets/logo.png": BINARY, "assets/notes.md": b"one\r\ntwo\r\n"},
    )

    # Act
    written = _run(root, "--write")
    target_logo = root / ".claude/skills/ce-alpha/assets/logo.png"
    copied = target_logo.read_bytes()
    target_logo.write_bytes(BINARY.replace(b"\r\n", b"\n"))
    capsys.readouterr()
    drifted = _run(root, "--check")

    # Assert
    assert written == 0
    assert copied == BINARY
    assert (root / ".agents/skills/ce-alpha/assets/notes.md").read_bytes() == b"one\ntwo\n"
    assert drifted == 1
    assert ".claude/skills/ce-alpha: missing or stale" in capsys.readouterr().out


def test_should_propagate_add_modify_rename_and_remove(tmp_path):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    _skill(root, "ce-alpha", body="Changed.\n")
    _skill(root, "ce-gamma")
    (root / "skills" / "ce-beta").rename(root / "skills" / "ce-delta")
    _skill(root, "ce-delta")  # a rename also updates the frontmatter name

    # Act
    stale = _run(root, "--check")
    written = _run(root, "--write")

    # Assert
    assert stale == 1
    assert written == 0
    assert _run(root, "--check") == 0
    for target in TARGETS:
        names = {path.name for path in (root / target).iterdir() if path.is_dir()}
        assert {"ce-alpha", "ce-gamma", "ce-delta"} <= names
        assert "ce-beta" not in names
        alpha = (root / target / "ce-alpha" / "SKILL.md").read_text(encoding="utf-8")
        assert alpha.endswith("Changed.\n")


# ---------------------------------------------------------------------------
# Drift detection and ownership
# ---------------------------------------------------------------------------


def test_should_detect_drift_without_mutating_files(tmp_path, capsys):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    skill_md = root / ".claude/skills/ce-alpha/SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    (root / ".agents/skills/ce-beta/references/guide.md").unlink()
    before = _snapshot(root)
    capsys.readouterr()

    # Act
    result = _run(root, "--check")

    # Assert
    output = capsys.readouterr().out
    assert result == 1
    assert _snapshot(root) == before
    assert ".claude/skills/ce-alpha: missing or stale" in output
    assert ".agents/skills/ce-beta: missing or stale" in output


def test_should_own_generated_packages_and_never_touch_other_entries(tmp_path):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    local = root / ".claude/skills" / "local-only"
    local.mkdir()
    (local / "SKILL.md").write_text(LOCAL_SKILL, encoding="utf-8")
    (root / ".claude/skills" / "README.md").write_text("local\n", encoding="utf-8")
    stray = root / ".agents/skills" / "ce-alpha" / "local-notes.md"
    stray.write_text("edited generated package\n", encoding="utf-8")
    _remove_tree(root / "skills" / "ce-beta")

    # Act
    written = _run(root, "--write")

    # Assert
    assert written == 0
    assert _run(root, "--check") == 0
    assert (local / "SKILL.md").read_text(encoding="utf-8") == LOCAL_SKILL
    assert (root / ".claude/skills" / "README.md").read_text(encoding="utf-8") == "local\n"
    assert not stray.exists()
    assert not any((root / target / "ce-beta").exists() for target in TARGETS)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_should_leave_every_target_unchanged_when_one_target_is_blocked(tmp_path, capsys, kind):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    blocker = root / ".agents/skills" / "ce-new"
    if kind == "directory":
        blocker.mkdir()
        (blocker / "notes.md").write_text("mine\n", encoding="utf-8")
    else:
        blocker.write_text("mine\n", encoding="utf-8")
    _skill(root, "ce-alpha", body="Changed.\n")
    _skill(root, "ce-new")
    before = _snapshot(root)
    capsys.readouterr()

    # Act
    result = _run(root, "--write")

    # Assert
    assert result == 1
    assert _snapshot(root) == before
    output = capsys.readouterr().out
    assert "no target was modified" in output
    assert "not a generated package" in output


# ---------------------------------------------------------------------------
# Validation: configuration, names and filesystem objects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "targets"),
    [
        pytest.param("/abs/skills", '[".claude/skills"]', id="absolute-source"),
        pytest.param("skills", '["../outside"]', id="traversal-target"),
        pytest.param("skills", '["skills/generated"]', id="target-inside-source"),
        pytest.param("skills", '[".claude", ".claude/skills"]', id="nested-targets"),
        pytest.param("skills", '["Skills"]', id="case-insensitive-overlap"),
        pytest.param("skills", '["a\\\\b"]', id="backslash"),
        pytest.param("skills", '["C:/skills"]', id="drive-letter"),
    ],
)
def test_should_reject_unsafe_or_overlapping_configuration(tmp_path, capsys, source, targets):
    # Arrange
    root = _repo(tmp_path)
    (root / "pyproject.toml").write_text(
        f'[tool.skill-packages]\nsource = "{source}"\ntargets = {targets}\n', encoding="utf-8"
    )

    # Act
    result = _run(root, "--write")

    # Assert
    assert result == 1
    assert "failed" in capsys.readouterr().out
    assert not any((root / target).exists() for target in TARGETS)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/abs",
        "C:/x",
        "a\\b",
        "a/../b",
        "a//b",
        "./a",
        "CON.md",
        "x/aux",
        "dot.",
        "space ",
        "a:b",
    ],
)
def test_should_reject_non_portable_relative_paths(value):
    with pytest.raises(ValidationError):
        validate_relative_path(value, context="test")


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        pytest.param(
            lambda root: (root / "skills" / "Bad_Name").mkdir(),
            "invalid skill name",
            id="skill-name",
        ),
        pytest.param(
            lambda root: (root / "skills/ce-alpha/SKILL.md").write_text(
                "---\nname: ce-other\ndescription: Wrong.\n---\n", encoding="utf-8"
            ),
            "does not match directory",
            id="name-mismatch",
        ),
    ],
)
def test_should_reject_invalid_source_before_writing(tmp_path, capsys, setup, message):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    _skill(root, "ce-beta", body="Changed.\n")
    setup(root)
    before = _snapshot(root)
    capsys.readouterr()

    # Act
    result = _run(root, "--write")

    # Assert
    assert result == 1
    assert _snapshot(root) == before
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("location", ["source-skill", "managed-target-skill", "target-root"])
def test_should_reject_links_instead_of_following_them(tmp_path, capsys, kind, location):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("outside\n", encoding="utf-8")
    if location == "source-skill":
        _make_link(root / "skills/ce-alpha/linked", outside, kind)
    elif location == "managed-target-skill":
        _make_link(root / ".claude/skills/ce-alpha/linked", outside, kind)
    else:
        _remove_tree(root / ".agents/skills")
        _make_link(root / ".agents/skills", outside, kind)
    _skill(root, "ce-beta", body="Changed.\n")
    before = _snapshot(root)
    capsys.readouterr()

    # Act
    result = _run(root, "--write")

    # Assert
    assert result == 1
    assert _snapshot(root) == before
    assert (outside / "secret.md").read_text(encoding="utf-8") == "outside\n"
    assert sorted(p.name for p in outside.iterdir()) == ["secret.md"]
    output = capsys.readouterr().out
    assert "symbolic link or junction" in output or "symbolic links and junctions" in output
