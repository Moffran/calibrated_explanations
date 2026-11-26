"""Tests for generated platform skill packages (scripts/quality/sync_skill_packages.py)."""

from __future__ import annotations

from pathlib import Path

from scripts.quality.sync_skill_packages import main

CONFIG = """\
[tool.skill-packages]
source = "skills"
targets = [".claude/skills", ".agents/skills"]
"""
TARGETS = (".claude/skills", ".agents/skills")


def _skill(root: Path, name: str, body: str = "Body.\n") -> None:
    skill_dir = root / "skills" / name
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Example skill {name}.\n---\n\n{body}",
        encoding="utf-8",
        newline="\n",
    )
    (skill_dir / "references" / "guide.md").write_text(f"{name} guide\n", encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(CONFIG, encoding="utf-8")
    _skill(tmp_path, "ce-alpha")
    _skill(tmp_path, "ce-beta")
    return tmp_path


def _run(root: Path, mode: str) -> int:
    return main(["--root", str(root), mode])


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for target in TARGETS
        for path in (root / target).rglob("*")
        if path.is_file()
    }


def test_should_generate_complete_deterministic_packages(tmp_path):
    # Arrange
    root = _repo(tmp_path)

    # Act
    first = _run(root, "--write")
    snapshot = _snapshot(root)
    second = _run(root, "--write")

    # Assert
    assert first == 0
    assert second == 0
    assert _snapshot(root) == snapshot
    assert _run(root, "--check") == 0
    for target in TARGETS:
        skill_md = (root / target / "ce-alpha" / "SKILL.md").read_text(encoding="utf-8")
        assert skill_md.startswith("---\nname: ce-alpha\ndescription: Example skill ce-alpha.\n")
        assert "Generated from skills/ce-alpha/SKILL.md" in skill_md
        assert skill_md.endswith("Body.\n")
        guide = root / target / "ce-alpha" / "references" / "guide.md"
        assert guide.read_text(encoding="utf-8") == "ce-alpha guide\n"


def test_should_propagate_add_modify_rename_and_remove(tmp_path):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    _skill(root, "ce-alpha", body="Changed.\n")
    _skill(root, "ce-gamma")
    (root / "skills" / "ce-beta").rename(root / "skills" / "ce-delta")

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


def test_should_detect_drift_without_mutating_files(tmp_path, capsys):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    claude = root / ".claude/skills"
    (claude / "ce-alpha" / "SKILL.md").write_text("edited\n", encoding="utf-8")
    (claude / "ce-alpha" / "extra.md").write_text("extra\n", encoding="utf-8")
    (root / ".agents/skills/ce-beta/references/guide.md").unlink()
    manifest = claude / ".skill-packages.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"source": "skills"', '"source": "old"'),
        encoding="utf-8",
    )
    before = _snapshot(root)
    capsys.readouterr()

    # Act
    result = _run(root, "--check")

    # Assert
    output = capsys.readouterr().out
    assert result == 1
    assert _snapshot(root) == before
    assert ".claude/skills/ce-alpha/SKILL.md: stale or modified" in output
    assert ".claude/skills/ce-alpha/extra.md: unexpected file in managed skill" in output
    assert ".agents/skills/ce-beta/references/guide.md: missing" in output
    assert ".claude/skills/.skill-packages.json: stale or missing manifest" in output


def test_should_never_touch_unmanaged_content(tmp_path, capsys):
    # Arrange
    root = _repo(tmp_path)
    _run(root, "--write")
    local = root / ".claude/skills" / "local-only"
    local.mkdir()
    (local / "SKILL.md").write_text("local\n", encoding="utf-8")
    collision = root / ".agents/skills" / "ce-new"
    collision.mkdir()
    (collision / "notes.md").write_text("mine\n", encoding="utf-8")
    _skill(root, "ce-new")
    leftover = root / ".claude/skills" / "ce-beta" / "local-notes.md"
    leftover.write_text("keep me\n", encoding="utf-8")
    capsys.readouterr()

    # Act: ce-beta is first edited locally, then removed from the source.
    removed_source = root / "skills" / "ce-beta"
    for path in sorted(removed_source.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    removed_source.rmdir()
    result = _run(root, "--write")

    # Assert
    output = capsys.readouterr().out
    assert result == 1
    assert (local / "SKILL.md").read_text(encoding="utf-8") == "local\n"
    assert "WARN: .claude/skills/local-only: unmanaged directory left untouched" in output
    assert (collision / "notes.md").read_text(encoding="utf-8") == "mine\n"
    assert not (collision / "SKILL.md").exists()
    assert "collides with an unmanaged directory" in output
    assert leftover.read_text(encoding="utf-8") == "keep me\n"
    assert not (root / ".claude/skills" / "ce-beta" / "SKILL.md").exists()
    assert not (root / ".agents/skills" / "ce-beta").exists()
