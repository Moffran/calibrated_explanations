"""Generate platform skill-discovery packages from one canonical skill catalogue.

Generic, dependency-free tool. Repository-specific settings live in the
``[tool.skill-packages]`` table of ``pyproject.toml``::

    [tool.skill-packages]
    source = "skills"                         # canonical <name>/SKILL.md catalogue
    targets = [".claude/skills", ".agents/skills"]

Every directory directly under ``source`` that contains ``SKILL.md`` is a skill.
Each target receives a complete copy of every skill (SKILL.md plus all
references, scripts, assets and examples). ``SKILL.md`` copies get a one-line
"generated, do not edit" notice after their frontmatter; all other files are
byte-for-byte copies.

Ownership is recorded in ``<target>/.skill-packages.json``. Reconciliation is
full-set: added, changed, renamed and removed skills propagate on ``--write``,
and files of a removed skill are deleted only when the manifest lists them.
Directories the manifest does not list are unmanaged: ``--write`` never touches
them and ``--check`` reports them as warnings. ``--check`` never writes and
fails on any missing, stale, modified or unexpected managed file, a stale
manifest, or a canonical skill whose name collides with an unmanaged directory.

Usage
-----
    python scripts/quality/sync_skill_packages.py --check
    python scripts/quality/sync_skill_packages.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (dev extras provide tomli)
    import tomli as tomllib  # type: ignore[no-redef]

MANIFEST_NAME = ".skill-packages.json"
MANIFEST_SCHEMA = 1
SKILL_FILE = "SKILL.md"
_IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store", "Thumbs.db"})


@dataclass(frozen=True)
class Config:
    """Resolved synchronization settings."""

    root: Path
    source: Path
    targets: tuple[Path, ...]


@dataclass
class Report:
    """Outcome of one reconciliation pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)


class ConfigError(ValueError):
    """Raised when the synchronization configuration or source is invalid."""


def load_config(root: Path, pyproject: Path | None = None) -> Config:
    """Read ``[tool.skill-packages]`` from ``pyproject.toml`` under *root*."""
    path = pyproject or root / "pyproject.toml"
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8"))["tool"]["skill-packages"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: missing or invalid [tool.skill-packages] table") from exc
    source = table.get("source")
    targets = table.get("targets")
    if not isinstance(source, str) or not source:
        raise ConfigError("[tool.skill-packages] source must be a non-empty string")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(t, str) and t for t in targets)
    ):
        raise ConfigError("[tool.skill-packages] targets must be a non-empty string list")
    if source in targets or len(set(targets)) != len(targets):
        raise ConfigError(
            "[tool.skill-packages] targets must be distinct from each other and source"
        )
    return Config(root=root, source=Path(source), targets=tuple(Path(t) for t in targets))


def _files(directory: Path) -> list[str]:
    """Return sorted POSIX paths of files below *directory*, skipping OS/tool noise."""
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and not _IGNORED_NAMES.intersection(path.relative_to(directory).parts)
    )


def _notice(source: Path, skill: str) -> str:
    return (
        f"<!-- Generated from {source.as_posix()}/{skill}/{SKILL_FILE} by "
        "scripts/quality/sync_skill_packages.py; do not edit. -->"
    )


def _render(source_file: Path, relative: str, *, source: Path, skill: str) -> bytes:
    """Return the expected bytes of one generated file."""
    data = source_file.read_bytes()
    if relative != SKILL_FILE:
        return data
    lines = data.decode("utf-8").splitlines(keepends=True)
    fences = [index for index, line in enumerate(lines) if line.strip().lstrip("﻿") == "---"]
    if len(fences) < 2 or fences[0] != 0:
        raise ConfigError(f"{source_file.as_posix()}: missing YAML frontmatter")
    closing = lines[fences[1]]
    newline = closing[len(closing.rstrip("\r\n")) :] or "\n"
    lines.insert(fences[1] + 1, _notice(source, skill) + newline)
    return "".join(lines).encode("utf-8")


def _same(actual: bytes, expected: bytes) -> bool:
    """Compare file content, ignoring CRLF/LF differences from git checkouts."""
    return actual.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n")


def discover_skills(config: Config) -> dict[str, list[str]]:
    """Return ``{skill name: sorted files}`` for the canonical catalogue."""
    source_root = config.root / config.source
    if not source_root.is_dir():
        raise ConfigError(f"canonical skill source {config.source.as_posix()} does not exist")
    skills: dict[str, list[str]] = {}
    for child in sorted(source_root.iterdir()):
        if not child.is_dir() or child.name in _IGNORED_NAMES:
            continue
        if not (child / SKILL_FILE).is_file():
            raise ConfigError(f"{config.source.as_posix()}/{child.name}: missing {SKILL_FILE}")
        skills[child.name] = _files(child)
    if not skills:
        raise ConfigError(f"no skills found under {config.source.as_posix()}")
    return skills


def _manifest_text(skills: dict[str, list[str]], source: Path) -> str:
    payload = {
        "generated_by": "scripts/quality/sync_skill_packages.py",
        "schema_version": MANIFEST_SCHEMA,
        "source": source.as_posix(),
        "skills": skills,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_manifest(path: Path, label: str, report: Report) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        skills = payload["skills"]
        if payload.get("schema_version") != MANIFEST_SCHEMA or not isinstance(skills, dict):
            raise ValueError("unsupported manifest")
        return {str(name): [str(f) for f in files] for name, files in skills.items()}
    except (ValueError, KeyError, TypeError, AttributeError):
        report.errors.append(f"{label}: unreadable ownership manifest")
        return {}


def _reconcile_target(
    config: Config, target: Path, skills: dict[str, list[str]], *, write: bool, report: Report
) -> None:
    target_root = config.root / target
    manifest_path = target_root / MANIFEST_NAME
    previous = _read_manifest(manifest_path, (target / MANIFEST_NAME).as_posix(), report)
    existing = (
        {c.name for c in target_root.iterdir() if c.is_dir() and c.name not in _IGNORED_NAMES}
        if target_root.is_dir()
        else set()
    )

    for name in sorted(existing - set(previous) - set(skills)):
        report.warnings.append(f"{(target / name).as_posix()}: unmanaged directory left untouched")
    for name in sorted((existing - set(previous)) & set(skills)):
        report.errors.append(
            f"{(target / name).as_posix()}: canonical skill collides with an unmanaged "
            "directory; move or delete it manually"
        )

    for name, files in skills.items():
        if name in existing and name not in previous:
            continue  # collision reported above; never overwrite unmanaged content
        skill_dir = target_root / name
        for relative in files:
            expected = _render(
                config.root / config.source / name / relative,
                relative,
                source=config.source,
                skill=name,
            )
            destination = skill_dir / relative
            label = (target / name / relative).as_posix()
            if destination.is_file() and _same(destination.read_bytes(), expected):
                continue
            state = "stale or modified" if destination.exists() else "missing"
            if write:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(expected)
                report.changes.append(f"wrote {label}")
            else:
                report.errors.append(f"{label}: {state}")
        if skill_dir.is_dir():
            for relative in sorted(set(_files(skill_dir)) - set(files)):
                label = (target / name / relative).as_posix()
                if write:
                    (skill_dir / relative).unlink()
                    report.changes.append(f"removed {label}")
                else:
                    report.errors.append(f"{label}: unexpected file in managed skill")

    for name in sorted(set(previous) - set(skills)):
        skill_dir = target_root / name
        if not skill_dir.is_dir():
            continue
        listed = set(previous[name])
        present = set(_files(skill_dir))
        if not write:
            report.errors.append(f"{(target / name).as_posix()}: skill removed from source")
            continue
        for relative in sorted(present & listed):
            (skill_dir / relative).unlink()
            report.changes.append(f"removed {(target / name / relative).as_posix()}")
        for relative in sorted(present - listed):
            report.warnings.append(
                f"{(target / name / relative).as_posix()}: not in manifest; left untouched"
            )
        for directory in sorted(
            (p for p in skill_dir.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if not any(directory.iterdir()):
                directory.rmdir()
        if not any(skill_dir.iterdir()):
            skill_dir.rmdir()

    managed = {
        name: files for name, files in skills.items() if name not in existing - set(previous)
    }
    manifest = _manifest_text(managed, config.source)
    current = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else None
    if current is not None and _same(current.encode(), manifest.encode()):
        return
    if write:
        target_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest, encoding="utf-8", newline="\n")
        report.changes.append(f"wrote {(target / MANIFEST_NAME).as_posix()}")
    else:
        report.errors.append(f"{(target / MANIFEST_NAME).as_posix()}: stale or missing manifest")


def synchronize(config: Config, *, write: bool) -> Report:
    """Reconcile every target with the canonical catalogue."""
    report = Report()
    skills = discover_skills(config)
    for target in config.targets:
        _reconcile_target(config, target, skills, write=write, report=report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--config", type=Path, default=None, help="pyproject.toml to read.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Regenerate skill packages.")
    mode.add_argument("--check", action="store_true", help="Check without changing files.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        config = load_config(root, args.config)
        report = synchronize(config, write=args.write)
    except (ConfigError, OSError, UnicodeDecodeError) as exc:
        print(f"Skill package synchronization failed: {exc}")
        return 1

    for line in report.changes:
        print(line)
    for line in report.warnings:
        print(f"WARN: {line}")
    if report.errors:
        print("Skill package check failed:" if args.check else "Skill package sync failed:")
        for line in report.errors:
            print(f"- {line}")
        if args.check:
            print("Run: make skills-sync")
        return 1
    targets = ", ".join(t.as_posix() for t in config.targets)
    if args.write:
        print(f"Skill packages synchronized ({len(report.changes)} change(s)) for: {targets}")
    else:
        print(f"Skill package check passed for: {targets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
