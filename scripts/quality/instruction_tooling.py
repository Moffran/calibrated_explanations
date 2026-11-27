"""Generic, repository-agnostic helpers for instruction and skill tooling.

This module has no repository-specific paths or settings and no third-party
dependencies, so it can be copied to another project together with
``sync_skill_packages.py``. It provides:

* a minimal frontmatter reader for the supported metadata subset (scalars,
  JSON-style quoted strings and lists, and folded/literal block scalars);
* portable relative-path and skill-name validation;
* filesystem scanning that never follows symbolic links or Windows junctions
  and rejects unsupported filesystem objects.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

#: Agent Skills naming rule: lowercase alphanumerics separated by single hyphens.
SKILL_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
SKILL_NAME_MAX_LENGTH = 64
SKILL_DESCRIPTION_MAX_LENGTH = 1024

_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*\\')
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class ValidationError(ValueError):
    """Raised when metadata, paths or filesystem objects violate the contract."""


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def _parse_scalar(path: Path, value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith(('"', "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{path.as_posix()}: invalid scalar {value!r}") from exc
    return stripped


def parse_frontmatter(path: Path) -> dict[str, Any]:
    """Return the frontmatter mapping of a Markdown file.

    Supported subset: ``key: value`` scalars, JSON-style quoted strings and
    lists, and ``>``/``>-``/``|``/``|-`` block scalars. Nested mappings,
    duplicate keys and indented continuation lines outside block scalars are
    rejected rather than guessed.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValidationError(f"{path.as_posix()}: missing YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValidationError(f"{path.as_posix()}: unterminated YAML frontmatter") from exc

    metadata: dict[str, Any] = {}
    index = 1
    while index < end:
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        if line[:1].isspace() or ":" not in line:
            raise ValidationError(f"{path.as_posix()}: malformed frontmatter line {line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key or key in metadata:
            raise ValidationError(f"{path.as_posix()}: duplicate or empty frontmatter key {key!r}")
        raw_value = raw_value.strip()
        if raw_value in {">", ">-", "|", "|-"}:
            block: list[str] = []
            index += 1
            while index < end and (not lines[index] or lines[index][:1].isspace()):
                block.append(lines[index].strip())
                index += 1
            separator = "\n" if raw_value.startswith("|") else " "
            metadata[key] = re.sub(r"\s+", " ", separator.join(block).strip())
            continue
        metadata[key] = _parse_scalar(path, raw_value)
        index += 1
    return metadata


def require_string(path: Path, metadata: dict[str, Any], key: str) -> str:
    """Return a required non-empty string field."""
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path.as_posix()}: missing or invalid {key}")
    return value.strip()


def validate_skill_metadata(skill_md: Path, directory_name: str) -> tuple[str, str]:
    """Validate Agent Skills ``name``/``description`` and return them."""
    metadata = parse_frontmatter(skill_md)
    name = require_string(skill_md, metadata, "name")
    description = require_string(skill_md, metadata, "description")
    validate_skill_name(name, context=skill_md.as_posix())
    if name != directory_name:
        raise ValidationError(
            f"{skill_md.as_posix()}: skill name {name!r} does not match directory "
            f"{directory_name!r}"
        )
    if len(description) > SKILL_DESCRIPTION_MAX_LENGTH:
        raise ValidationError(
            f"{skill_md.as_posix()}: description exceeds {SKILL_DESCRIPTION_MAX_LENGTH} characters"
        )
    return name, description


# ---------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------


def validate_skill_name(name: str, *, context: str) -> None:
    """Reject skill names outside the portable Agent Skills naming rule."""
    if len(name) > SKILL_NAME_MAX_LENGTH or SKILL_NAME_PATTERN.fullmatch(name) is None:
        raise ValidationError(
            f"{context}: invalid skill name {name!r}; use lowercase letters, digits and "
            f"single hyphens (max {SKILL_NAME_MAX_LENGTH})"
        )


def validate_portable_component(component: str, *, context: str) -> None:
    """Reject a path component that is unsafe on Windows, macOS or Linux."""
    if component in {"", ".", ".."}:
        raise ValidationError(f"{context}: empty, '.' or '..' path component")
    if any(ord(char) < 32 or char in _FORBIDDEN_CHARACTERS for char in component):
        raise ValidationError(f"{context}: unsafe character in {component!r}")
    if component[-1] in {" ", "."}:
        raise ValidationError(f"{context}: {component!r} ends with a space or dot")
    if component.split(".", 1)[0].casefold() in _WINDOWS_RESERVED:
        raise ValidationError(f"{context}: {component!r} is a reserved Windows name")
    if len(component.encode("utf-8")) > 255:
        raise ValidationError(f"{context}: {component!r} is too long")


def validate_relative_path(value: str, *, context: str) -> PurePosixPath:
    """Return *value* as a portable, normalized relative POSIX path.

    Rejects absolute paths, drive letters, backslashes, traversal, empty
    components and names that are unsafe on any supported platform.
    """
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context}: path must be a non-empty string")
    if value.startswith("/") or re.match(r"[A-Za-z]:", value) or "\\" in value:
        raise ValidationError(f"{context}: {value!r} is not a portable relative path")
    parts = value.split("/")
    for component in parts:
        validate_portable_component(component, context=f"{context}: {value!r}")
    return PurePosixPath(*parts)


def casefold_key(path: PurePosixPath | str) -> str:
    """Return the key used to detect case-insensitive collisions."""
    return str(path).casefold()


def paths_overlap(first: PurePosixPath, second: PurePosixPath) -> bool:
    """Return True when one path equals or contains the other (case-insensitively)."""
    a = tuple(part.casefold() for part in first.parts)
    b = tuple(part.casefold() for part in second.parts)
    shortest = min(len(a), len(b))
    return a[:shortest] == b[:shortest]


# ---------------------------------------------------------------------------
# Filesystem objects
# ---------------------------------------------------------------------------


def is_link_like(path: Path) -> bool:
    """Return True for symbolic links and Windows junctions/reparse points."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def ensure_no_link_ancestors(root: Path, relative: PurePosixPath, *, context: str) -> None:
    """Reject *root/relative* when it, or any existing ancestor below *root*, is a link."""
    current = root
    for part in relative.parts:
        current = current / part
        if is_link_like(current):
            raise ValidationError(f"{context}: {current.as_posix()} is a symbolic link or junction")


@dataclass(frozen=True)
class TreeEntry:
    """One regular file or directory found by :func:`scan_tree`."""

    relative: PurePosixPath
    is_dir: bool


def scan_tree(
    directory: Path, *, context: str, ignored_names: frozenset[str] = frozenset()
) -> list[TreeEntry]:
    """Return every file and directory below *directory*, sorted by path.

    Never follows links. Raises :class:`ValidationError` on a symbolic link, a
    junction or any object that is neither a regular file nor a directory.
    Entries named in *ignored_names* (and their contents) are skipped.
    """
    entries: list[TreeEntry] = []
    pending = [PurePosixPath()]
    while pending:
        relative = pending.pop()
        with os.scandir(directory / relative) as iterator:
            for item in iterator:
                if item.name in ignored_names:
                    continue
                child = relative / item.name
                label = f"{context}/{child.as_posix()}"
                path = Path(item.path)
                if is_link_like(path):
                    raise ValidationError(f"{label}: symbolic links and junctions are not allowed")
                mode = os.lstat(path).st_mode
                if stat.S_ISDIR(mode):
                    entries.append(TreeEntry(child, True))
                    pending.append(child)
                elif stat.S_ISREG(mode):
                    entries.append(TreeEntry(child, False))
                else:
                    raise ValidationError(f"{label}: unsupported filesystem object")
    return sorted(entries, key=lambda entry: entry.relative.as_posix())


def read_text_or_binary(path: Path) -> tuple[bytes, bool]:
    """Return file bytes and whether they are UTF-8 text without NUL bytes."""
    data = path.read_bytes()
    if b"\x00" in data:
        return data, False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data, False
    return data, True
