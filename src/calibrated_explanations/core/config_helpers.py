"""Configuration parsing and coercion utilities for CalibratedExplainer.

This module provides helper functions for reading and parsing external
configuration sources like pyproject.toml and environment variables.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from calibrated_explanations.core.exceptions import CalibratedError

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _tomllib = None  # type: ignore[assignment]

try:
    import tomli_w as _tomli_w
except ModuleNotFoundError:  # pragma: no cover - optional for writing
    _tomli_w = None  # type: ignore[assignment]


# Sentinel to distinguish explicit `None` from omitted arguments
_UNSET = object()


def get_toml_modules_for_testing() -> tuple[Any, Any]:
    """Return the current TOML reader/writer modules (testing helper)."""
    return _tomllib, _tomli_w


def set_toml_modules_for_testing(
    *, tomllib: Any | None = _UNSET, tomli_w: Any | None = _UNSET
) -> None:
    """Override TOML reader/writer modules for tests without private access.

    Use a sentinel default so callers can explicitly set a module to ``None``
    (for example to simulate an unavailable writer) without being ignored.
    """
    global _tomllib, _tomli_w  # noqa: PLW0603 - test override helper
    if tomllib is not _UNSET:
        _tomllib = tomllib
    if tomli_w is not _UNSET:
        _tomli_w = tomli_w


def read_pyproject_section(path: Sequence[str]) -> Dict[str, Any]:
    """Return a mapping from the requested ``pyproject.toml`` section.

    Parameters
    ----------
    path : Sequence[str]
        Nested keys to traverse in the pyproject.toml structure.
        For example, ``("tool", "calibrated_explanations", "explanations")``
        will navigate to ``[tool.calibrated_explanations.explanations]``.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the requested configuration section,
        or an empty dict if the file does not exist or parsing fails.

    Examples
    --------
    >>> config = read_pyproject_section(("tool", "calibrated_explanations"))
    >>> if config:
    ...     print(f"Found config: {config}")
    """
    if _tomllib is None:
        return {}

    candidate = Path.cwd() / "pyproject.toml"
    if not candidate.exists():
        return {}
    try:
        with candidate.open("rb") as fh:  # type: ignore[arg-type]
            data = _tomllib.load(fh)
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        return {}

    cursor: Any = data
    for key in path:
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        else:
            return {}
    if isinstance(cursor, dict):
        return dict(cursor)
    return {}


def write_pyproject_section(path: Sequence[str], value: Dict[str, Any]) -> bool:
    """Update a section in pyproject.toml with the given value.

    Parameters
    ----------
    path : Sequence[str]
        Nested keys to traverse in the pyproject.toml structure.
    value : Dict[str, Any]
        The value to set at the specified path.

    Returns
    -------
    bool
        True if the file was updated, False otherwise.

    Notes
    -----
    This function requires tomli_w to be installed for writing TOML.
    If not available, it will return False without modifying the file.
    """
    if _tomllib is None or _tomli_w is None:
        return False

    candidate = Path.cwd() / "pyproject.toml"
    if not candidate.exists():
        return False

    try:
        with open(candidate, "rb") as f:
            data = _tomllib.load(f)
    except CalibratedError:
        return False

    # Navigate to the parent of the target section
    cursor = data
    for key in path[:-1]:
        if not isinstance(cursor, dict):
            return False
        if key not in cursor:
            cursor[key] = {}
        cursor = cursor[key]

    if not isinstance(cursor, dict):
        return False

    cursor[path[-1]] = value

    try:
        with open(candidate, "wb") as f:
            _tomli_w.dump(data, f)
        return True
    except CalibratedError:
        return False


def split_csv(value: str | None) -> Tuple[str, ...]:
    """Split a comma-separated environment variable into a tuple of strings.

    Parameters
    ----------
    value : str or None
        A comma-separated string (e.g., from an environment variable).
        Leading and trailing whitespace around each entry is stripped.

    Returns
    -------
    Tuple[str, ...]
        Tuple of non-empty string entries. Returns empty tuple if value is None
        or contains no non-empty entries.

    Examples
    --------
    >>> split_csv("foo, bar , baz")
    ('foo', 'bar', 'baz')

    >>> split_csv(None)
    ()

    >>> split_csv("   ,  ,  ")
    ()
    """
    if not value:
        return ()
    entries = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(entries)


def coerce_string_tuple(value: Any) -> Tuple[str, ...]:
    """Coerce a configuration value into a tuple of strings.

    Handles multiple input types:
    - ``None`` → empty tuple
    - ``str`` → tuple with single entry (if non-empty)
    - ``Iterable[str]`` → tuple of non-empty entries
    - Other types → empty tuple

    Parameters
    ----------
    value : Any
        Value to coerce into a string tuple.

    Returns
    -------
    Tuple[str, ...]
        Tuple of non-empty strings, or empty tuple if value cannot be coerced.

    Examples
    --------
    >>> coerce_string_tuple("myvalue")
    ('myvalue',)

    >>> coerce_string_tuple(["foo", "bar", ""])
    ('foo', 'bar')

    >>> coerce_string_tuple(None)
    ()

    >>> coerce_string_tuple("")
    ()
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return tuple(result)
    return ()
