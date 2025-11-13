"""Configuration parsing and coercion utilities for CalibratedExplainer.

This module provides helper functions for reading and parsing external
configuration sources like pyproject.toml and environment variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _tomllib = None  # type: ignore[assignment]


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
    except Exception:  # pragma: no cover - permissive fallback
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
            if isinstance(item, str) and item:
                result.append(item)
        return tuple(result)
    return ()
