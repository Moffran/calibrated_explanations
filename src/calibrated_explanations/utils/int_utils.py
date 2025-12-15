"""Utility helpers for safely normalising integer inputs."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any, List

import numpy as np


def _normalize_digit_string(value: str) -> str | None:
    """Return a cleaned string that only contains an optional sign and digits."""
    stripped = value.strip()
    if not stripped:
        return None
    negative = stripped.startswith("-")
    digits = stripped[1:] if negative else stripped
    if digits.isdigit():
        return stripped
    return None


def coerce_to_int(value: Any) -> int | None:
    """Return *value* converted to an ``int`` when safe, otherwise ``None``."""
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        normalized = _normalize_digit_string(value)
        if normalized is None:
            return None
        return int(normalized)
    return None


def collect_ints(values: Any) -> List[int]:
    """Collect all integers from *values* without raising conversion errors."""
    if values is None:
        return []
    ints: List[int] = []
    if isinstance(values, (str, bytes)):
        candidate = coerce_to_int(values.decode() if isinstance(values, bytes) else values)
        if candidate is not None:
            ints.append(candidate)
        return ints
    if isinstance(values, Integral):
        ints.append(int(values))
        return ints
    if isinstance(values, Iterable):
        for element in values:
            if isinstance(element, (str, bytes)):
                candidate = coerce_to_int(
                    element.decode() if isinstance(element, bytes) else element
                )
            else:
                candidate = coerce_to_int(element)
            if candidate is not None:
                ints.append(candidate)
    return ints


def as_int_array(values: Any) -> np.ndarray:
    """Return *values* as an integer *ndarray* with invalid entries filtered."""
    return np.array(collect_ints(values), dtype=int)


__all__ = ["coerce_to_int", "collect_ints", "as_int_array"]
