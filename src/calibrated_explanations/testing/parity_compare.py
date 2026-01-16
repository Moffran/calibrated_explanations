"""Helpers for deterministic parity comparisons."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def parity_compare(
    expected: Any,
    actual: Any,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> list[dict[str, Any]]:
    """Compare nested structures, returning structured diffs.

    Parameters
    ----------
    expected : Any
        The expected nested structure.
    actual : Any
        The actual nested structure to compare.
    rtol : float
        Relative tolerance for numeric comparisons.
    atol : float
        Absolute tolerance for numeric comparisons.
    """
    diffs: list[dict[str, Any]] = []

    def normalize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [normalize(item) for item in value.tolist()]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(val) for key, val in value.items()}
        return value

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)

    def both_nan(left: Any, right: Any) -> bool:
        return (
            is_number(left)
            and is_number(right)
            and math.isnan(float(left))
            and math.isnan(float(right))
        )

    def add_diff(path: str, expected_value: Any, actual_value: Any, reason: str) -> None:
        diffs.append(
            {
                "path": path,
                "expected": expected_value,
                "actual": actual_value,
                "reason": reason,
            }
        )

    def path_join(base: str, token: str) -> str:
        if base == "$":
            return f"{base}.{token}"
        return f"{base}.{token}"

    def compare(expected_value: Any, actual_value: Any, path: str) -> None:
        expected_value = normalize(expected_value)
        actual_value = normalize(actual_value)

        if both_nan(expected_value, actual_value):
            return

        if is_number(expected_value) and is_number(actual_value):
            if math.isclose(float(expected_value), float(actual_value), rel_tol=rtol, abs_tol=atol):
                return
            add_diff(path, expected_value, actual_value, "value_mismatch")
            return

        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            expected_keys = set(expected_value.keys())
            actual_keys = set(actual_value.keys())
            for key in sorted(expected_keys - actual_keys):
                add_diff(path_join(path, str(key)), expected_value[key], None, "missing_key")
            for key in sorted(actual_keys - expected_keys):
                add_diff(path_join(path, str(key)), None, actual_value[key], "extra_key")
            for key in sorted(expected_keys & actual_keys):
                compare(expected_value[key], actual_value[key], path_join(path, str(key)))
            return

        if isinstance(expected_value, list) and isinstance(actual_value, list):
            if len(expected_value) != len(actual_value):
                add_diff(path, len(expected_value), len(actual_value), "length_mismatch")
            for idx, (left, right) in enumerate(zip(expected_value, actual_value, strict=False)):
                compare(left, right, f"{path}[{idx}]")
            return

        if expected_value != actual_value:
            add_diff(path, expected_value, actual_value, "value_mismatch")

    compare(expected, actual, "$")
    return diffs


__all__ = ["parity_compare"]
