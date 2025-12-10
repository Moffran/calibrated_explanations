"""Helpers for testing how the codebase handles deprecation warnings."""

from __future__ import annotations

import os
from contextlib import contextmanager
import pytest


def env_flag_deprecations_error() -> bool:
    """Return True when CE_DEPRECATIONS forces errors for deprecated features."""
    val = os.getenv("CE_DEPRECATIONS", "").strip().lower()
    return val in ("1", "true", "error", "raise")


def deprecations_error_enabled() -> bool:
    """Public helper to check whether raise-on-deprecations mode is active."""
    return env_flag_deprecations_error()


@contextmanager
def warns_or_raises(match: str | None = None):
    """Context manager that ensures a DeprecationWarning is raised or warned.

    The behavior mirrors `pytest.raises` when `CE_DEPRECATIONS` signals an
    error-like value and `pytest.warns` otherwise.
    """
    if env_flag_deprecations_error():
        with pytest.raises(DeprecationWarning, match=match):
            yield
    else:
        with pytest.warns(DeprecationWarning, match=match):
            yield
