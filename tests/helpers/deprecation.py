from __future__ import annotations

import os
from contextlib import contextmanager
import pytest


def _env_flag_deprecations_error() -> bool:
    val = os.getenv("CE_DEPRECATIONS", "").strip().lower()
    return val in ("1", "true", "error", "raise")


def deprecations_error_enabled() -> bool:
    """Public helper to check whether raise-on-deprecations mode is active."""
    return _env_flag_deprecations_error()


@contextmanager
def warns_or_raises(match: str | None = None):
    """Context manager that yields a context which expects either a
    DeprecationWarning warning or a raised DeprecationWarning depending on
    the `CE_DEPRECATIONS` environment variable.

    - If `CE_DEPRECATIONS` indicates error-like value, this context acts as
      `pytest.raises(DeprecationWarning, match=...)`.
    - Otherwise it acts as `pytest.warns(DeprecationWarning, match=...)`.
    """
    if _env_flag_deprecations_error():
        with pytest.raises(DeprecationWarning, match=match):
            yield
    else:
        with pytest.warns(DeprecationWarning, match=match):
            yield
