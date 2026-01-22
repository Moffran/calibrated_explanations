"""Central deprecation helper implementing per-test/session semantics.

This helper centralises deprecation emission. Behaviour:
- When run under pytest, deprecations are deduplicated per-test so each
  test can observe warnings independently.
- Outside pytest, deprecations are emitted once-per-session by default.
- Set `CE_DEPRECATIONS=error` to elevate deprecations to exceptions (CI).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Set

# Keys emitted for the whole interpreter session (non-test runs)
_EMITTED: Set[str] = set()

# When running under pytest, deduplicate per-test so tests don't interfere
# with each other's expectations. Pytest exposes `PYTEST_CURRENT_TEST` in
# the environment during each test run.
_EMITTED_PER_TEST: Dict[str, Set[str]] = {}


def _should_raise() -> bool:
    raw = os.getenv("CE_DEPRECATIONS")
    if not raw:
        return False
    # Honor explicit error-like values for CE_DEPRECATIONS unconditionally.
    # The previous behaviour suppressed raising when pytest was active and
    # the variable was present at import time; ADRs require that deprecations
    # escalate to errors when requested, so we always treat the following
    # values as enabling raise-on-deprecations.
    return raw.lower() in {"1", "true", "error", "raise"}


def should_raise() -> bool:
    """Backwards-compatible alias for _should_raise."""
    return _should_raise()


def deprecate(message: str, *, key: str | None = None, stacklevel: int = 2) -> None:
    """Emit a `DeprecationWarning` for *message*.

    - If `CE_DEPRECATIONS` is set to an error value, raise a
      `DeprecationWarning` exception.
    - When running under pytest, deduplicate emissions per test (using
      `PYTEST_CURRENT_TEST`) so each test run can observe warnings.
    - Outside pytest, only emit once per interpreter session.

    Parameters
    ----------
    message:
        Human-facing deprecation message.
    key:
        Stable identifier for the deprecated symbol. If omitted, the
        message text is used (not recommended).
    stacklevel:
        Forwarded to ``warnings.warn`` so the warning points at user code.
    """
    if key is None:
        key = message

    # Strict CI mode: raise instead of warning
    if _should_raise():
        # Record the key so callers can inspect emitted deprecations even when
        # we raise in strict CI mode. When running under pytest, only record
        # into the per-test map to avoid polluting session-wide state and
        # preventing other tests from observing warnings.
        pytest_id = os.getenv("PYTEST_CURRENT_TEST")
        if pytest_id:
            _EMITTED_PER_TEST.setdefault(pytest_id, set()).add(key)
            raise DeprecationWarning(message)
        _EMITTED.add(key)
        raise DeprecationWarning(message)

    # Per-test dedup when running under pytest
    pytest_id = os.getenv("PYTEST_CURRENT_TEST")
    if pytest_id:
        # Under pytest, emit a warning on each call so tests can assert
        # emissions even if the same deprecated symbol is used multiple
        # times within a single test. Record the key for inspection but
        # do not suppress subsequent warnings within the same test.
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
        _EMITTED_PER_TEST.setdefault(pytest_id, set()).add(key)
        return

    # Fallback: once-per-session dedup
    if key in _EMITTED:
        return
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    _EMITTED.add(key)


def deprecate_alias(alias: str, canonical: str, *, stacklevel: int = 3) -> None:
    """Provide convenience helper for deprecated parameter/module aliases.

    Emits a concise message using a stable key derived from the alias.
    """
    key = f"alias:{alias}"
    deprecate(
        "Parameter or alias '" + alias + "' is deprecated; use '" + canonical + "'",
        key=key,
        stacklevel=stacklevel,
    )


def emitted_keys() -> set[str]:
    """Return a shallow copy of session-emitted deprecation keys."""
    return set(_EMITTED)


def emitted_per_test() -> dict[str, set[str]]:
    """Return a shallow copy of the per-test emitted map."""
    return {k: set(v) for k, v in _EMITTED_PER_TEST.items()}


def clear_emitted() -> None:
    """Clear the session-wide emitted deprecation keys."""
    _EMITTED.clear()


def clear_emitted_per_test() -> None:
    """Clear the per-test emitted deprecation map."""
    _EMITTED_PER_TEST.clear()


__all__ = [
    "deprecate",
    "deprecate_alias",
    "should_raise",
    "emitted_keys",
    "emitted_per_test",
    "clear_emitted",
    "clear_emitted_per_test",
]
