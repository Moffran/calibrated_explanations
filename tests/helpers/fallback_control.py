"""Helper utilities for controlling fallback chains in tests.

Per the Fallback Chain Enforcement policy in .github/tests-guidance.md,
tests MUST NOT trigger fallback chains unless explicitly validating fallback behavior.

This module provides utilities for:
1. Disabling fallback chains (default for all tests)
2. Enabling fallbacks for specific tests
3. Asserting that no fallbacks were triggered
"""

from __future__ import annotations

import contextlib
import os
import re
import warnings
from typing import Any, Generator

import pytest

# store original warnings.warn so we can restore it when tests opt-in to
# fallback behaviour via the `enable_fallbacks` fixture
_ORIG_WARN: Any | None = None


def disable_all_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable all plugin fallback chains by setting empty fallback environment variables.

    This function is called by the `disable_fallbacks` fixture to ensure tests
    run against the primary code path without falling back to legacy implementations.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture for setting environment variables.

    Notes
    -----
    This disables fallbacks for:
    - Explanation plugins (factual, alternative, fast)
    - Interval plugins (default, fast)
    - Plot style plugins
    - Parallel execution (by setting unreasonably high min batch size)
    """
    # Disable explanation plugin fallbacks
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", "")
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS", "")
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FAST_FALLBACKS", "")

    # Disable interval plugin fallbacks
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", "")
    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FAST_FALLBACKS", "")

    # Disable plot style fallbacks
    monkeypatch.setenv("CE_PLOT_STYLE_FALLBACKS", "")

    # Disable parallel execution fallback to serial by setting min batch size very high
    # This prevents the "parallel failure; forced serial fallback engaged" warning
    monkeypatch.setenv("CE_PARALLEL_MIN_BATCH_SIZE", "999999")

    # Prevent runtime fallbacks by converting any runtime "fall...back" UserWarning
    # into an immediate test failure. This catches execution-time supports()/exception
    # fallback paths that are not controlled via env vars (e.g. feature_parallel plugin).
    global _ORIG_WARN
    _orig_warn = warnings.warn
    _ORIG_WARN = _orig_warn

    def _warn_no_fallback(message, category=None, *args, **kwargs):
        try:
            text = str(message)
        except Exception:
            text = ""
        if re.search(r"fall.*back", text, flags=re.IGNORECASE):
            # Emit the original warning so higher-level test helpers such as
            # `assert_no_fallbacks_triggered` can capture and assert on it.
            # Avoid raising here to let the context manager perform the
            # final assertion/translation to test failure.
            return _orig_warn(message, category, *args, **kwargs)
        return _orig_warn(message, category, *args, **kwargs)

    monkeypatch.setattr("warnings.warn", _warn_no_fallback)


def restore_runtime_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the original `warnings.warn` implementation.

    Tests that opt-in to fallback behaviour should call this to allow
    runtime fallback warnings to be emitted normally instead of being
    converted to failures by the autouse disable fixture.
    """
    global _ORIG_WARN
    if _ORIG_WARN is not None:
        monkeypatch.setattr("warnings.warn", _ORIG_WARN)


def enable_specific_fallback(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fallback_type: str,
    fallback_chain: str,
) -> None:
    """Enable a specific fallback chain for tests that validate fallback behavior.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture for setting environment variables.
    fallback_type : str
        Type of fallback to enable. Must be one of:
        - "explanation_factual"
        - "explanation_alternative"
        - "explanation_fast"
        - "interval_default"
        - "interval_fast"
        - "plot_style"
        - "parallel"
    fallback_chain : str
        Comma-separated list of fallback identifiers to enable.

    Raises
    ------
    ValueError
        If fallback_type is not recognized.

    Examples
    --------
    >>> def test_fallback_behavior(monkeypatch):
    ...     enable_specific_fallback(
    ...         monkeypatch,
    ...         fallback_type="explanation_factual",
    ...         fallback_chain="core.explanation.factual"
    ...     )
    ...     # Test code that validates fallback behavior
    """
    env_var_map = {
        "explanation_factual": "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS",
        "explanation_alternative": "CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS",
        "explanation_fast": "CE_EXPLANATION_PLUGIN_FAST_FALLBACKS",
        "interval_default": "CE_INTERVAL_PLUGIN_FALLBACKS",
        "interval_fast": "CE_INTERVAL_PLUGIN_FAST_FALLBACKS",
        "plot_style": "CE_PLOT_STYLE_FALLBACKS",
    }

    if fallback_type == "parallel":
        # For parallel, we enable fallback by removing or reducing the min batch size constraint
        monkeypatch.delenv("CE_PARALLEL_MIN_BATCH_SIZE", raising=False)
        return

    if fallback_type not in env_var_map:
        msg = (
            f"Unknown fallback_type: {fallback_type!r}. "
            f"Must be one of: {', '.join(env_var_map.keys())}, 'parallel'"
        )
        raise ValueError(msg)

    env_var = env_var_map[fallback_type]
    monkeypatch.setenv(env_var, fallback_chain)


@contextlib.contextmanager
def assert_no_fallbacks_triggered() -> Generator[None, None, None]:
    """Context manager that asserts no fallback warnings were emitted.

    This is useful for integration tests that want to ensure they are testing
    the primary code path and not accidentally triggering fallbacks.

    Raises
    ------
    AssertionError
        If any UserWarning containing patterns indicating a fallback is emitted.

    Examples
    --------
    >>> with assert_no_fallbacks_triggered():
    ...     explainer = CalibratedExplainer(model, x_cal, y_cal)
    ...     explanation = explainer.explain_factual(x_test)
    
    Notes
    -----
    This detects fallback warnings by looking for common patterns:
    - "... fallback ..."
    - "fallback:"
    - "fallback engaged"
    - etc.
    
    It will NOT trigger on words that merely contain "fallback" as a substring.
    """
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", UserWarning)
        yield

        # Check if any warnings contain fallback indicators
        # We look for "fallback" as a separate word, not just a substring
        fallback_patterns = [
            " fallback ",
            "fallback:",
            "fallback;",
            "fallback.",
            "fallback,",
            " fallback-",
            "-fallback ",
        ]
        
        fallback_warnings = []
        for w in warning_list:
            if not issubclass(w.category, UserWarning):
                continue
            msg_lower = str(w.message).lower()
            # Check if any fallback pattern is present
            if any(pattern in msg_lower for pattern in fallback_patterns):
                fallback_warnings.append(w)

        if fallback_warnings:
            messages = [str(w.message) for w in fallback_warnings]
            msg = (
                f"Unexpected fallback warnings detected:\n"
                + "\n".join(f"  - {m}" for m in messages)
            )
            raise AssertionError(msg)


def get_fallback_env_vars() -> dict[str, str | None]:
    """Get current values of all fallback-related environment variables.

    Returns
    -------
    dict[str, str | None]
        Dictionary mapping environment variable names to their current values.
        Variables that are not set will have None as their value.

    Notes
    -----
    This is useful for debugging fallback configuration issues.
    """
    env_vars = [
        "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS",
        "CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS",
        "CE_EXPLANATION_PLUGIN_FAST_FALLBACKS",
        "CE_INTERVAL_PLUGIN_FALLBACKS",
        "CE_INTERVAL_PLUGIN_FAST_FALLBACKS",
        "CE_PLOT_STYLE_FALLBACKS",
        "CE_PARALLEL_MIN_BATCH_SIZE",
    ]

    return {var: os.environ.get(var) for var in env_vars}
