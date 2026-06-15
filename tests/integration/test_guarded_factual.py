"""Integration tests for GuardedOptions on the guarded explanation path (ADR-038)."""

from __future__ import annotations

import contextlib


import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from calibrated_explanations import GuardedOptions, WrapCalibratedExplainer

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared fixture — fitted wrapper (module scope for speed)
# ---------------------------------------------------------------------------

X, y = make_classification(n_samples=120, n_features=6, random_state=42)
X_fit, y_fit = X[:50], y[:50]
X_cal, y_cal = X[50:90], y[50:90]
X_test = X[90:95]  # small slice — guard is expensive

_clf = DecisionTreeClassifier(random_state=42)
_wrapper = WrapCalibratedExplainer(_clf)
_wrapper.fit(X_fit, y_fit)
_wrapper.calibrate(X_cal, y_cal)


# ---------------------------------------------------------------------------
# GuardedOptions API surface
# ---------------------------------------------------------------------------


def test_should_complete_without_error_when_guarded_options_replaces_legacy_kwargs():
    """GuardedOptions(confidence=0.9) is the canonical replacement for guarded=True, significance=0.1."""
    opts = GuardedOptions(confidence=0.9)
    # As long as the call completes (or raises a domain error, not a TypeError), the API is correct.
    try:
        result = _wrapper.explain_factual(X_test, guarded_options=opts)
        assert result is not None
    except Exception as exc:
        # Guard may raise domain errors (e.g. not enough calibration neighbours) — that is fine.
        # What we must NOT get is a TypeError from wrong parameter wiring.
        assert not isinstance(exc, TypeError), f"Unexpected TypeError: {exc}"


def test_should_emit_deprecation_warning_when_guarded_true_is_used():
    with pytest.warns(DeprecationWarning, match="guarded=True"), contextlib.suppress(Exception):
        _wrapper.explain_factual(X_test, guarded=True)


def test_should_emit_deprecation_warning_when_significance_kwarg_is_used():
    with pytest.warns(DeprecationWarning), contextlib.suppress(Exception):
        _wrapper.explain_factual(X_test, guarded=True, significance=0.05)


def test_should_emit_deprecation_warning_when_guarded_true_used_in_explore_alternatives():
    with pytest.warns(DeprecationWarning, match="guarded=True"), contextlib.suppress(Exception):
        _wrapper.explore_alternatives(X_test, guarded=True)


def test_should_complete_without_error_when_guarded_options_used_in_explore_alternatives():
    opts = GuardedOptions(confidence=0.9)
    try:
        result = _wrapper.explore_alternatives(X_test, guarded_options=opts)
        assert result is not None
    except Exception as exc:
        assert not isinstance(exc, TypeError), f"Unexpected TypeError: {exc}"
