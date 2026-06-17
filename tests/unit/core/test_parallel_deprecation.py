"""Tests for ParallelConfig strategy='auto' deprecation (ADR-004 Gap 1)."""

from __future__ import annotations

import pytest

from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


def test_auto_strategy_deprecation_fires_when_enabled():
    """strategy='auto' with enabled=True must emit DeprecationWarning."""
    config = ParallelConfig(strategy="auto", enabled=True)
    executor = ParallelExecutor(config)
    with pytest.warns(DeprecationWarning, match=r"strategy.*auto"):
        executor.resolve_strategy()


def test_auto_strategy_deprecation_silent_when_disabled(recwarn):
    """strategy='auto' with enabled=False must NOT emit DeprecationWarning."""
    config = ParallelConfig(strategy="auto", enabled=False)
    executor = ParallelExecutor(config)
    executor.resolve_strategy()
    dep_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, DeprecationWarning)
        and "strategy" in str(w.message)
        and "auto" in str(w.message)
    ]
    assert not dep_warnings


def test_explicit_strategy_no_deprecation(recwarn):
    """Explicit strategy must NOT emit the auto-strategy DeprecationWarning."""
    for strategy in ("sequential", "threads"):
        config = ParallelConfig(strategy=strategy, enabled=True)
        executor = ParallelExecutor(config)
        executor.resolve_strategy()
    dep_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, DeprecationWarning)
        and "strategy" in str(w.message)
        and "auto" in str(w.message)
    ]
    assert not dep_warnings
