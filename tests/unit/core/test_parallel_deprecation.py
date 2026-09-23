"""Tests for ParallelConfig strategy='auto' deprecation (ADR-004 Gap 1)."""

from __future__ import annotations

import pytest

from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


def test_auto_strategy_raises_configuration_error_when_enabled():
    """strategy='auto' with enabled=True must raise ConfigurationError in v1.0.0."""
    from calibrated_explanations.utils.exceptions import ConfigurationError

    config = ParallelConfig(strategy="auto", enabled=True)
    executor = ParallelExecutor(config)
    with pytest.raises(ConfigurationError, match="strategy.*auto"):
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


def test_entering_enabled_auto_executor_raises_instead_of_auto_selecting():
    """Entering an enabled 'auto' executor must not silently pick a backend (#218)."""
    from calibrated_explanations.utils.exceptions import ConfigurationError

    executor = ParallelExecutor(ParallelConfig(enabled=True, strategy="auto"))

    with pytest.raises(ConfigurationError, match="strategy='auto'"):
        executor.__enter__()

    assert executor.pool is None
    assert executor.active_strategy_name is None


def test_entering_disabled_auto_executor_is_a_no_op():
    """The 'auto' default stays valid while the executor is disabled."""
    executor = ParallelExecutor(ParallelConfig(enabled=False, strategy="auto"))

    with executor as entered:
        assert entered.pool is None
        assert entered.active_strategy_name is None


def test_internal_explain_runtime_executor_does_not_use_auto(monkeypatch):
    """The explain runtime's own executor must use an explicit strategy (#218)."""
    from types import SimpleNamespace

    from calibrated_explanations.core.explain.parallel_runtime import ExplainParallelRuntime

    monkeypatch.delenv("CE_PARALLEL", raising=False)

    runtime = ExplainParallelRuntime.from_explainer(SimpleNamespace())
    with runtime:
        active = runtime.executor.active_strategy_name

    assert runtime.executor.config.enabled is True
    assert runtime.executor.config.strategy == "sequential"
    assert active == "sequential"


def test_internal_explain_runtime_executor_keeps_explicit_env_strategy(monkeypatch):
    """An explicit CE_PARALLEL strategy is still honoured by the internal executor."""
    from types import SimpleNamespace

    from calibrated_explanations.core.explain.parallel_runtime import ExplainParallelRuntime

    monkeypatch.setenv("CE_PARALLEL", "threads")

    runtime = ExplainParallelRuntime.from_explainer(SimpleNamespace())

    assert runtime.executor.config.strategy == "threads"
