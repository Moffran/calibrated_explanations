from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from calibrated_explanations.perf import CacheConfig, ParallelConfig, PerfFactory, from_config
from calibrated_explanations.perf.parallel import ParallelExecutor


def test_perf_factory_respects_disabled_cache() -> None:
    factory = PerfFactory(cache=CacheConfig(enabled=False), parallel=ParallelConfig())
    cache = factory.make_cache()
    assert not cache.enabled


def test_parallel_executor_disabled_behaves_sequentially() -> None:
    executor = ParallelExecutor(ParallelConfig(enabled=False))
    assert executor.map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]


def test_parallel_executor_threads_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)
    executor = ParallelExecutor(cfg)

    calls: List[int] = []

    def record(x: int) -> int:
        calls.append(x)
        return x * 2

    result = executor.map(record, [1, 2, 3, 4])
    assert sorted(calls) == [1, 2, 3, 4]
    assert result == [2, 4, 6, 8]


def test_parallel_executor_fallback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ParallelConfig(enabled=True, strategy="processes", min_batch_size=1)
    executor = ParallelExecutor(cfg)

    def fail_strategy():
        raise RuntimeError("strategy setup failed")

    monkeypatch.setattr(executor, "_resolve_strategy", fail_strategy)

    result = executor.map(lambda x: x + 1, [1, 2])
    assert result == [2, 3]
    assert executor.metrics.fallbacks == 1


def test_from_config_reads_perf_attributes() -> None:
    cfg = SimpleNamespace(
        perf_cache_enabled=True,
        perf_cache_namespace="ns",
        perf_cache_version="v9",
        perf_cache_max_items=5,
        perf_parallel_enabled=True,
        perf_parallel_backend="threads",
        perf_parallel_workers=3,
    )
    factory = from_config(cfg)
    assert factory.cache.enabled is True
    assert factory.cache.max_items == 5
    assert factory.cache.namespace == "ns"
    assert factory.parallel.enabled is True
    assert factory.parallel.strategy == "threads"
    assert factory.parallel.max_workers == 3


def test_from_config_defaults() -> None:
    factory = from_config(SimpleNamespace())
    assert factory.cache.enabled is False
    assert factory.cache.max_items == 512
    assert factory.parallel.enabled is False
    assert factory.parallel.strategy == "auto"
