from __future__ import annotations

from types import SimpleNamespace

from calibrated_explanations.perf import CacheConfig, ParallelConfig, PerfFactory, from_config


def test_perf_factory_defaults_produce_disabled_primitives() -> None:
    factory = PerfFactory(cache=CacheConfig(), parallel=ParallelConfig())
    cache = factory.make_cache()
    executor = factory.make_parallel_executor(cache)
    assert cache.enabled is False
    assert executor.map(lambda x: x + 1, [1, 2]) == [2, 3]


def test_perf_factory_cache_eviction_behavior() -> None:
    factory = PerfFactory(
        cache=CacheConfig(enabled=True, max_items=2),
        parallel=ParallelConfig(),
    )
    cache = factory.make_cache()
    cache.set(stage="predict", parts=[("idx", 1)], value={"payload": 1})
    cache.set(stage="predict", parts=[("idx", 2)], value={"payload": 2})
    cache.set(stage="predict", parts=[("idx", 3)], value={"payload": 3})
    assert cache.get(stage="predict", parts=[("idx", 1)]) is None


def test_from_config_builds_factory() -> None:
    cfg = SimpleNamespace(
        perf_cache_enabled=True,
        perf_cache_max_items=4,
        perf_parallel_enabled=True,
        perf_parallel_backend="threads",
        perf_parallel_granularity="instance",
    )
    factory = from_config(cfg)
    cache = factory.make_cache()
    assert cache.enabled is True
    assert cache.config.max_items == 4
    executor = factory.make_parallel_executor(cache)
    assert executor.config.strategy == "threads"
    assert executor.config.granularity == "instance"


def test_parallel_config_parses_granularity_env(monkeypatch) -> None:
    monkeypatch.setenv("CE_PARALLEL", "enable,granularity=instance")
    cfg = ParallelConfig.from_env(ParallelConfig())
    assert cfg.enabled is True
    assert cfg.granularity == "instance"
