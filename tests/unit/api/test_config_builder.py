from __future__ import annotations

from calibrated_explanations.api import config as config_module


class DummyModel:
    pass


def test_builder_propagates_configuration_fields():
    builder = config_module.ExplainerBuilder(DummyModel())
    cfg = (
        builder.task("regression")
        .low_high_percentiles((10, 90))
        .threshold(0.2)
        .preprocessor("prep")
        .auto_encode(True)
        .unseen_category_policy("ignore")
        .parallel_workers(4)
        .perf_cache(True, max_items=42)
        .perf_parallel(True, backend="joblib", granularity="instance")
        .perf_telemetry(lambda *args, **kwargs: None)
        .build_config()
    )

    assert cfg.task == "regression"
    assert cfg.low_high_percentiles == (10, 90)
    assert cfg.threshold == 0.2
    assert cfg.preprocessor == "prep"
    assert cfg.auto_encode is True
    assert cfg.unseen_category_policy == "ignore"
    assert cfg.parallel_workers == 4
    assert cfg.perf_cache_enabled is True
    assert cfg.perf_cache_max_items == 42
    assert cfg.perf_parallel_enabled is True
    assert cfg.perf_parallel_backend == "joblib"
    assert cfg.perf_parallel_granularity == "instance"
    assert hasattr(cfg, "_perf_factory")
    assert cfg.perf_telemetry is not None


def test_builder_allows_overriding_all_perf_cache_and_parallel_fields():
    builder = config_module.ExplainerBuilder(DummyModel())

    cfg = (
        builder.perf_cache(
            True,
            max_items=128,
            max_bytes=2048,
            namespace="alt",
            version="v9",
            ttl=12.5,
        )
        .perf_parallel(
            True,
            backend="threads",
            workers=8,
            min_batch=4,
            tiny_workload=12,
            granularity="instance",
        )
        .build_config()
    )

    assert cfg.perf_cache_enabled is True
    assert cfg.perf_cache_max_items == 128
    assert cfg.perf_cache_max_bytes == 2048
    assert cfg.perf_cache_namespace == "alt"
    assert cfg.perf_cache_version == "v9"
    assert cfg.perf_cache_ttl == 12.5
    assert cfg.perf_parallel_enabled is True
    assert cfg.perf_parallel_backend == "threads"
    assert cfg.perf_parallel_workers == 8
    assert cfg.perf_parallel_min_batch == 4
    assert cfg.perf_parallel_tiny_workload == 12
    assert cfg.perf_parallel_granularity == "instance"


def test_builder_swallows_perf_factory_errors(monkeypatch):
    builder = config_module.ExplainerBuilder(DummyModel())

    def boom(cfg):  # pragma: no cover - helper should not be traced
        raise RuntimeError("perf unavailable")

    monkeypatch.setattr(config_module, "_perf_from_config", boom)

    cfg = builder.build_config()

    assert cfg._perf_factory is None  # type: ignore[attr-defined]
