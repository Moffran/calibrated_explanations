"""Performance primitives exposed to the rest of the code base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cache import CalibratorCache, CacheConfig, CacheMetrics, TelemetryCallback
from .parallel import ParallelConfig, ParallelExecutor, ParallelMetrics


@dataclass
class PerfFactory:
    """Factory bundling cache and parallel primitives behind feature flags."""

    cache: CacheConfig
    parallel: ParallelConfig

    def make_cache(self) -> CalibratorCache[Any]:
        return CalibratorCache(self.cache)

    def make_parallel_executor(
        self, cache: CalibratorCache[Any] | None = None
    ) -> ParallelExecutor:
        return ParallelExecutor(self.parallel, cache=cache)

    # Backwards compatible name retained for earlier scaffolding usage
    def make_parallel_backend(
        self, cache: CalibratorCache[Any] | None = None
    ) -> ParallelExecutor:
        return self.make_parallel_executor(cache=cache)


def from_config(cfg: Any) -> PerfFactory:
    """Build a :class:`PerfFactory` from an ``ExplainerConfig`` like object."""

    cache_cfg = CacheConfig(
        enabled=getattr(cfg, "perf_cache_enabled", False),
        namespace=getattr(cfg, "perf_cache_namespace", "calibrator"),
        version=getattr(cfg, "perf_cache_version", "v1"),
        max_items=getattr(cfg, "perf_cache_max_items", 512),
        max_bytes=getattr(cfg, "perf_cache_max_bytes", 32 * 1024 * 1024),
        ttl_seconds=getattr(cfg, "perf_cache_ttl", None),
        telemetry=getattr(cfg, "perf_telemetry", None),
    )
    cache_cfg = CacheConfig.from_env(cache_cfg)
    parallel_cfg = ParallelConfig(
        enabled=getattr(cfg, "perf_parallel_enabled", False),
        strategy=getattr(cfg, "perf_parallel_backend", "auto"),
        max_workers=getattr(cfg, "perf_parallel_workers", None),
        min_batch_size=getattr(cfg, "perf_parallel_min_batch", 32),
        telemetry=getattr(cfg, "perf_telemetry", None),
    )
    parallel_cfg = ParallelConfig.from_env(parallel_cfg)
    return PerfFactory(cache=cache_cfg, parallel=parallel_cfg)


__all__ = [
    "CalibratorCache",
    "CacheConfig",
    "CacheMetrics",
    "ParallelConfig",
    "ParallelExecutor",
    "ParallelMetrics",
    "PerfFactory",
    "TelemetryCallback",
    "from_config",
]

