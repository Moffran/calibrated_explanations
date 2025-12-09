"""Deprecated shim forwarding to :mod:`calibrated_explanations.cache` and :mod:`calibrated_explanations.parallel`.

Per ADR-001 Stage 1b, cache and parallel concerns are now in dedicated packages.
This shim will be removed in v0.11.0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

warnings.warn(
    "The 'calibrated_explanations.perf' module is deprecated. "
    "Use 'calibrated_explanations.cache' and 'calibrated_explanations.parallel' instead. "
    "This shim will be removed in v0.11.0.",
    DeprecationWarning,
    stacklevel=2,
)

from ..cache import CacheConfig, CacheMetrics, CalibratorCache, TelemetryCallback  # noqa: E402
from ..parallel import ParallelConfig, ParallelExecutor, ParallelMetrics  # noqa: E402


@dataclass
class PerfFactory:  # pragma: no cover
    """Factory bundling cache and parallel primitives (deprecated)."""

    cache: CacheConfig
    parallel: ParallelConfig

    def make_cache(self) -> CalibratorCache[Any]:
        """Build a cache backend based on the stored configuration."""
        return CalibratorCache(self.cache)

    def make_parallel_executor(self, cache: CalibratorCache[Any] | None = None) -> ParallelExecutor:
        """Create a parallel executor wired to this factory's parallel config."""
        return ParallelExecutor(self.parallel, cache=cache)

    def make_parallel_backend(self, cache: CalibratorCache[Any] | None = None) -> ParallelExecutor:
        """Alias for :meth:`make_parallel_executor` for backwards compatibility."""
        return self.make_parallel_executor(cache=cache)


def from_config(cfg: Any) -> PerfFactory:  # pragma: no cover
    """Build a PerfFactory from config object (deprecated)."""
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
        granularity=getattr(cfg, "perf_parallel_granularity", "feature"),
        telemetry=getattr(cfg, "perf_telemetry", None),
    )
    parallel_cfg = ParallelConfig.from_env(parallel_cfg)
    return PerfFactory(cache=cache_cfg, parallel=parallel_cfg)


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Forward other attribute access to cache and parallel packages."""
    try:
        from .. import cache

        return getattr(cache, name)
    except AttributeError:
        from .. import parallel

        return getattr(parallel, name)


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
