"""Performance primitives exposed to the rest of the code base.

DEPRECATED SHIM: This module is a compatibility shim for backward compatibility.
As part of ADR-001 (Stage 1b), cache and parallel concerns have been split into
dedicated top-level packages: calibrated_explanations.cache and
calibrated_explanations.parallel.

This shim will be removed in v1.1.0. Migration guide:
- Old: from calibrated_explanations.perf import CalibratorCache
- New: from calibrated_explanations.cache import CalibratorCache
"""

from __future__ import annotations

import warnings

# Emit deprecation warning
warnings.warn(
    "The 'calibrated_explanations.perf' module is deprecated. "
    "Use 'calibrated_explanations.cache' and 'calibrated_explanations.parallel' instead. "
    "This shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new packages for backward compatibility
from ..cache import CacheConfig, CacheMetrics, CalibratorCache, TelemetryCallback
from ..parallel import ParallelConfig, ParallelExecutor, ParallelMetrics

# Keep factory function for compatibility
from dataclasses import dataclass
from typing import Any


@dataclass
class PerfFactory:
    """Factory bundling cache and parallel primitives behind feature flags.
    
    DEPRECATED: This factory is maintained for backward compatibility only.
    Use the cache and parallel packages directly instead.
    """

    cache: CacheConfig
    parallel: ParallelConfig

    def make_cache(self) -> CalibratorCache[Any]:
        """Return a cache instance configured with the factory defaults."""
        return CalibratorCache(self.cache)

    def make_parallel_executor(self, cache: CalibratorCache[Any] | None = None) -> ParallelExecutor:
        """Return a parallel executor configured with the factory defaults."""
        return ParallelExecutor(self.parallel, cache=cache)

    # Backwards compatible name retained for earlier scaffolding usage
    def make_parallel_backend(self, cache: CalibratorCache[Any] | None = None) -> ParallelExecutor:
        """Return a parallel executor using the legacy backend name."""
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
        granularity=getattr(cfg, "perf_parallel_granularity", "feature"),
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
