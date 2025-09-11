"""Performance foundations (feature-flagged): cache and parallel backends.

These utilities are scaffolded for Phase 3 (ADR-003/ADR-004) and are not wired
into the main flows by default. They can be imported and used explicitly, and
will be integrated behind feature flags in future steps.

This module also provides a small, conservative factory (disabled by default)
that callers can instantiate from an :class:`ExplainerConfig` to opt-in to the
LRU cache and parallel backend. The factory keeps the default behavior
disabled so importing this module does not change runtime semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cache import LRUCache, make_key  # noqa: F401
from .parallel import JoblibBackend, ParallelBackend, sequential_map  # noqa: F401


@dataclass
class PerfFactory:
    """Small factory to create perf primitives behind feature flags.

    Usage: callers may construct this from an ``ExplainerConfig`` and then
    call ``make_cache()`` and ``make_parallel_backend()``. When the
    corresponding "enabled" flag is False, the factory returns ``None`` or a
    safe sequential backend to avoid changing behavior.
    """

    cache_enabled: bool = False
    cache_max_items: int = 128
    parallel_enabled: bool = False
    parallel_backend: str = "auto"

    def make_cache(self) -> Optional[LRUCache]:
        """Return an LRUCache if enabled, otherwise ``None``."""
        if not self.cache_enabled:
            return None
        return LRUCache(max_items=self.cache_max_items)

    def make_parallel_backend(self) -> ParallelBackend:
        """Return a ParallelBackend implementation based on the factory settings.

        If parallel is disabled, a simple sequential backend is returned to
        preserve existing single-threaded behaviour.
        """
        if not self.parallel_enabled:

            class _SeqBackend:
                def map(self, fn, items, *, workers=None):
                    return sequential_map(fn, items)

            return _SeqBackend()
        if self.parallel_backend == "joblib":
            return JoblibBackend()
        # "auto" or unknown → prefer JoblibBackend which falls back to sequential
        return JoblibBackend()


def from_config(cfg) -> PerfFactory:
    """Create a PerfFactory from an ExplainerConfig-like object.

    The function is intentionally permissive and only reads the perf-related
    attributes. It does no side-effects.
    """
    return PerfFactory(
        cache_enabled=getattr(cfg, "perf_cache_enabled", False),
        cache_max_items=getattr(cfg, "perf_cache_max_items", 128),
        parallel_enabled=getattr(cfg, "perf_parallel_enabled", False),
        parallel_backend=getattr(cfg, "perf_parallel_backend", "auto"),
    )


__all__ = [
    "LRUCache",
    "make_key",
    "ParallelBackend",
    "JoblibBackend",
    "sequential_map",
    # factory exports
    "PerfFactory",
    "from_config",
]
