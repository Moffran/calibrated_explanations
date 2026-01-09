"""Deprecated shim forwarding to :mod:`calibrated_explanations.cache`.

This module previously contained the cache implementation. Per ADR-001 and
ADR-003 the canonical implementation now lives in
``calibrated_explanations.cache``.

Compatibility notes
-------------------
* Importing this module emits a :class:`DeprecationWarning`.
* The shim will be removed after v1.1.0 once downstream callers migrate.
* All attributes are forwarded to :mod:`calibrated_explanations.cache.cache`
  to preserve monkeypatch targets used by existing tests and plugins.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..cache import cache as _cache

warnings.warn(
    "The 'calibrated_explanations.perf.cache' module is deprecated. "
    "Use 'calibrated_explanations.cache' instead. "
    "This shim will be removed after v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export canonical symbols
CacheConfig = _cache.CacheConfig
CacheMetrics = _cache.CacheMetrics
CalibratorCache = _cache.CalibratorCache
LRUCache = _cache.LRUCache
TelemetryCallback = _cache.TelemetryCallback
make_key = _cache.make_key

# Expose implementation details so legacy monkeypatch targets continue to work
default_size_estimator = _cache.default_size_estimator
_hash_part = _cache._hash_part
hash_part = _cache.hash_part
monotonic = _cache.monotonic


def __getattr__(name: str) -> Any:  # pragma: no cover - thin shim
    return getattr(_cache, name)


__all__ = [
    "CacheConfig",
    "CacheMetrics",
    "CalibratorCache",
    "LRUCache",
    "TelemetryCallback",
    "make_key",
]
