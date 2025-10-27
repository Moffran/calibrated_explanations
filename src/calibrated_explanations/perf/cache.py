"""Cache primitives for the calibrated explanations performance layer.

The cache implementation shipped in the first scaffolding pass only provided a
small ``OrderedDict`` based LRU structure.  ADR-003 requires considerably more
behaviour:

* Namespaced, versioned keys so multiple callers can safely share the cache.
* Optional TTL and memory budgets in addition to the entry-count limit.
* Thread-safety and fork-awareness for opt-in multiprocessing scenarios.
* Lightweight telemetry counters so staging environments can validate impact.

This module fulfils those requirements while remaining intentionally small.  It
avoids external dependencies and keeps the public surface limited to the pieces
used by the explainer runtime and documentation examples.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass as _dataclass
from dataclasses import field
from hashlib import sha256
from time import monotonic
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    TypeVar,
)

import numpy as np

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 10):
    dataclass = _dataclass
else:  # pragma: no cover - exercised in older Python versions via CI

    def dataclass(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("slots", None)
        return _dataclass(*args, **kwargs)


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

TelemetryCallback = Callable[[str, Mapping[str, Any]], None]


def _default_size_estimator(value: Any) -> int:
    """Best-effort size estimator used for memory budgets.

    ``sys.getsizeof`` is avoided because it dramatically underestimates numpy
    arrays.  Instead, rely on ``nbytes`` when available and fall back to a
    conservative constant.
    """

    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - extremely defensive
            logger.debug("Failed to read nbytes attribute for %s: %s", type(value).__name__, exc)
    if hasattr(value, "__array_interface__"):
        try:
            view = np.asarray(value)
            return int(view.nbytes)
        except Exception as exc:  # pragma: no cover - extremely defensive
            logger.debug("Failed to coerce array interface for %s: %s", type(value).__name__, exc)
    # Fallback constant that biases towards early eviction instead of OOM
    return 256


def _hash_part(part: Any) -> Hashable:
    """Normalise cache key parts to stable, hashable values."""

    if part is None:
        return None
    if isinstance(part, (str, bytes, int, float, bool, tuple)):
        return part
    if isinstance(part, np.ndarray):
        return ("nd", part.shape, part.dtype.str, sha256(part.view(np.uint8)).hexdigest())
    if isinstance(part, (list, set, frozenset)):
        return tuple(sorted(_hash_part(item) for item in part))
    if isinstance(part, Mapping):
        return tuple(sorted((k, _hash_part(v)) for k, v in part.items()))
    # As a last resort, hash the ``repr`` – this keeps keys deterministic
    return ("repr", repr(part))


def make_key(namespace: str, version: str, parts: Iterable[Any]) -> Tuple[Hashable, ...]:
    """Create a namespaced, versioned cache key.

    ``namespace`` and ``version`` isolate unrelated callers sharing a cache
    instance, while ``parts`` encodes the payload specific to the current
    computation.
    """

    return (namespace, version, *(_hash_part(part) for part in parts))


@dataclass(slots=True)
class CacheEntry(Generic[V]):
    """Internal helper capturing the cached value and eviction metadata."""

    value: V
    expires_at: float | None
    cost: int


@dataclass(slots=True)
class CacheMetrics:
    """Telemetry counters aggregated by :class:`LRUCache`."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0
    resets: int = 0

    def snapshot(self) -> Mapping[str, int]:
        """Return a dictionary suitable for logging or JSON serialisation."""

        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "resets": self.resets,
        }


@dataclass
class CacheConfig:
    """Configuration settings for the calibrator cache."""

    enabled: bool = False
    namespace: str = "calibrator"
    version: str = "v1"
    max_items: int = 512
    max_bytes: int | None = 32 * 1024 * 1024
    ttl_seconds: float | None = None
    telemetry: TelemetryCallback | None = None
    size_estimator: Callable[[Any], int] = _default_size_estimator

    @classmethod
    def from_env(cls, base: "CacheConfig | None" = None) -> "CacheConfig":
        """Merge ``CE_CACHE`` overrides with ``base`` defaults."""

        cfg = CacheConfig(**(base.__dict__ if base is not None else {}))
        raw = os.getenv("CE_CACHE")
        if not raw:
            return cfg
        tokens = [segment.strip() for segment in raw.split(",") if segment.strip()]
        # ``CE_CACHE=1`` or ``on`` enables the cache with defaults
        if len(tokens) == 1 and tokens[0].lower() in {"1", "true", "on", "yes"}:
            cfg.enabled = True
            return cfg
        for token in tokens:
            if token.lower() in {"0", "off", "false", "no"}:
                cfg.enabled = False
                continue
            if token.startswith("namespace="):
                cfg.namespace = token.split("=", 1)[1]
                continue
            if token.startswith("version="):
                cfg.version = token.split("=", 1)[1]
                continue
            if token.startswith("max_items="):
                cfg.max_items = max(1, int(token.split("=", 1)[1]))
                continue
            if token.startswith("max_bytes="):
                cfg.max_bytes = max(1, int(token.split("=", 1)[1]))
                continue
            if token.startswith("ttl="):
                cfg.ttl_seconds = max(0.0, float(token.split("=", 1)[1]))
                continue
            if token == "enable":  # noqa: S105  # nosec B105 - configuration toggle keyword
                cfg.enabled = True
        return cfg


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with TTL and memory budget support."""

    def __init__(
        self,
        *,
        namespace: str,
        version: str,
        max_items: int,
        max_bytes: int | None,
        ttl_seconds: float | None,
        telemetry: TelemetryCallback | None,
        size_estimator: Callable[[Any], int],
    ) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive when provided")
        self.namespace = namespace
        self.version = version
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._size_estimator = size_estimator
        self._telemetry = telemetry
        self._store: MutableMapping[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = 0
        self.metrics = CacheMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: K, default: V | None = None) -> V | None:
        """Return the cached value for ``key`` if present and not expired."""

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.metrics.misses += 1
                self._emit("cache_miss", {"key": key})
                return default
            if entry.expires_at is not None and entry.expires_at < monotonic():
                self.metrics.expirations += 1
                self.metrics.misses += 1
                self._evict_no_lock(key)
                self._emit("cache_expired", {"key": key})
                return default
            self._store.move_to_end(key)
            self.metrics.hits += 1
            self._emit("cache_hit", {"key": key})
            return entry.value

    def set(self, key: K, value: V) -> None:
        """Store ``value`` under ``key`` honouring eviction policies."""

        cost = max(0, self._safe_estimate(value))
        if self.max_bytes is not None and cost > self.max_bytes:
            # Value is larger than the entire cache budget – skip storing.
            self.metrics.misses += 1
            self._emit("cache_skip", {"reason": "oversize", "key": key, "cost": cost})
            return
        expires_at = None
        if self.ttl_seconds is not None:
            expires_at = monotonic() + self.ttl_seconds
        entry = CacheEntry(value=value, expires_at=expires_at, cost=cost)
        with self._lock:
            if key in self._store:
                existing = self._store[key]
                self._bytes -= existing.cost
            self._store[key] = entry
            self._store.move_to_end(key)
            self._bytes += cost
            self.metrics.sets += 1
            self._emit("cache_store", {"key": key, "cost": cost})
            self._shrink_if_needed()

    def __contains__(self, key: K) -> bool:  # pragma: no cover - trivial
        with self._lock:
            return key in self._store

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------
    def forksafe_reset(self) -> None:
        """Reset cache state after ``fork`` to avoid cross-process leakage."""

        with self._lock:
            self._store.clear()
            self._bytes = 0
            self.metrics.resets += 1
            self._emit("cache_reset", {"reason": "forksafe"})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _safe_estimate(self, value: Any) -> int:
        try:
            return int(self._size_estimator(value))
        except Exception:  # pragma: no cover - defensive fall-back
            return 0

    def _shrink_if_needed(self) -> None:
        if self.max_bytes is None and len(self._store) <= self.max_items:
            return
        while len(self._store) > self.max_items:
            self._evict_oldest()
        if self.max_bytes is None:
            return
        while self._bytes > self.max_bytes and self._store:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        key, _ = next(iter(self._store.items()))
        self._evict_no_lock(key)

    def _evict_no_lock(self, key: K) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._bytes -= entry.cost
            self.metrics.evictions += 1
            self._emit("cache_evict", {"key": key, "cost": entry.cost})

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        if self._telemetry is None:
            return
        try:  # pragma: no cover - telemetry is optional best effort
            self._telemetry(event, {"namespace": self.namespace, **payload})
        except Exception as exc:
            logger.debug("Telemetry callback failed for %s: %s", event, exc)


@dataclass
class CalibratorCache(Generic[V]):
    """Namespace-aware cache wrapper used by the explainer runtime."""

    config: CacheConfig
    _cache: LRUCache[Tuple[Hashable, ...], V] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.config.enabled:
            self._cache = LRUCache(
                namespace=self.config.namespace,
                version=self.config.version,
                max_items=self.config.max_items,
                max_bytes=self.config.max_bytes,
                ttl_seconds=self.config.ttl_seconds,
                telemetry=self.config.telemetry,
                size_estimator=self.config.size_estimator,
            )
        else:
            self._cache = None

    @property
    def enabled(self) -> bool:
        return self._cache is not None

    @property
    def metrics(self) -> CacheMetrics:
        if self._cache is None:
            return CacheMetrics()
        return self._cache.metrics

    def get(self, *, stage: str, parts: Iterable[Any]) -> V | None:
        if self._cache is None:
            return None
        key = make_key(self.config.namespace, f"{self.config.version}:{stage}", parts)
        return self._cache.get(key)

    def set(self, *, stage: str, parts: Iterable[Any], value: V) -> None:
        if self._cache is None:
            return
        key = make_key(self.config.namespace, f"{self.config.version}:{stage}", parts)
        self._cache.set(key, value)

    def compute(self, *, stage: str, parts: Iterable[Any], fn: Callable[[], V]) -> V:
        cached = self.get(stage=stage, parts=parts)
        if cached is not None:
            return cached
        value = fn()
        self.set(stage=stage, parts=parts, value=value)
        return value

    def forksafe_reset(self) -> None:
        if self._cache is not None:
            self._cache.forksafe_reset()


__all__ = [
    "CacheConfig",
    "CacheMetrics",
    "CalibratorCache",
    "LRUCache",
    "TelemetryCallback",
    "make_key",
]
