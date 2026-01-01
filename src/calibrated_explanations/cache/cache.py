"""Cache primitives for the calibrated explanations performance layer.

ADR-003 specifies the caching strategy with the following requirements:

* Namespaced, versioned keys so multiple callers can safely share the cache.
* LRU eviction policy using the `cachetools` library.
* Optional TTL and memory budgets in addition to the entry-count limit.
* Thread-safety and fork-awareness for opt-in multiprocessing scenarios.
* Lightweight telemetry counters so staging environments can validate impact.

The implementation uses `cachetools.TTLCache` as the core backend with custom
lifecycle management for versioning, flush/reset operations, and telemetry.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import warnings
from dataclasses import dataclass as _dataclass
from dataclasses import field
from hashlib import blake2b
from time import monotonic
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Tuple,
    TypeVar,
)

try:  # pragma: no cover - behaviour varies by environment
    import cachetools

    _HAVE_CACHETOOLS = True
except:  # noqa: E722
    if not isinstance(sys.exc_info()[1], Exception):
        raise
    cachetools = None  # type: ignore
    _HAVE_CACHETOOLS = False
    # Visible notification: cachetools missing, falling back to minimal backend
    _logger = logging.getLogger(__name__)
    _logger.info("cachetools not available; falling back to minimal LRU/TTL cache backend")
    warnings.warn(
        "Cache backend fallback: using minimal in-package LRU/TTL implementation due to missing 'cachetools'",
        UserWarning,
        stacklevel=2,
    )
    # Provide a tiny, well-tested fallback for environments where
    # `cachetools` is not installed (CI minimal images). The fallback
    # implements the minimal API used by this module: `LRUCache` and
    # `TTLCache` supporting `maxsize`, iteration, get/set, pop and clear.
    import time
    from collections import OrderedDict

    class _FallbackBase:
        pass

    class LRUCache(OrderedDict):
        """A minimal LRU cache compatible with cachetools.LRUCache.

        Behaviour:
        - `maxsize` limits number of entries and evicts least-recently-used.
        - Accessing an entry moves it to the end (most-recently-used).
        """

        def __init__(self, maxsize: int):
            super().__init__()
            self.maxsize = int(maxsize)

        def __getitem__(self, key):
            """Retrieve the value for *key* and mark it as recently used.

            This method moves the key to the end to record recent usage.
            Raises KeyError if the key is not present.
            """
            value = super().__getitem__(key)
            # mark as recently used
            # In some Python versions (e.g. 3.10), OrderedDict.popitem
            # may trigger __getitem__ after the key is removed.
            with contextlib.suppress(KeyError):  # pragma: no cover
                self.move_to_end(key)
            return value

        def get(self, key, default=None):
            """Return the value for *key* if present, otherwise *default*.

            If the key is present this marks it as recently used.
            """
            if key in self:
                return self.__getitem__(key)
            return default

        def __setitem__(self, key, value):
            """Set *key* to *value*, update recency, and evict if needed.

            Existing keys are moved to the end (most-recently-used). When the
            cache exceeds ``maxsize`` the least-recently-used item is evicted.
            """
            if key in self:
                # overwrite and mark as recent
                super().__setitem__(key, value)
                self.move_to_end(key)
                return
            super().__setitem__(key, value)
            # Evict least-recently-used if over capacity
            while self.maxsize is not None and len(self) > self.maxsize:
                self.popitem(last=False)

    class TTLCache(LRUCache):
        """A minimal TTL cache that stores expiry timestamps alongside values."""

        def __init__(self, maxsize: int, ttl: float):
            super().__init__(maxsize=maxsize)
            self._ttl = float(ttl)
            # store mapping key -> expiry
            self._expiries = {}

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self._expiries[key] = time.time() + self._ttl

        def __getitem__(self, key):
            if key in self._expiries and time.time() >= self._expiries.get(key, 0):
                # expired
                try:
                    # Use OrderedDict.__delitem__ to avoid recursion if pop calls __getitem__
                    OrderedDict.__delitem__(self, key)
                except KeyError:
                    pass
                finally:
                    self._expiries.pop(key, None)
                raise KeyError(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            try:
                return self.__getitem__(key)
            except KeyError:
                return default

        def pop(self, key, *args):
            self._expiries.pop(key, None)
            return super().pop(key, *args)

        def clear(self):
            self._expiries.clear()
            super().clear()

    # Expose compatible names expected elsewhere in the module
    class _CacheModuleShim:
        LRUCache = LRUCache
        TTLCache = TTLCache

    cachetools = _CacheModuleShim()
import numpy as np

logger = logging.getLogger(__name__)

# Export monotonic to support legacy shims/tests that reference
# `calibrated_explanations.cache.cache.monotonic`.
# This keeps a small compatibility surface without changing behaviour.
monotonic = monotonic

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

# Sentinel for distinguishing None values from missing keys
_NONE_SENTINEL = object()


def default_size_estimator(value: Any) -> int:
    """Best-effort size estimator used for memory budgets.

    ``sys.getsizeof`` is avoided because it dramatically underestimates numpy
    arrays.  Instead, rely on ``nbytes`` when available and fall back to a
    conservative constant.
    """
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)  # type: ignore[arg-type]
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            logger.debug(
                "Failed to read nbytes attribute for %s: %s",
                type(value).__name__,
                sys.exc_info()[1],
            )
    if hasattr(value, "__array_interface__"):
        try:
            view = np.asarray(value)
            return int(view.nbytes)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            logger.debug(
                "Failed to coerce array interface for %s: %s",
                type(value).__name__,
                sys.exc_info()[1],
            )
    # Fallback constant that biases towards early eviction instead of OOM
    return 256


def hash_part(part: Any) -> Hashable:
    """Normalise cache key parts to stable, hashable values using blake2b.

    Uses blake2b for consistency with ADR-003 specification while maintaining
    deterministic key generation across runs.
    """
    if part is None:
        return None
    if isinstance(part, (str, bytes, int, float, bool, tuple)):
        return part
    if isinstance(part, np.ndarray):
        # Use blake2b to hash array content for stable keys
        digest = blake2b(part.view(np.uint8), digest_size=16).hexdigest()
        return ("nd", part.shape, part.dtype.str, digest)
    if isinstance(part, (list, set, frozenset)):
        return tuple(sorted(hash_part(item) for item in part))
    if isinstance(part, Mapping):
        return tuple(sorted((k, hash_part(v)) for k, v in part.items()))
    # As a last resort, hash the ``repr`` - this keeps keys deterministic
    return ("repr", repr(part))


# Backwards-compatible alias
_hash_part = hash_part


def make_key(namespace: str, version: str, parts: Iterable[Any]) -> Tuple[Hashable, ...]:
    """Create a namespaced, versioned cache key.

    ``namespace`` and ``version`` isolate unrelated callers sharing a cache
    instance, while ``parts`` encodes the payload specific to the current
    computation.
    """
    return (namespace, version, *(hash_part(part) for part in parts))


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
    """Configuration settings for the calibrator cache.

    Parameters
    ----------
    enabled : bool
        Whether caching is enabled. Default: False (opt-in).
    namespace : str
        Cache namespace for isolation. Default: "calibrator".
    version : str
        Version tag for invalidation on config/code changes. Default: "v1".
        Bump this tag to invalidate all cached entries in the namespace.
    max_items : int
        Maximum number of cached entries. Default: 512.
    max_bytes : int | None
        Maximum memory budget in bytes. Default: 32 MB.
    ttl_seconds : float | None
        Time-to-live for cache entries in seconds. Default: None (no expiry).
    telemetry : TelemetryCallback | None
        Optional callback for cache events. Default: None.
    size_estimator : Callable
        Function to estimate value size. Default: uses nbytes or getsizeof.
    """

    enabled: bool = False
    namespace: str = "calibrator"
    version: str = "v1"
    max_items: int = 512
    max_bytes: int | None = 32 * 1024 * 1024
    ttl_seconds: float | None = None
    telemetry: TelemetryCallback | None = None
    size_estimator: Callable[[Any], int] = default_size_estimator

    @classmethod
    def from_env(cls, base: "CacheConfig | None" = None) -> "CacheConfig":
        """Merge ``CE_CACHE`` overrides with ``base`` defaults."""
        cfg = CacheConfig(**(base.__dict__ if base is not None else {}))
        raw = os.getenv("CE_CACHE")
        if not raw:
            return cfg
        tokens = [segment.strip() for segment in raw.split(",") if segment.strip()]
        # ``CE_CACHE=1`` or ``on`` enables the cache with defaults
        enabled_labels = {"1", "true", "on", "yes", "enable"}
        if len(tokens) == 1 and tokens[0].lower() in enabled_labels:
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
            if token in enabled_labels:  # noqa: S105  # nosec B105 - configuration toggle keyword
                cfg.enabled = True
        return cfg


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache using cachetools.TTLCache with memory budget support.

    Wraps cachetools.TTLCache (or TTLCache without TTL when ttl_seconds is None)
    with custom memory accounting and telemetry emission.
    """

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
        """Initialize cache with cachetools backend."""
        from ..utils.exceptions import ValidationError

        if max_items <= 0:
            raise ValidationError(
                "max_items must be positive",
                details={"param": "max_items", "value": max_items, "requirement": "positive"},
            )
        if max_bytes is not None and max_bytes <= 0:
            raise ValidationError(
                "max_bytes must be positive when provided",
                details={"param": "max_bytes", "value": max_bytes, "requirement": "positive"},
            )
        self.namespace = namespace
        self.version = version
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._size_estimator = size_estimator
        self._telemetry = telemetry

        # Use cachetools.TTLCache if TTL is specified, otherwise LRUCache
        if ttl_seconds is not None:
            self._store: cachetools.Cache[K, V] = cachetools.TTLCache(
                maxsize=max_items, ttl=ttl_seconds
            )
        else:
            self._store = cachetools.LRUCache(maxsize=max_items)

        self._lock = threading.RLock()
        self._bytes = 0
        self.metrics = CacheMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: K, default: V | None = None) -> V | None:
        """Return the cached value for ``key`` if present and not expired."""
        with self._lock:
            try:
                value = self._store.get(key, _NONE_SENTINEL)
                if value is _NONE_SENTINEL:
                    self.metrics.misses += 1
                    self._emit("cache_miss", {"key": key})
                    return default
                # Unwrap None values
                if value is None or (isinstance(value, tuple) and value == (_NONE_SENTINEL,)):
                    unwrapped = None
                else:
                    unwrapped = value
                self.metrics.hits += 1
                self._emit("cache_hit", {"key": key})
                return unwrapped
            except KeyError:
                self.metrics.misses += 1
                self._emit("cache_miss", {"key": key})
                return default

    def set(self, key: K, value: V) -> None:
        """Store ``value`` under ``key`` honouring eviction policies.

        Note: None values are wrapped to distinguish them from missing keys.
        """
        # Wrap None values to distinguish from missing keys
        stored_value = (_NONE_SENTINEL,) if value is None else value
        cost = max(0, self._safe_estimate(stored_value))
        if self.max_bytes is not None and cost > self.max_bytes:
            # Value is larger than the entire cache budget – skip storing.
            self.metrics.misses += 1
            self._emit("cache_skip", {"reason": "oversize", "key": key, "cost": cost})
            return

        with self._lock:
            # Track the old keys before adding the new entry
            old_keys = set(self._store.keys())

            # Track memory for existing entry if present
            if key in self._store:
                existing = self._store[key]
                old_cost = max(0, self._safe_estimate(existing))
                self._bytes -= old_cost

            # Evict by size if needed BEFORE adding the new entry
            # to ensure we don't exceed the budget
            if self.max_bytes is not None:
                # Make room for the new entry
                while self._bytes + cost > self.max_bytes and self._store:
                    self._evict_oldest()

            # Add the new entry (cachetools may auto-evict if max_items reached)
            self._store[key] = stored_value
            self._bytes += cost

            # Detect and track any auto-evictions by cachetools
            new_keys = set(self._store.keys())
            evicted_by_cachetools = old_keys - new_keys - {key}
            for evicted_key in evicted_by_cachetools:
                self.metrics.evictions += 1
                self._emit("cache_evict", {"key": evicted_key, "reason": "auto_by_cachetools"})

            self.metrics.sets += 1
            self._emit("cache_store", {"key": key, "cost": cost})

    def __contains__(self, key: K) -> bool:  # pragma: no cover - trivial
        """Return True when *key* is present in the cache."""
        with self._lock:
            return key in self._store

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of cached entries."""
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
        """Best-effort estimate of object size, swallowing estimator errors."""
        try:
            return int(self._size_estimator(value))
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            return 0

    def _evict_oldest(self) -> None:
        """Remove the least-recently-used entry."""
        # cachetools.LRUCache maintains order; peek at the first key
        try:
            key = next(iter(self._store))
            self._evict_no_lock(key)
        except (StopIteration, KeyError):
            pass

    def _evict_no_lock(self, key: K) -> None:
        """Remove ``key`` from the store without acquiring the lock."""
        if key in self._store:
            value = self._store.pop(key)
            cost = max(0, self._safe_estimate(value))
            self._bytes -= cost
            self.metrics.evictions += 1
            self._emit("cache_evict", {"key": key, "cost": cost})

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        """Send telemetry events when a callback is registered."""
        if self._telemetry is None:
            return
        try:  # pragma: no cover - telemetry is optional best effort
            self._telemetry(event, {"namespace": self.namespace, **payload})
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            logger.debug("Telemetry callback failed for %s: %s", event, sys.exc_info()[1])


@dataclass
class CalibratorCache(Generic[V]):
    """Namespace-aware cache wrapper used by the explainer runtime.

    Provides invalidation and flush APIs per ADR-003 requirements, including:
    - Manual flush() and reset_version() for controlled invalidation
    - Version tracking and strategy revision updates
    - Thread-safe access to all operations
    """

    config: CacheConfig
    cache: LRUCache[Tuple[Hashable, ...], V] | None = field(init=False, default=None)
    _version_lock: threading.RLock = field(init=False, default_factory=threading.RLock)

    def __post_init__(self) -> None:
        """Initialise the underlying cache if caching is enabled."""
        if self.config.enabled:
            self.cache = LRUCache(
                namespace=self.config.namespace,
                version=self.config.version,
                max_items=self.config.max_items,
                max_bytes=self.config.max_bytes,
                ttl_seconds=self.config.ttl_seconds,
                telemetry=self.config.telemetry,
                size_estimator=self.config.size_estimator,
            )
        else:
            self.cache = None

    @property
    def enabled(self) -> bool:
        """Return True when the cache backend is active."""
        return self.cache is not None

    @property
    def metrics(self) -> CacheMetrics:
        """Return telemetry counters, falling back to empty metrics when disabled."""
        if self.cache is None:
            return CacheMetrics()
        return self.cache.metrics

    @property
    def version(self) -> str:
        """Return the current cache version tag used for key namespacing."""
        with self._version_lock:
            return self.config.version

    def get(self, *, stage: str, parts: Iterable[Any]) -> V | None:
        """Retrieve a cached value for ``stage`` and ``parts``."""
        if self.cache is None:
            return None
        key = make_key(self.config.namespace, f"{self.config.version}:{stage}", parts)
        return self.cache.get(key)

    def set(self, *, stage: str, parts: Iterable[Any], value: V) -> None:
        """Store ``value`` for ``stage`` and ``parts`` if caching is enabled."""
        if self.cache is None:
            return
        key = make_key(self.config.namespace, f"{self.config.version}:{stage}", parts)
        self.cache.set(key, value)

    def compute(self, *, stage: str, parts: Iterable[Any], fn: Callable[[], V]) -> V:
        """Return a cached value or compute and store it when missing."""
        cached = self.get(stage=stage, parts=parts)
        if cached is not None:
            return cached
        value = fn()
        self.set(stage=stage, parts=parts, value=value)
        return value

    def flush(self) -> None:
        """Manually flush all cache entries.

        This clears all cached values without changing the version tag,
        used when invalidation is triggered by user action or external signal.
        """
        if self.cache is not None:
            with self.cache._lock:
                self.cache._store.clear()
                self.cache._bytes = 0
                self.cache.metrics.resets += 1
                self.cache._emit("cache_flush", {"reason": "manual"})

    def reset_version(self, new_version: str) -> None:
        """Reset the version tag to invalidate all cache entries.

        Bump the version when algorithm parameters, code logic, or strategy
        implementation changes affect the meaning of cached values. All entries
        with the old version become unreachable (orphaned).

        Parameters
        ----------
        new_version : str
            New version tag (e.g., "v2", "calibrator_v1.1").
        """
        with self._version_lock:
            old_version = self.config.version
            self.config.version = new_version
            if self.cache is not None:
                self.cache._emit(
                    "cache_version_reset",
                    {"old_version": old_version, "new_version": new_version},
                )
            logger.info(
                "Cache version updated from %s to %s (namespace: %s)",
                old_version,
                new_version,
                self.config.namespace,
            )

    def forksafe_reset(self) -> None:
        """Reset the cache safely after ``fork`` events."""
        if self.cache is not None:
            self.cache.forksafe_reset()

    def __getstate__(self) -> dict:
        """Support pickling by excluding the unpicklable RLock."""
        state = self.__dict__.copy()
        # Remove the unpicklable _version_lock; it will be recreated in __setstate__
        state.pop("_version_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and recreate the unpicklable RLock."""
        self.__dict__.update(state)
        # Recreate the lock after unpickling
        self._version_lock = threading.RLock()


__all__ = [
    "CacheConfig",
    "CacheMetrics",
    "CalibratorCache",
    "LRUCache",
    "TelemetryCallback",
    "make_key",
]
