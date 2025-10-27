"""Parallel execution helpers for ADR-004 compliant runtime toggles."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, List, Literal, Mapping, Sequence, TypeVar

try:  # pragma: no cover - optional dependency
    from joblib import Parallel as _JoblibParallel
    from joblib import delayed as _joblib_delayed
except Exception:  # pragma: no cover - joblib remains optional
    _JoblibParallel = None  # type: ignore[assignment]
    _joblib_delayed = None  # type: ignore[assignment]

from .cache import CalibratorCache, TelemetryCallback

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelMetrics:
    """Telemetry counters collected by :class:`ParallelExecutor`."""

    submitted: int = 0
    completed: int = 0
    fallbacks: int = 0
    failures: int = 0

    def snapshot(self) -> Mapping[str, int]:
        """Return the metrics as a serialisable mapping."""
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
        }


@dataclass
class ParallelConfig:
    """Configuration options for the parallel executor."""

    enabled: bool = False
    strategy: Literal["auto", "threads", "processes", "joblib", "sequential"] = "auto"
    max_workers: int | None = None
    min_batch_size: int = 32
    granularity: Literal["feature", "instance"] = "feature"
    telemetry: TelemetryCallback | None = None

    @classmethod
    def from_env(cls, base: "ParallelConfig | None" = None) -> "ParallelConfig":
        """Merge ``CE_PARALLEL`` overrides with an optional ``base`` configuration."""
        cfg = ParallelConfig(**(base.__dict__ if base is not None else {}))
        raw = os.getenv("CE_PARALLEL")
        if not raw:
            return cfg
        tokens = [segment.strip() for segment in raw.split(",") if segment.strip()]
        if len(tokens) == 1 and tokens[0].lower() in {"1", "true", "on"}:
            cfg.enabled = True
            return cfg
        for token in tokens:
            lowered = token.lower()
            if lowered in {"0", "off", "false"}:
                cfg.enabled = False
                continue
            if lowered in {"threads", "processes", "joblib", "sequential", "auto"}:
                cfg.strategy = lowered  # type: ignore[assignment]
                continue
            if token.startswith("workers="):
                cfg.max_workers = max(1, int(token.split("=", 1)[1]))
                continue
            if token.startswith("min_batch="):
                cfg.min_batch_size = max(1, int(token.split("=", 1)[1]))
                continue
            if token == "enable":  # noqa: S105  # nosec B105 - configuration toggle keyword
                cfg.enabled = True
                continue
            if token.startswith("granularity="):
                value = token.split("=", 1)[1].strip().lower()
                if value in {"feature", "instance"}:
                    cfg.granularity = value  # type: ignore[assignment]
        return cfg


class ParallelExecutor:
    """Facade that selects a strategy and provides graceful fallbacks."""

    def __init__(
        self,
        config: ParallelConfig,
        *,
        cache: CalibratorCache[Any] | None = None,
    ) -> None:
        """Store configuration and telemetry state for later map calls."""
        self.config = config
        self.cache = cache
        self.metrics = ParallelMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def map(
        self,
        fn: Callable[[T], R],
        items: Sequence[T] | Iterable[T],
        *,
        workers: int | None = None,
        work_items: int | None = None,
    ) -> List[R]:
        """Execute *fn* across *items* using the configured parallel strategy."""
        items_list = list(items)
        if not self.config.enabled or len(items_list) == 0:
            return [fn(item) for item in items_list]
        candidate = work_items if work_items is not None else len(items_list)
        if candidate < self.config.min_batch_size:
            return [fn(item) for item in items_list]
        self.metrics.submitted += len(items_list)
        try:
            strategy = self._resolve_strategy()
            results = strategy(fn, items_list, workers=workers)
        except Exception as exc:  # pragma: no cover - best effort fallback
            self.metrics.failures += 1
            self.metrics.fallbacks += 1
            self._emit("parallel_fallback", {"error": repr(exc)})
            results = [fn(item) for item in items_list]
        else:
            self.metrics.completed += len(results)
        return results

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    def _resolve_strategy(
        self,
    ) -> Callable[[Callable[[T], R], Sequence[T], Any], List[R]]:
        """Return a concrete execution strategy based on configuration."""
        strategy = self.config.strategy
        if strategy == "auto":
            strategy = self._auto_strategy()
        if strategy == "threads":
            return partial(self._thread_strategy)
        if strategy == "processes":
            return partial(self._process_strategy)
        if strategy == "joblib":
            return partial(self._joblib_strategy)
        return partial(self._serial_strategy)

    def _auto_strategy(self) -> str:
        """Choose a sensible default backend for the current platform."""
        if os.name == "nt":  # prefer threads on Windows for compatibility
            return "threads"
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 2:
            return "threads"
        # When joblib is available prefer it because of smarter chunking
        if _JoblibParallel is not None:
            return "joblib"
        return "processes"

    # ------------------------------------------------------------------
    # Individual strategies
    # ------------------------------------------------------------------
    def _serial_strategy(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        """Fallback strategy executing sequentially in the current process."""
        return [fn(item) for item in items]

    def _thread_strategy(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        """Execute work items using a thread pool."""
        max_workers = workers or self.config.max_workers or min(32, (os.cpu_count() or 1) * 5)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(fn, items))

    def _process_strategy(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        """Execute work items using a process pool with cache isolation."""
        max_workers = workers or self.config.max_workers or (os.cpu_count() or 1)
        # Reset cache to avoid cross-process contamination (fork safety)
        if self.cache is not None:
            self.cache.forksafe_reset()
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=None) as pool:
            return list(pool.map(fn, items))

    def _joblib_strategy(
        self, fn: Callable[[T], R], items: Sequence[T], *, workers: int | None = None
    ) -> List[R]:
        """Dispatch work through joblib's Parallel abstraction when available."""
        if _JoblibParallel is None:
            return self._thread_strategy(fn, items, workers=workers)
        n_jobs = workers or self.config.max_workers or -1
        parallel = _JoblibParallel(n_jobs=n_jobs, prefer="processes")
        return parallel(_joblib_delayed(fn)(item) for item in items)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        """Emit telemetry payloads guarding against user callback failures."""
        if self.config.telemetry is None:
            return
        try:  # pragma: no cover - telemetry best effort
            self.config.telemetry(event, payload)
        except Exception as exc:
            logger.debug("Parallel telemetry callback failed for %s: %s", event, exc)


__all__ = ["ParallelConfig", "ParallelExecutor", "ParallelMetrics"]
