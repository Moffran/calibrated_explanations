"""Parallel execution helpers for ADR-004 compliant runtime toggles."""

from __future__ import annotations

import logging
import os
import time
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

from ..cache import CalibratorCache, TelemetryCallback

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
    total_duration: float = 0.0
    max_workers: int = 0

    def snapshot(self) -> Mapping[str, int | float]:
        """Return the metrics as a serialisable mapping."""
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
            "total_duration": self.total_duration,
            "max_workers": self.max_workers,
        }


@dataclass
class ParallelConfig:
    """Configuration options for the parallel executor."""

    enabled: bool = False
    strategy: Literal["auto", "threads", "processes", "joblib", "sequential"] = "auto"
    max_workers: int | None = None
    min_batch_size: int = 32
    instance_chunk_size: int | None = None
    feature_chunk_size: int | None = None
    granularity: Literal["feature", "instance"] = "feature"
    task_size_hint_bytes: int = 0
    force_serial_on_failure: bool = False
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
            if token.startswith("instance_chunk="):
                cfg.instance_chunk_size = max(1, int(token.split("=", 1)[1]))
                continue
            if token.startswith("feature_chunk="):
                cfg.feature_chunk_size = max(1, int(token.split("=", 1)[1]))
                continue
            if token.startswith("task_bytes="):
                cfg.task_size_hint_bytes = max(0, int(token.split("=", 1)[1]))
                continue
            if token.startswith("force_serial="):
                val = token.split("=", 1)[1].lower()
                cfg.force_serial_on_failure = val in {"1", "true", "on"}
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
        self._pool: Any = None
        self._active_strategy_name: str | None = None

    def __enter__(self) -> "ParallelExecutor":
        """Initialize the execution pool if parallelism is enabled."""
        if not self.config.enabled:
            return self

        strategy_name = self.config.strategy
        if strategy_name == "auto":
            strategy_name = self._auto_strategy()

        self._active_strategy_name = strategy_name

        try:
            if strategy_name == "threads":
                max_workers = self.config.max_workers or min(32, (os.cpu_count() or 1) * 5)
                self._pool = ThreadPoolExecutor(max_workers=max_workers)
            elif strategy_name == "processes":
                max_workers = self.config.max_workers or (os.cpu_count() or 1)
                if self.cache is not None:
                    self.cache.forksafe_reset()
                self._pool = ProcessPoolExecutor(max_workers=max_workers)
            elif strategy_name == "joblib":
                if _JoblibParallel is not None:
                    n_jobs = self.config.max_workers or -1
                    self._pool = _JoblibParallel(n_jobs=n_jobs, prefer="processes")
        except Exception as exc:
            logger.warning("Failed to initialize parallel pool: %s. Falling back to serial.", exc)
            self._pool = None
            self._active_strategy_name = "sequential"

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shutdown the execution pool."""
        if exc_type is not None:
            self.cancel()
            return

        if self._pool is not None:
            if hasattr(self._pool, "shutdown"):
                self._pool.shutdown(wait=True)
            self._pool = None
        self._active_strategy_name = None

    def cancel(self) -> None:
        """Cancel all pending tasks and shutdown the pool immediately."""
        if self._pool is not None:
            if hasattr(self._pool, "shutdown"):
                try:
                    # Python 3.9+ supports cancel_futures
                    self._pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Fallback for older Pythons or executors without cancel_futures
                    self._pool.shutdown(wait=False)
            self._pool = None
        self._active_strategy_name = None

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
        chunksize: int | None = None,
    ) -> List[R]:
        """Execute *fn* across *items* using the configured parallel strategy."""
        items_list = list(items)
        if not self.config.enabled or len(items_list) == 0:
            return [fn(item) for item in items_list]
        candidate = work_items if work_items is not None else len(items_list)
        if candidate < self.config.min_batch_size:
            return [fn(item) for item in items_list]

        self.metrics.submitted += len(items_list)
        start_time = time.perf_counter()

        try:
            strategy = self._resolve_strategy(work_items=candidate)
            results = strategy(fn, items_list, workers=workers, chunksize=chunksize)
        except Exception as exc:  # pragma: no cover - best effort fallback
            self.metrics.failures += 1
            self.metrics.fallbacks += 1
            self._emit("parallel_fallback", {"error": repr(exc)})
            if self.config.force_serial_on_failure:
                results = [fn(item) for item in items_list]
            else:
                raise exc
        else:
            self.metrics.completed += len(results)
            duration = time.perf_counter() - start_time
            self.metrics.total_duration += duration

            # Estimate workers used
            current_workers = workers or self.config.max_workers or 1
            if self._pool is not None and hasattr(self._pool, "_max_workers"):
                current_workers = self._pool._max_workers
            self.metrics.max_workers = max(self.metrics.max_workers, current_workers)

            self._emit(
                "parallel_execution",
                {
                    "strategy": self._active_strategy_name or self.config.strategy,
                    "items": len(items_list),
                    "duration": duration,
                    "workers": current_workers,
                    "work_items": candidate,
                    "task_size_hint_bytes": self.config.task_size_hint_bytes,
                    "granularity": self.config.granularity,
                },
            )
        return results

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    def _resolve_strategy(
        self,
        *,
        work_items: int | None = None,
    ) -> Callable[[Callable[[T], R], Sequence[T], Any], List[R]]:
        """Return a concrete execution strategy based on configuration."""
        strategy = self._active_strategy_name or self.config.strategy
        if strategy == "auto":
            strategy = self._auto_strategy(work_items=work_items)
        if strategy == "threads":
            return partial(self._thread_strategy)
        if strategy == "processes":
            return partial(self._process_strategy)
        if strategy == "joblib":
            return partial(self._joblib_strategy)
        return partial(self._serial_strategy)

    def _auto_strategy(self, *, work_items: int | None = None) -> str:
        """Choose a sensible default backend for the current platform."""
        # 0. Honour explicit hints to stay serial for tiny workloads
        if work_items is not None and work_items < max(2 * self.config.min_batch_size, 64):
            return "sequential"

        # 1. Platform constraints
        if os.name == "nt":  # prefer threads on Windows for compatibility
            return "threads"

        # 2. Resource constraints
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 2:
            return "threads"

        # 3. Workload heuristics
        # If tasks are very large, pickling overhead in processes might outweigh GIL benefits
        # Threshold: 10MB (arbitrary, but conservative)
        if self.config.task_size_hint_bytes > 10 * 1024 * 1024:
            return "threads"

        if work_items is not None:
            # Prefer threads when payloads are modest or when feature granularity favours
            # in-process sharing; tilt toward processes only when the workload is heavy
            # enough to amortise spin-up costs.
            if work_items < 5000:
                return "threads"
            if work_items > 50000 and self.config.granularity == "instance":
                return "processes"

        # 4. Library preference
        # When joblib is available prefer it because of smarter chunking
        if _JoblibParallel is not None:
            return "joblib"
        return "processes"

    # ------------------------------------------------------------------
    # Individual strategies
    # ------------------------------------------------------------------
    def _serial_strategy(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        *,
        workers: int | None = None,
        chunksize: int | None = None,
    ) -> List[R]:
        """Fallback strategy executing sequentially in the current process."""
        return [fn(item) for item in items]

    def _thread_strategy(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        *,
        workers: int | None = None,
        chunksize: int | None = None,
    ) -> List[R]:
        """Execute work items using a thread pool."""
        # ThreadPoolExecutor.map uses chunksize=1 by default if not specified
        chunksize = chunksize if chunksize is not None else 1

        if self._pool is not None and isinstance(self._pool, ThreadPoolExecutor):
            return list(self._pool.map(fn, items, chunksize=chunksize))

        max_workers = workers or self.config.max_workers or min(32, (os.cpu_count() or 1) * 5)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(fn, items, chunksize=chunksize))

    def _process_strategy(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        *,
        workers: int | None = None,
        chunksize: int | None = None,
    ) -> List[R]:
        """Execute work items using a process pool with cache isolation."""
        # ProcessPoolExecutor.map uses a heuristic for chunksize if not specified
        chunksize = chunksize if chunksize is not None else 1

        if self._pool is not None and isinstance(self._pool, ProcessPoolExecutor):
            return list(self._pool.map(fn, items, chunksize=chunksize))

        max_workers = workers or self.config.max_workers or (os.cpu_count() or 1)
        # Reset cache to avoid cross-process contamination (fork safety)
        if self.cache is not None:
            self.cache.forksafe_reset()
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=None) as pool:
            return list(pool.map(fn, items, chunksize=chunksize))

    def _joblib_strategy(
        self,
        fn: Callable[[T], R],
        items: Sequence[T],
        *,
        workers: int | None = None,
        chunksize: int | None = None,
    ) -> List[R]:
        """Dispatch work through joblib's Parallel abstraction when available."""
        if _JoblibParallel is None:
            return self._thread_strategy(fn, items, workers=workers, chunksize=chunksize)

        # joblib uses 'batch_size' instead of 'chunksize'
        batch_size = chunksize if chunksize is not None else "auto"

        if self._pool is not None and isinstance(self._pool, _JoblibParallel):
            # Reusing joblib pool is tricky as it's usually a context manager or object
            # If self._pool is a Parallel instance, we can call it.
            return self._pool(_joblib_delayed(fn)(item) for item in items)

        n_jobs = workers or self.config.max_workers or -1
        parallel = _JoblibParallel(n_jobs=n_jobs, prefer="processes", batch_size=batch_size)
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
