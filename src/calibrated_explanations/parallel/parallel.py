"""Parallel execution helpers for ADR-004 compliant runtime toggles."""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, List, Literal, Mapping, Sequence, TypeVar

try:  # pragma: no cover - optional dependency
    from joblib import Parallel as _JoblibParallel
    from joblib import delayed as _joblib_delayed
except BaseException:  # pragma: no cover - joblib remains optional
    if not isinstance(sys.exc_info()[1], Exception):
        raise
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
    worker_utilisation_pct: float = 0.0

    def snapshot(self) -> Mapping[str, int | float]:
        """Return the metrics as a serialisable mapping."""
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
            "total_duration": self.total_duration,
            "max_workers": self.max_workers,
            "worker_utilisation_pct": self.worker_utilisation_pct,
        }


@dataclass
class ParallelConfig:
    """Configuration options for the parallel executor."""

    enabled: bool = False
    strategy: Literal["auto", "threads", "processes", "joblib", "sequential"] = "auto"
    max_workers: int | None = None
    min_batch_size: int = 8
    min_instances_for_parallel: int | None = None
    tiny_workload_threshold: int | None = None
    instance_chunk_size: int | None = None
    feature_chunk_size: int | None = None
    granularity: Literal["instance"] = "instance"
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
            if token.startswith("min_instances="):
                cfg.min_instances_for_parallel = max(1, int(token.split("=", 1)[1]))
                continue
            if token.startswith("tiny="):
                cfg.tiny_workload_threshold = max(1, int(token.split("=", 1)[1]))
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
                if value == "feature":
                    warnings.warn(
                        "Feature parallelism is deprecated and removed. Using 'instance' parallelism instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    cfg.granularity = "instance"
                elif value == "instance":
                    cfg.granularity = "instance"
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
        self._warned_min_batch: bool = False
        self._warned_tiny_workload: bool = False

    def __getstate__(self) -> dict[str, Any]:
        """Return state for pickling, excluding the pool."""
        state = self.__dict__.copy()
        # Exclude the pool as it is not picklable (ProcessPoolExecutor)
        # and we don't want to share the pool with workers anyway.
        state["_pool"] = None
        return state

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
                    self._pool.__enter__()
        except BaseException:  # ADR-002: catch all exceptions including system exits
            exc = sys.exc_info()[1]
            if not isinstance(exc, Exception):
                raise
            logger.warning("Failed to initialize parallel pool: %s. Falling back to serial.", exc)
            logger.info("Parallel pool init failure; switching to sequential execution")
            warnings.warn(
                f"Failed to initialize parallel pool ({exc!r}); falling back to sequential execution.",
                UserWarning,
                stacklevel=2,
            )
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
            elif hasattr(self._pool, "__exit__"):
                self._pool.__exit__(exc_type, exc_val, exc_tb)
            self._pool = None
        self._active_strategy_name = None

    def cancel(self) -> None:
        """Cancel all pending tasks and shutdown the pool immediately."""
        if self._pool is not None:
            if hasattr(self._pool, "shutdown"):
                try:
                    # Python 3.9+ supports cancel_futures
                    self._pool.shutdown(wait=False, cancel_futures=True)
                except:  # noqa: E722 - ADR-002: check for specific exception types
                    if not isinstance(sys.exc_info()[1], TypeError):
                        raise
                    # Fallback for older Pythons or executors without cancel_futures
                    self._pool.shutdown(wait=False)
            self._pool = None
        self._active_strategy_name = None

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------
    def _effective_min_instances(self) -> int:
        """Return the minimum instance count that should trigger parallelism."""
        if self.config.min_instances_for_parallel is not None:
            return max(1, self.config.min_instances_for_parallel)
        chunk = self.config.instance_chunk_size or self.config.min_batch_size
        return max(8, chunk, 1)

    def _effective_min_batch_threshold(self) -> int:
        """Compute the gating threshold for parallel execution."""
        if self.config.granularity == "instance":
            return self._effective_min_instances()
        return max(1, self.config.min_batch_size)

    def _tiny_workload_threshold(
        self, min_batch_threshold: int, *, work_items: int | None = None
    ) -> int:
        """Compute the tiny-workload guard threshold, respecting overrides."""
        if self.config.tiny_workload_threshold is not None:
            return max(min_batch_threshold, self.config.tiny_workload_threshold)

        if self.config.granularity == "instance":
            return max(
                min_batch_threshold,
                self._effective_min_instances(),
                self.config.instance_chunk_size or 0,
            )

        base_floor = 8 if self.config.min_batch_size >= 8 else min_batch_threshold
        base = max(min_batch_threshold, base_floor)
        scaled = max(int(base * 1.5), base)
        cap = max(base * 2, 16)
        return min(scaled, cap)

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
        min_batch_threshold = self._effective_min_batch_threshold()
        if candidate < min_batch_threshold:
            if not self._warned_min_batch:
                logger.warning(
                    "Parallel execution disabled: workload (%d) below min parallel threshold (%d); running sequential.",
                    candidate,
                    min_batch_threshold,
                )
                logger.info(
                    "Parallel decision: sequential (reason=below_min_batch_size, workload=%d, threshold=%d)",
                    candidate,
                    min_batch_threshold,
                )
                warnings.warn(
                    f"Parallel execution disabled: workload ({candidate}) below minimum parallel threshold ({min_batch_threshold}); running sequential.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_min_batch = True
            self._emit(
                "parallel_decision",
                {
                    "decision": "sequential",
                    "reason": "below_min_batch_size",
                    "work_items": candidate,
                    "min_batch_size": self.config.min_batch_size,
                    "min_instances_for_parallel": self.config.min_instances_for_parallel,
                    "effective_min_threshold": min_batch_threshold,
                    "granularity": self.config.granularity,
                },
            )
            return [fn(item) for item in items_list]
        tiny_threshold = self._tiny_workload_threshold(min_batch_threshold, work_items=candidate)
        if candidate < tiny_threshold:
            if not self._warned_tiny_workload:
                logger.warning(
                    "Parallel execution disabled: workload (%d) below tiny-workload threshold (%d); running sequential.",
                    candidate,
                    tiny_threshold,
                )
                logger.info(
                    "Parallel decision: sequential (reason=tiny_workload, workload=%d, threshold=%d)",
                    candidate,
                    tiny_threshold,
                )
                warnings.warn(
                    f"Parallel execution disabled: workload ({candidate}) below tiny-workload threshold ({tiny_threshold}); running sequential.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_tiny_workload = True
            self._emit(
                "parallel_decision",
                {
                    "decision": "sequential",
                    "reason": "tiny_workload",
                    "work_items": candidate,
                    "min_batch_size": self.config.min_batch_size,
                    "tiny_threshold": tiny_threshold,
                    "min_instances_for_parallel": self.config.min_instances_for_parallel,
                    "granularity": self.config.granularity,
                },
            )
            return [fn(item) for item in items_list]

        self.metrics.submitted += len(items_list)
        start_time = time.perf_counter()

        try:
            strategy = self._resolve_strategy(work_items=candidate)
            results = strategy(fn, items_list, workers=workers, chunksize=chunksize)
        except BaseException:  # pragma: no cover - best effort fallback; ADR-002
            exc = sys.exc_info()[1]
            if not isinstance(exc, Exception):
                raise
            self.metrics.failures += 1
            self.metrics.fallbacks += 1
            self._emit("parallel_fallback", {"error": repr(exc)})
            if self.config.force_serial_on_failure:
                warnings.warn(
                    f"Parallel execution failed ({exc!r}); falling back to sequential execution.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.info("Parallel failure; forced serial fallback engaged")
                results = [fn(item) for item in items_list]
            else:
                raise exc from None
        else:
            self.metrics.completed += len(results)
            duration = time.perf_counter() - start_time
            self.metrics.total_duration += duration

            # Estimate workers used
            current_workers = workers or self.config.max_workers or 1
            if self._pool is not None and hasattr(self._pool, "_max_workers"):
                current_workers = self._pool._max_workers
            self.metrics.max_workers = max(self.metrics.max_workers, current_workers)

            # Calculate utilisation (simple saturation proxy)
            if current_workers > 0:
                util = min(len(items_list), current_workers) / current_workers * 100.0
                self.metrics.worker_utilisation_pct = util

            self._emit(
                "parallel_execution",
                {
                    "strategy": self._active_strategy_name or self.config.strategy,
                    "items": len(items_list),
                    "duration": duration,
                    "workers": current_workers,
                    "worker_utilisation_pct": self.metrics.worker_utilisation_pct,
                    "work_items": candidate,
                    "task_size_hint_bytes": self.config.task_size_hint_bytes,
                    "granularity": self.config.granularity,
                    "effective_min_threshold": min_batch_threshold,
                    "tiny_threshold": tiny_threshold,
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

    @staticmethod
    def _get_cgroup_cpu_quota() -> float | None:
        """Read the CPU quota from cgroup v2 or v1 interface."""
        if os.name == "nt":
            return None
        try:
            cgroup_root = Path("/sys/fs/cgroup")
        except NotImplementedError:
            return None
        # cgroup v2
        cpu_max = cgroup_root / "cpu.max"
        if cpu_max.exists():
            try:
                parts = cpu_max.read_text(encoding="utf-8").strip().split()
            except OSError:
                parts = []

            if len(parts) == 2:
                quota_s, period_s = parts
                if quota_s != "max" and quota_s.isdigit() and period_s.isdigit():
                    period = int(period_s)
                    if period > 0:
                        return int(quota_s) / period

        # cgroup v1
        cpu_quota = cgroup_root / "cpu/cpu.cfs_quota_us"
        cpu_period = cgroup_root / "cpu/cpu.cfs_period_us"
        if cpu_quota.exists() and cpu_period.exists():
            try:
                quota_s = cpu_quota.read_text(encoding="utf-8").strip()
                period_s = cpu_period.read_text(encoding="utf-8").strip()
            except OSError:
                return None

            # allow -1 sentinel for quota
            if (
                quota_s.isdigit() or (quota_s.startswith("-") and quota_s[1:].isdigit())
            ) and period_s.isdigit():
                quota = int(quota_s)
                period = int(period_s)
                if quota != -1 and period > 0:
                    return quota / period

        return None

    @staticmethod
    def _is_ci_environment() -> bool:
        """Detect if running in a CI environment."""
        return (
            os.getenv("CI", "").lower() == "true"
            or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        )

    def _auto_strategy(self, *, work_items: int | None = None) -> str:
        """Choose a sensible default backend for the current platform."""
        # 0. Honour explicit hints to stay serial for tiny workloads
        if work_items is not None:
            min_batch_threshold = self._effective_min_batch_threshold()
            tiny_threshold = self._tiny_workload_threshold(
                min_batch_threshold, work_items=work_items
            )
            if work_items < tiny_threshold:
                self._emit(
                    "parallel_decision",
                    {
                        "decision": "sequential",
                        "reason": "tiny_workload",
                        "tiny_threshold": tiny_threshold,
                        "granularity": self.config.granularity,
                    },
                )
                return "sequential"

        # 1. CI Environment Guardrails
        if self._is_ci_environment():
            self._emit("parallel_decision", {"decision": "sequential", "reason": "ci_environment"})
            return "sequential"

        # 2. Resource constraints
        cpu_count = os.cpu_count() or 1
        cgroup_limit = self._get_cgroup_cpu_quota()
        if cgroup_limit is not None:
            cpu_count = min(cpu_count, int(cgroup_limit))

        if cpu_count <= 2:
            self._emit("parallel_decision", {"decision": "threads", "reason": "low_cpu_count"})
            return "threads"

        # 3. Workload heuristics
        # If tasks are very large, pickling overhead in processes might outweigh GIL benefits
        # Threshold: 10MB (arbitrary, but conservative)
        if self.config.task_size_hint_bytes > 10 * 1024 * 1024:
            self._emit("parallel_decision", {"decision": "threads", "reason": "large_task_size"})
            return "threads"

        if work_items is not None:
            # Benchmarks indicate sequential execution is faster for < 2500 instances
            # due to process spawn/pickle overhead dominating the optimized sequential path.
            if work_items < 2500:
                self._emit(
                    "parallel_decision", {"decision": "sequential", "reason": "small_workload"}
                )
                return "sequential"

            if work_items > 50000 and self.config.granularity == "instance":
                self._emit(
                    "parallel_decision",
                    {"decision": "processes", "reason": "large_instance_workload"},
                )
                return "processes"

        # 4. Library preference
        # When joblib is available prefer it because of smarter chunking
        if _JoblibParallel is not None:
            self._emit("parallel_decision", {"decision": "joblib", "reason": "joblib_available"})
            return "joblib"

        # 5. Default fallback
        if os.name == "nt":
            # Windows multiprocessing is slow (spawn), so prefer threads if joblib is missing
            self._emit("parallel_decision", {"decision": "threads", "reason": "windows_default"})
            return "threads"

        self._emit("parallel_decision", {"decision": "processes", "reason": "default_fallback"})
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
        max_workers = workers or self.config.max_workers or min(32, (os.cpu_count() or 1) * 5)
        if self._pool is not None and isinstance(self._pool, ThreadPoolExecutor):
            # Use the pool's max_workers if available
            max_workers = getattr(self._pool, "_max_workers", max_workers)

        # Heuristic: if chunksize is not specified, use a larger chunksize to reduce queue contention
        if chunksize is None:
            # For threads, we can be more aggressive with small chunks, but 1 is still too small for tiny tasks
            chunksize, extra = divmod(len(items), max_workers * 4)
            if extra:
                chunksize += 1
            chunksize = max(1, chunksize)

        if self._pool is not None and isinstance(self._pool, ThreadPoolExecutor):
            return list(self._pool.map(fn, items, chunksize=chunksize))

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
        max_workers = workers or self.config.max_workers or (os.cpu_count() or 1)
        if self._pool is not None and isinstance(self._pool, ProcessPoolExecutor):
            max_workers = getattr(self._pool, "_max_workers", max_workers)

        # Heuristic: if chunksize is not specified, calculate one to amortize IPC overhead
        if chunksize is None:
            chunksize, extra = divmod(len(items), max_workers * 4)
            if extra:
                chunksize += 1
            chunksize = max(1, chunksize)

        if self._pool is not None and isinstance(self._pool, ProcessPoolExecutor):
            return list(self._pool.map(fn, items, chunksize=chunksize))

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
            warnings.warn(
                "Joblib is not available; falling back to thread-based parallel execution.",
                UserWarning,
                stacklevel=2,
            )
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
        except BaseException:  # ADR-002: catch all exceptions including system exits
            exc = sys.exc_info()[1]
            if not isinstance(exc, Exception):
                raise
            logger.debug("Parallel telemetry callback failed for %s: %s", event, exc)


__all__ = ["ParallelConfig", "ParallelExecutor", "ParallelMetrics"]
