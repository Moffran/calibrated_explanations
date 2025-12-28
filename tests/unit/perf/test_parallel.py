from __future__ import annotations
from functools import partial

from typing import Any

import pytest

from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor, ParallelMetrics


class DummyCache:
    def __init__(self):
        self.reset_calls = 0

    def forksafe_reset(self):
        self.reset_calls += 1


class DummyPool:
    def __init__(self, max_workers=None, **kwargs):
        self.max_workers = max_workers
        self.mapped = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, items, chunksize=None):
        self.mapped = list(items)
        return [fn(item) for item in items]


def echo(x):
    return x


def test_parallel_metrics_snapshot():
    metrics = ParallelMetrics(submitted=3, completed=2, fallbacks=1, failures=0)
    assert metrics.snapshot() == {
        "submitted": 3,
        "completed": 2,
        "fallbacks": 1,
        "failures": 0,
        "total_duration": 0.0,
        "max_workers": 0,
        "worker_utilisation_pct": 0.0,
    }


def test_parallel_config_from_env(monkeypatch):
    base = ParallelConfig(enabled=False, strategy="sequential", max_workers=4, min_batch_size=8)
    monkeypatch.delenv("CE_PARALLEL", raising=False)
    assert ParallelConfig.from_env(base).enabled is False

    monkeypatch.setenv("CE_PARALLEL", "1")
    cfg = ParallelConfig.from_env(base)
    assert cfg.enabled is True
    assert cfg.strategy == "sequential"

    monkeypatch.setenv("CE_PARALLEL", "off,threads,workers=10,min_batch=2,tiny=11,enable")
    cfg = ParallelConfig.from_env(base)
    assert cfg.enabled is True
    assert cfg.strategy == "threads"
    assert cfg.max_workers == 10
    assert cfg.min_batch_size == 2
    assert cfg.tiny_workload_threshold == 11


def test_parallel_config_from_env_extended_tokens(monkeypatch):
    base = ParallelConfig(enabled=True, min_batch_size=4)
    monkeypatch.setenv(
        "CE_PARALLEL",
        "min_instances=3,instance_chunk=5,feature_chunk=7,task_bytes=1024,"
        "force_serial=true,granularity=feature",
    )
    with pytest.warns(DeprecationWarning):
        cfg = ParallelConfig.from_env(base)
    assert cfg.min_instances_for_parallel == 3
    assert cfg.instance_chunk_size == 5
    assert cfg.feature_chunk_size == 7
    assert cfg.task_size_hint_bytes == 1024
    assert cfg.force_serial_on_failure is True
    assert cfg.granularity == "instance"


def test_map_handles_disabled_and_small_batches():
    config = ParallelConfig(enabled=False)
    executor = ParallelExecutor(config)
    assert executor.map(lambda x: x + 1, [1, 2]) == [2, 3]

    config = ParallelConfig(enabled=True, min_batch_size=5)
    executor = ParallelExecutor(config)
    assert executor.map(lambda x: x + 1, [1, 2]) == [2, 3]
    assert executor.metrics.submitted == 0


def test_map_uses_strategy_and_updates_metrics(monkeypatch, enable_fallbacks):
    config = ParallelConfig(
        enabled=True, strategy="sequential", min_batch_size=1, min_instances_for_parallel=1
    )
    executor = ParallelExecutor(config)
    results = executor.map(lambda x: x * 2, [1, 2, 3])
    assert results == [2, 4, 6]
    assert executor.metrics.submitted == 3
    assert executor.metrics.completed == 3

    def failing_strategy(*args, **kwargs):
        raise RuntimeError("boom")

    events = []

    def telemetry(event, payload):
        events.append((event, payload))

    config = ParallelConfig(
        enabled=True,
        strategy="threads",
        min_batch_size=1,
        min_instances_for_parallel=1,
        telemetry=telemetry,
        force_serial_on_failure=True,
    )
    print(f"DEBUG: config.force_serial_on_failure={config.force_serial_on_failure}")
    executor = ParallelExecutor(config)
    monkeypatch.setattr(executor, "_resolve_strategy", lambda **k: failing_strategy)
    with pytest.warns(UserWarning, match="falling back to sequential"):
        assert executor.map(lambda x: x + 1, [1]) == [2]
    assert executor.metrics.failures == 1
    assert executor.metrics.fallbacks == 1
    assert events and events[0][0] == "parallel_fallback"


def test_resolve_strategy_variants(monkeypatch):
    config = ParallelConfig(enabled=True, strategy="threads")
    executor = ParallelExecutor(config)
    strategy = executor._resolve_strategy()
    assert isinstance(strategy, partial)
    assert strategy.func.__name__ == "_thread_strategy"

    config.strategy = "processes"
    strategy = ParallelExecutor(config)._resolve_strategy()
    assert strategy.func.__name__ == "_process_strategy"

    config.strategy = "joblib"
    resolved = ParallelExecutor(config)._resolve_strategy()
    assert resolved.func.__name__ == "_joblib_strategy"

    config.strategy = "sequential"
    strategy = ParallelExecutor(config)._resolve_strategy()
    assert strategy.func.__name__ == "_serial_strategy"

    config.strategy = "auto"
    monkeypatch.setattr(executor, "_auto_strategy", lambda **k: "threads")
    assert executor._resolve_strategy().func.__name__ == "_thread_strategy"


def test_instance_minimum_overrides_min_batch(monkeypatch):
    config = ParallelConfig(
        enabled=True,
        strategy="threads",
        granularity="instance",
        min_batch_size=32,
        min_instances_for_parallel=8,
    )
    executor = ParallelExecutor(config)

    def fake_strategy(fn, items, **_):
        return [fn(item) for item in items]

    monkeypatch.setattr(executor, "_resolve_strategy", lambda **_: fake_strategy)
    results = executor.map(lambda x: x + 1, list(range(10)), work_items=10)

    assert results == [i + 1 for i in range(10)]
    assert executor.metrics.submitted == 10


def test_tiny_threshold_override(monkeypatch):
    config = ParallelConfig(
        enabled=True,
        strategy="threads",
        min_batch_size=1,
        tiny_workload_threshold=5,
    )
    executor = ParallelExecutor(config)

    def fake_strategy(fn, items, **_):
        return [fn(item) for item in items]

    monkeypatch.setattr(executor, "_resolve_strategy", lambda **_: fake_strategy)

    # 3 < tiny threshold forces sequential path
    results = executor.map(lambda x: x, [1, 2, 3], work_items=3)
    assert results == [1, 2, 3]
    assert executor.metrics.submitted == 0


def test_auto_strategy(monkeypatch):
    config = ParallelConfig(enabled=True, strategy="auto")
    executor = ParallelExecutor(config)

    import os

    class MockOS:
        name = "nt"

        @staticmethod
        def cpu_count():
            return os.cpu_count()

        @staticmethod
        def getenv(key, default=None):
            return default

    monkeypatch.setattr("calibrated_explanations.parallel.parallel.os", MockOS, raising=False)
    # Low CPU counts stay on threads
    MockOS.cpu_count = lambda: 1
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )
    assert executor._auto_strategy() == "threads"

    # Joblib is preferred even on Windows
    MockOS.cpu_count = lambda: 8
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", object(), raising=False
    )
    assert executor._auto_strategy() == "joblib"

    # Without joblib fall back to threads on Windows (spawn is slow)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )
    assert executor._auto_strategy() == "threads"


def test_auto_strategy_work_items(monkeypatch):
    # Ensure joblib doesn't preempt the logic
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )

    # Mock OS as POSIX to test process fallback
    import os

    class MockOS:
        name = "posix"

        @staticmethod
        def cpu_count():
            return 8

        @staticmethod
        def getenv(key, default=None):
            return default

    monkeypatch.setattr("calibrated_explanations.parallel.parallel.os", MockOS, raising=False)

    config = ParallelConfig(enabled=True, strategy="auto", min_batch_size=16)
    executor = ParallelExecutor(config)

    assert executor._auto_strategy(work_items=10) == "sequential"
    assert executor._auto_strategy(work_items=4000) == "processes"

    class MockOS:
        name = "posix"

        @staticmethod
        def cpu_count():
            return os.cpu_count()

        @staticmethod
        def getenv(key, default=None):
            return default

    monkeypatch.setattr("calibrated_explanations.parallel.parallel.os", MockOS, raising=False)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )
    executor.config.granularity = "instance"
    assert executor._auto_strategy(work_items=60000) == "processes"


def test_resolve_strategy_forwards_work_items(monkeypatch):
    config = ParallelConfig(enabled=True, strategy="auto")
    executor = ParallelExecutor(config)

    captured: dict[str, int | None] = {"work_items": None}

    def fake_auto_strategy(*, work_items: int | None = None) -> str:
        captured["work_items"] = work_items
        return "threads"

    monkeypatch.setattr(executor, "_auto_strategy", fake_auto_strategy)
    strategy = executor._resolve_strategy(work_items=123)

    assert captured["work_items"] == 123
    assert strategy.func.__name__ == "_thread_strategy"


def test_map_passes_work_items_to_resolver(monkeypatch):
    config = ParallelConfig(enabled=True, min_batch_size=1)
    executor = ParallelExecutor(config)

    captured: dict[str, int | None] = {"work_items": None}

    def fake_resolve_strategy(*, work_items: int | None = None):
        captured["work_items"] = work_items

        def runner(fn, items, **_: Any):
            return [fn(item) for item in items]

        return runner

    monkeypatch.setattr(executor, "_resolve_strategy", fake_resolve_strategy)
    executor.map(lambda x: x + 1, [1, 2, 3], work_items=99)

    assert captured["work_items"] == 99


def test_thread_strategy(monkeypatch):
    captured = {}

    class RecordingPool(DummyPool):
        def __exit__(self, exc_type, exc, tb):
            captured["max_workers"] = self.max_workers
            return super().__exit__(exc_type, exc, tb)

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.ThreadPoolExecutor",
        RecordingPool,
        raising=False,
    )
    config = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)
    executor = ParallelExecutor(config)
    results = executor._thread_strategy(echo, [1, 2, 3])
    assert results == [1, 2, 3]
    assert captured["max_workers"] == 2


def test_process_strategy(monkeypatch):
    cache = DummyCache()
    recorded = {}

    class RecordingPool(DummyPool):
        def __exit__(self, exc_type, exc, tb):
            recorded["max_workers"] = self.max_workers
            return super().__exit__(exc_type, exc, tb)

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.ProcessPoolExecutor",
        RecordingPool,
        raising=False,
    )
    config = ParallelConfig(enabled=True, strategy="processes", max_workers=3, min_batch_size=1)
    executor = ParallelExecutor(config, cache=cache)
    results = executor._process_strategy(echo, [1, 2])
    assert results == [1, 2]
    assert cache.reset_calls == 1
    assert recorded["max_workers"] == 3


def test_joblib_strategy(monkeypatch):
    config = ParallelConfig(enabled=True, strategy="joblib", min_batch_size=1)
    executor = ParallelExecutor(config)

    calls = {}

    def fake_thread(fn, items, workers=None, chunksize=None):
        calls["thread"] = True
        return [fn(item) for item in items]

    monkeypatch.setattr(executor, "_thread_strategy", fake_thread)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )
    assert executor._joblib_strategy(echo, [1, 2]) == [1, 2]
    assert calls["thread"]

    class FakeParallel:
        def __init__(self, *, n_jobs, prefer, batch_size="auto"):
            self.n_jobs = n_jobs
            self.prefer = prefer
            self.batch_size = batch_size

        def __call__(self, iterator):
            return list(iterator)

    def fake_delayed(fn):
        return lambda value: fn(value)

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", FakeParallel, raising=False
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._joblib_delayed", fake_delayed, raising=False
    )
    result = executor._joblib_strategy(echo, [1, 2, 3])
    assert result == [1, 2, 3]


def test_emit_with_telemetry(monkeypatch):
    events = []

    def telemetry(event, payload):
        events.append((event, payload))

    config = ParallelConfig(enabled=True, telemetry=telemetry)
    executor = ParallelExecutor(config)
    executor.emit("test", {"value": 1})
    assert events == [("test", {"value": 1})]

    def broken(event, payload):
        raise RuntimeError("ignored")

    config = ParallelConfig(enabled=True, telemetry=broken)
    executor = ParallelExecutor(config)
    executor.emit("test", {"value": 2})  # should not raise


def test_parallel_executor_context_manager_initializes_and_cleans_up(monkeypatch):
    shutdown_calls: list[int] = []

    class RecordingPool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def shutdown(self, wait=True):
            shutdown_calls.append(int(wait))

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.ThreadPoolExecutor",
        RecordingPool,
        raising=False,
    )
    cfg = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)
    with ParallelExecutor(cfg) as executor:
        assert isinstance(executor._pool, RecordingPool)
        assert executor._active_strategy_name == "threads"
    assert shutdown_calls == [1]


def test_parallel_executor_context_manager_handles_init_failure(monkeypatch, enable_fallbacks):
    class ExplodingPool:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.ThreadPoolExecutor",
        ExplodingPool,
        raising=False,
    )
    cfg = ParallelConfig(enabled=True, strategy="threads", max_workers=1, min_batch_size=1)
    executor = ParallelExecutor(cfg)
    with pytest.warns(UserWarning, match="Failed to initialize parallel pool"), executor as ctx:
        assert ctx._active_strategy_name == "sequential"
        assert ctx._pool is None


def test_parallel_executor_context_manager_cancels_on_error(monkeypatch):
    cancel_calls = []

    def fake_cancel(self):
        cancel_calls.append(True)
        if self._pool is not None and hasattr(self._pool, "shutdown"):
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool = None

    monkeypatch.setattr(ParallelExecutor, "cancel", fake_cancel)

    class DummyPool:
        def __init__(self, max_workers):
            pass

        def shutdown(self, wait=True, cancel_futures=None):
            pass

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.ThreadPoolExecutor",
        DummyPool,
        raising=False,
    )
    cfg = ParallelConfig(enabled=True, strategy="threads", max_workers=1, min_batch_size=1)
    with pytest.raises(RuntimeError), ParallelExecutor(cfg):
        raise RuntimeError("fail")
    assert cancel_calls == [True]


def test_parallel_executor_cancel_handles_missing_cancel_futures():
    executor = ParallelExecutor(ParallelConfig(enabled=True))

    class LegacyPool:
        def __init__(self):
            self.calls: list[tuple[bool, Any | None]] = []

        def shutdown(self, wait=False, cancel_futures=None):
            self.calls.append((wait, cancel_futures))
            if cancel_futures is not None:
                raise TypeError("unsupported")

    pool = LegacyPool()
    executor._pool = pool
    executor.cancel()
    assert pool.calls == [(False, True), (False, None)]
    assert executor._pool is None


def test_parallel_executor_joblib_pool_reuse(monkeypatch):
    class FakeParallel:
        def __init__(self, *, n_jobs, prefer, batch_size="auto"):
            self.n_jobs = n_jobs
            self.prefer = prefer
            self.batch_size = batch_size
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited = True

        def __call__(self, iterator):
            return [job() for job in iterator]

    def fake_delayed(fn):
        def wrapper(*args, **kwargs):
            return lambda: fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel",
        FakeParallel,
        raising=False,
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._joblib_delayed",
        fake_delayed,
        raising=False,
    )

    cfg = ParallelConfig(enabled=True, strategy="joblib", max_workers=2, min_batch_size=1)
    executor = ParallelExecutor(cfg)
    with executor as ctx:
        assert isinstance(ctx._pool, FakeParallel)
        assert ctx._pool.entered is True
        result = ctx._joblib_strategy(lambda x: x + 1, [1, 2, 3])
        assert result == [2, 3, 4]
    assert executor._pool is None


def fake_path_factory(entries: dict[str, str]):
    class FakePath:
        def __init__(self, path: str):
            self.path = path

        def __truediv__(self, segment: str):
            base = self.path.rstrip("/")
            return FakePath(f"{base}/{segment}")

        def exists(self) -> bool:
            return self.path in entries

        def read_text(self, encoding: str = "utf-8") -> str:
            if self.path not in entries:
                raise OSError(f"missing {self.path}")
            return entries[self.path]

    return FakePath


def test_get_cgroup_cpu_quota_reads_v2(monkeypatch):
    entries = {"/sys/fs/cgroup/cpu.max": "200000 100000"}
    fake_path = fake_path_factory(entries)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.Path",
        fake_path,
        raising=False,
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.os.name",
        "posix",
        raising=False,
    )
    quota = ParallelExecutor._get_cgroup_cpu_quota()
    assert quota == 2


def test_get_cgroup_cpu_quota_reads_v1(monkeypatch):
    entries = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "500000",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    }
    fake_path = fake_path_factory(entries)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.Path",
        fake_path,
        raising=False,
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.os.name",
        "posix",
        raising=False,
    )
    assert ParallelExecutor._get_cgroup_cpu_quota() == 5


def test_auto_strategy_respects_ci_and_workload(monkeypatch):
    cfg = ParallelConfig(enabled=True, strategy="auto", min_batch_size=8)
    executor = ParallelExecutor(cfg)
    events: list[dict[str, Any]] = []

    def capture(event, payload):
        events.append(payload)

    monkeypatch.setattr(executor, "_emit", capture)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.os.name",
        "posix",
        raising=False,
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.os.cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        ParallelExecutor,
        "_get_cgroup_cpu_quota",
        staticmethod(lambda: None),
    )

    assert executor._auto_strategy(work_items=1) == "sequential"
    assert events[-1]["reason"] == "tiny_workload"

    monkeypatch.setenv("CI", "true")
    assert executor._auto_strategy() == "sequential"
    assert events[-1]["reason"] == "ci_environment"
    monkeypatch.delenv("CI")
    # Also ensure GITHUB_ACTIONS is not interfering if running in real CI
    monkeypatch.setattr(ParallelExecutor, "_is_ci_environment", staticmethod(lambda: False))

    cfg.task_size_hint_bytes = 12 * 1024 * 1024
    assert executor._auto_strategy() == "threads"
    assert events[-1]["reason"] == "large_task_size"
    cfg.task_size_hint_bytes = 0

    assert executor._auto_strategy(work_items=2000) == "sequential"
    assert events[-1]["reason"] == "small_workload"

    assert executor._auto_strategy(work_items=70000) == "processes"
    assert events[-1]["reason"] == "large_instance_workload"
