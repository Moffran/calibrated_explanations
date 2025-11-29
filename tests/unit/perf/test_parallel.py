from functools import partial

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

    def map(self, fn, items):
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
    }


def test_parallel_config_from_env(monkeypatch):
    base = ParallelConfig(enabled=False, strategy="sequential", max_workers=4, min_batch_size=8)
    monkeypatch.delenv("CE_PARALLEL", raising=False)
    assert ParallelConfig.from_env(base).enabled is False

    monkeypatch.setenv("CE_PARALLEL", "1")
    cfg = ParallelConfig.from_env(base)
    assert cfg.enabled is True
    assert cfg.strategy == "sequential"

    monkeypatch.setenv("CE_PARALLEL", "off,threads,workers=10,min_batch=2,enable")
    cfg = ParallelConfig.from_env(base)
    assert cfg.enabled is True
    assert cfg.strategy == "threads"
    assert cfg.max_workers == 10
    assert cfg.min_batch_size == 2


def test_map_handles_disabled_and_small_batches():
    config = ParallelConfig(enabled=False)
    executor = ParallelExecutor(config)
    assert executor.map(lambda x: x + 1, [1, 2]) == [2, 3]

    config = ParallelConfig(enabled=True, min_batch_size=5)
    executor = ParallelExecutor(config)
    assert executor.map(lambda x: x + 1, [1, 2]) == [2, 3]
    assert executor.metrics.submitted == 0


def test_map_uses_strategy_and_updates_metrics(monkeypatch):
    config = ParallelConfig(enabled=True, strategy="sequential", min_batch_size=1)
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

    config = ParallelConfig(enabled=True, strategy="threads", min_batch_size=1, telemetry=telemetry)
    executor = ParallelExecutor(config)
    monkeypatch.setattr(executor, "_resolve_strategy", lambda: failing_strategy)
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
    monkeypatch.setattr(executor, "_auto_strategy", lambda: "threads")
    assert executor._resolve_strategy().func.__name__ == "_thread_strategy"


def test_auto_strategy(monkeypatch):
    config = ParallelConfig(enabled=True, strategy="auto")
    executor = ParallelExecutor(config)

    monkeypatch.setattr("calibrated_explanations.parallel.parallel.os.name", "nt", raising=False)
    assert executor._auto_strategy() == "threads"

    monkeypatch.setattr("calibrated_explanations.parallel.parallel.os.name", "posix", raising=False)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.os.cpu_count", lambda: 1, raising=False
    )
    assert executor._auto_strategy() == "threads"

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel.os.cpu_count", lambda: 8, raising=False
    )
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", object(), raising=False
    )
    assert executor._auto_strategy() == "joblib"

    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )
    assert executor._auto_strategy() == "processes"


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

    def fake_thread(fn, items, workers=None):
        calls["thread"] = True
        return [fn(item) for item in items]

    monkeypatch.setattr(executor, "_thread_strategy", fake_thread)
    monkeypatch.setattr(
        "calibrated_explanations.parallel.parallel._JoblibParallel", None, raising=False
    )
    assert executor._joblib_strategy(echo, [1, 2]) == [1, 2]
    assert calls["thread"]

    class FakeParallel:
        def __init__(self, *, n_jobs, prefer):
            self.n_jobs = n_jobs
            self.prefer = prefer

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
    executor._emit("test", {"value": 1})
    assert events == [("test", {"value": 1})]

    def broken(event, payload):
        raise RuntimeError("ignored")

    config = ParallelConfig(enabled=True, telemetry=broken)
    executor = ParallelExecutor(config)
    executor._emit("test", {"value": 2})  # should not raise
