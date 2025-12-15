"""Tests for the ParallelBenchmark helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from calibrated_explanations.core.explain.benchmarks import ParallelBenchmark


@pytest.fixture(autouse=True)
def stub_parallel_executor(monkeypatch: pytest.MonkeyPatch):
    """Replace the heavy ParallelExecutor with a lightweight stub."""

    class FakeExecutor:
        def __init__(self, config):
            self.config = config
            self.metrics = SimpleNamespace(snapshot=lambda: {"submitted": 3, "completed": 3})

    monkeypatch.setattr(
        "calibrated_explanations.core.explain.benchmarks.ParallelExecutor",
        FakeExecutor,
    )


def test_parallel_benchmark_collects_results(monkeypatch: pytest.MonkeyPatch):
    """Benchmark sequential and threaded runs with deterministic timing."""

    class DummyExplainer:
        def __init__(self):
            self._perf_parallel = None
            self.executions = 0

        def explain(self, data):
            self.executions += 1
            return [{"result": value} for value in data]

    ticks = iter([0.0, 0.5, 1.0, 1.8])
    monkeypatch.setattr(
        "calibrated_explanations.core.explain.benchmarks.time.perf_counter",
        lambda: next(ticks),
    )

    benchmark = ParallelBenchmark(DummyExplainer(), data=[1, 2, 3])
    results = benchmark.run(strategies=["sequential", "threads"], worker_counts=[2])

    assert benchmark.explainer.executions == 2
    assert len(results) == 2
    assert all(result.metrics["completed"] == 3 for result in results)

    report = benchmark.report()
    assert "Parallel Execution Benchmark Results" in report
    assert "threads" in report
