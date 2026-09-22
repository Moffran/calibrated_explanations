import pytest
import time
from calibrated_explanations.parallel import ParallelConfig, ParallelExecutor


def square(x):
    return x * x


def slow_square(x):
    time.sleep(0.01)
    return x * x


def nested_task(x):
    # Simulate nested parallelism by creating a new executor
    # Note: Passing the original executor is hard due to pickling
    config = ParallelConfig(enabled=True, strategy="threads", max_workers=2)
    executor = ParallelExecutor(config)
    return sum(executor.map(square, range(x)))


class TestParallelLifecycle:
    @pytest.mark.parametrize("strategy", ["threads", "processes", "joblib"])
    def test_strategies(self, strategy):
        if strategy == "joblib":
            pytest.importorskip("joblib")

        config = ParallelConfig(enabled=True, strategy=strategy, max_workers=2, min_batch_size=1)
        executor = ParallelExecutor(config)

        try:
            results = executor.map(square, range(10))
        except PermissionError as exc:
            if getattr(exc, "winerror", None) == 5 and strategy in {"processes", "joblib"}:
                pytest.skip("Skipping process/joblib strategy on restricted Windows environment.")
            raise
        assert results == [x * x for x in range(10)]

    def test_context_manager(self):
        config = ParallelConfig(
            enabled=True,
            strategy="threads",
            max_workers=2,
            min_batch_size=1,
            min_instances_for_parallel=1,
        )

        with ParallelExecutor(config) as executor:
            results1 = executor.map(square, range(5))
            results2 = executor.map(square, range(5, 10))

        assert results1 == [x * x for x in range(5)]
        assert results2 == [x * x for x in range(5, 10)]

    def test_should_clear_pool_reference_when_executor_exits_context(self):
        """should_release_executor_pool_reference_when_context_exits."""
        config = ParallelConfig(enabled=True, strategy="threads", max_workers=2, min_batch_size=1)

        executor = ParallelExecutor(config)
        with executor:
            _ = executor.map(square, range(3))

        assert executor.pool is None

    def test_force_serial_on_failure(self, caplog, enable_fallbacks):
        import logging

        config = ParallelConfig(
            enabled=True,
            strategy="threads",
            force_serial_on_failure=True,
            min_batch_size=1,
            min_instances_for_parallel=1,
        )
        executor = ParallelExecutor(config)

        # Inject a failure in strategy resolution to trigger the serial fallback.
        def failing_resolve(*args, **kwargs):
            raise RuntimeError("Simulated failure")

        executor.resolve_strategy = failing_resolve

        with (
            caplog.at_level(logging.INFO, logger="calibrated_explanations"),
            pytest.warns(UserWarning, match=r"Simulated failure.*falling back to sequential"),
        ):
            results = executor.map(square, range(5))

        assert results == [x * x for x in range(5)]
        assert executor.metrics.fallbacks == 1
        assert any(
            "Parallel execution failed" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        )
