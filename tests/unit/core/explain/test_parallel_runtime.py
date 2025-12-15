"""Unit tests for ExplainParallelRuntime."""

import warnings
from unittest.mock import MagicMock, patch
import pytest
from calibrated_explanations.core.explain.parallel_runtime import ExplainParallelRuntime
from calibrated_explanations.parallel import ParallelExecutor, ParallelConfig


class TestExplainParallelRuntime:
    def test_from_explainer_resolves_executor(self):
        explainer = MagicMock()
        explainer._perf_parallel = None
        explainer.min_instances_for_parallel = None  # Fix: prevent MagicMock return
        executor = MagicMock(spec=ParallelExecutor)
        executor.config = ParallelConfig(min_instances_for_parallel=10, instance_chunk_size=5)
        explainer.executor = executor

        runtime = ExplainParallelRuntime.from_explainer(explainer)

        assert runtime.executor is executor
        assert runtime.min_instances_for_parallel == 10
        assert runtime.chunk_size == 5

    def test_from_explainer_ignores_granularity_arg(self):
        explainer = MagicMock()
        explainer._perf_parallel = None
        explainer.executor = None
        explainer.min_instances_for_parallel = None  # Fix: prevent MagicMock return

        # Passing granularity should not affect the runtime object itself as it was removed
        runtime = ExplainParallelRuntime.from_explainer(explainer, granularity="instance")

        assert not hasattr(runtime, "granularity")

    def test_build_config_derives_granularity_from_executor(self):
        explainer = MagicMock()
        explainer.num_features = 5

        executor = MagicMock(spec=ParallelExecutor)
        executor.config = ParallelConfig(granularity="instance")

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=10, chunk_size=5
        )

        config = runtime.build_config(explainer)
        assert config.granularity == "instance"

    def test_build_config_defaults_granularity_to_none_without_executor(self):
        explainer = MagicMock()
        explainer.num_features = 5

        runtime = ExplainParallelRuntime(executor=None, min_instances_for_parallel=10, chunk_size=5)

        config = runtime.build_config(explainer)
        assert config.granularity == "none"

    def test_context_manager_delegates_to_executor(self):
        executor = MagicMock(spec=ParallelExecutor)
        executor.metrics = MagicMock()
        executor.metrics.total_duration = 0
        executor.config = MagicMock()
        executor.config.enabled = True
        executor.config.strategy = "processes"
        executor._active_strategy_name = "processes"

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=10, chunk_size=5
        )

        with runtime:
            pass

        executor.__enter__.assert_called_once()
        executor.__exit__.assert_called_once()

    def test_cancel_delegates_to_executor(self):
        executor = MagicMock(spec=ParallelExecutor)
        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=10, chunk_size=5
        )

        runtime.cancel()
        executor.cancel.assert_called_once()

    def test_telemetry_returns_snapshot(self):
        executor = MagicMock(spec=ParallelExecutor)
        executor.metrics = MagicMock()
        executor.metrics.snapshot.return_value = {"foo": "bar"}

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=10, chunk_size=5
        )

        assert runtime.telemetry == {"foo": "bar"}

    def test_fallback_warning(self):
        executor = MagicMock(spec=ParallelExecutor)
        executor.metrics = MagicMock()
        executor.metrics.total_duration = 0
        executor.config = MagicMock()
        executor.config.enabled = True
        executor.config.strategy = "processes"
        executor._active_strategy_name = "sequential"  # Fallback happened

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=10, chunk_size=5
        )

        with pytest.warns(UserWarning, match="Parallel execution fell back to sequential"):
            with runtime:
                pass

    def test_telemetry_duration_tracking(self):
        """Verify that total_duration is updated on exit."""
        executor = MagicMock(spec=ParallelExecutor)
        executor.metrics = MagicMock()
        executor.metrics.total_duration = 0
        executor.config = MagicMock()
        executor.config.enabled = True
        executor.config.strategy = "threads"
        executor._active_strategy_name = "threads"

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=1, chunk_size=1
        )

        with patch("time.perf_counter", side_effect=[100.0, 100.5]):
            with runtime:
                pass

        # The runtime adds the duration to the existing total_duration
        assert executor.metrics.total_duration == 0.5

    def test_no_fallback_warning_when_expected(self):
        """Verify that no warning is issued when sequential was requested."""
        executor = MagicMock(spec=ParallelExecutor)
        executor.metrics = MagicMock()
        executor.metrics.total_duration = 0
        executor.config = MagicMock()
        executor.config.enabled = True
        executor.config.strategy = "sequential"
        executor._active_strategy_name = "sequential"

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=1, chunk_size=1
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with runtime:
                pass
            assert len(w) == 0
