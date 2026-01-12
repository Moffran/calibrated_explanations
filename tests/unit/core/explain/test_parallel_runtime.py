"""Unit tests for ExplainParallelRuntime."""

import warnings
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from calibrated_explanations.core.explain.parallel_runtime import ExplainParallelRuntime
from calibrated_explanations.parallel import ParallelExecutor, ParallelConfig


class TestExplainParallelRuntime:
    def test_from_explainer_resolves_executor(self):
        explainer = MagicMock()
        explainer.parallel_executor = None
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
        explainer.parallel_executor = None
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
        executor.active_strategy_name = "processes"

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
        executor.active_strategy_name = "sequential"  # Fallback happened

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=10, chunk_size=5
        )

        with pytest.warns(UserWarning, match="Parallel execution fell back to sequential"), runtime:
            pass

    def test_telemetry_duration_tracking(self):
        """Verify that total_duration is updated on exit."""
        executor = MagicMock(spec=ParallelExecutor)
        executor.metrics = MagicMock()
        executor.metrics.total_duration = 0
        executor.config = MagicMock()
        executor.config.enabled = True
        executor.config.strategy = "threads"
        executor.active_strategy_name = "threads"

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=1, chunk_size=1
        )

        with patch("time.perf_counter", side_effect=[100.0, 100.5]), runtime:
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
        executor.active_strategy_name = "sequential"

        runtime = ExplainParallelRuntime(
            executor=executor, min_instances_for_parallel=1, chunk_size=1
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with runtime:
                pass
            assert len(w) == 0

    def test_from_explainer_auto_creates_executor(self):
        """Test that a ParallelExecutor is created if the explainer doesn't have one."""
        explainer = MagicMock()
        # Ensure attributes are missing/None
        explainer.parallel_executor = None
        explainer.executor = None
        # We assume _perf_parallel is not in __dict__ by default for a fresh mock
        explainer.min_instances_for_parallel = None

        runtime = ExplainParallelRuntime.from_explainer(explainer)

        assert runtime.executor is not None
        assert isinstance(runtime.executor, ParallelExecutor)
        # Verify test-friendly defaults
        assert runtime.executor.config.granularity == "instance"
        assert runtime.min_instances_for_parallel == 1

    def test_worker_init_sets_module_global(self):
        """Test that worker init populates the module-level explain_slice."""
        import numpy as np
        from calibrated_explanations.core.explain import parallel_runtime
        from calibrated_explanations.core.explain.parallel_runtime import worker_init_from_explainer_spec

        # We must ensure CalibratedExplainer can be instantiated by the worker init
        # We also mock SequentialExplainExecutor BEFORE init so the captured class is the mock
        with patch("calibrated_explanations.core.calibrated_explainer.CalibratedExplainer") as mock_ce_cls, \
             patch("calibrated_explanations.core.explain.sequential.SequentialExplainExecutor") as mock_seq_cls:
            
            mock_ce_instance = mock_ce_cls.return_value
            # Ensure it doesn't look like Mondrian to avoid shape checks on bins
            mock_ce_instance.is_mondrian = False
            # Ensure other attributes are present if needed
            mock_ce_instance.num_features = 1
            # Ensure condition_source is a string, not a Mock
            mock_ce_instance.condition_source = "observed"

            # Setup return value of predict_calibrated to avoid unpack error
            mock_ce_instance.predict_calibrated.return_value = (
                np.array([0.5]),
                np.array([0.4]),
                np.array([0.6]),
                np.array([1])
            )

            # Setup return value for get_calibration_summaries
            mock_ce_instance.get_calibration_summaries.return_value = ({}, {})

            # Setup the SequentialExplainExecutor mock
            mock_executor_instance = mock_seq_cls.return_value
            mock_executor_instance.execute.return_value = {"results": "mocked"}

            # Setup a minimal spec
            spec = {"learner": MagicMock(), "x_cal": [[1]], "y_cal": [1], "mode": "classification"}
    
            # Run the init - this should pick up the mocked SequentialExplainExecutor
            worker_init_from_explainer_spec(spec)

            assert hasattr(parallel_runtime, "explain_slice")
            assert callable(parallel_runtime.explain_slice)
            assert hasattr(parallel_runtime, "worker_harness")
    
            # Execute the slice
            slice_func = parallel_runtime.explain_slice
            
            # The state dict needs keys expected by explain_slice
            state = {
                "subset": np.array([[1]]),
                "config_state": {},
                "threshold_slice": None,
                "low_high_percentiles": None,
                "bins_slice": None,
                "features_to_ignore_array": np.array([]),
                "feature_filter_per_instance_ignore": None
            }

            result = slice_func(0, 1, state)
            
            # Verify the result comes from our mock
            assert result == {"results": "mocked"}
            # Verify executor was instantiated and called
            mock_seq_cls.assert_called()
            mock_executor_instance.execute.assert_called_once()

    def test_build_explain_execution_plan(self):
        import numpy as np
        from calibrated_explanations.core.explain.parallel_runtime import build_explain_execution_plan, ExplainParallelRuntime
        
        explainer = MagicMock()
        explainer.num_features = 2
        explainer.features_to_ignore = []
        explainer.categorical_features = []
        explainer.parallel_executor = None
        explainer.executor = None
        
        # x input
        x = [[1, 2]]
        
        # Request object
        request = MagicMock()
        # Ensure mocks are traversable
        request.features_to_ignore = []
        request.threshold = 0.5
        request.low_high_percentiles = (5, 95)
        request.bins = None
        request.feature_filter_per_instance_ignore = None
        
        # Mock validate_and_prepare_input from _helpers
        with patch("calibrated_explanations.core.explain.parallel_runtime.validate_and_prepare_input", return_value=np.array(x)), \
             patch("calibrated_explanations.core.explain.parallel_runtime.merge_ignore_features", return_value=np.array([])):
                req, cfg, run = build_explain_execution_plan(explainer, x, request)
                
                np.testing.assert_array_equal(req.x, np.array(x))
                assert cfg.num_features == 2
                assert isinstance(run, ExplainParallelRuntime)


