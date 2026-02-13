"""Unit tests for ExplainParallelRuntime."""

import warnings
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from calibrated_explanations.core.explain.parallel_runtime import ExplainParallelRuntime
from calibrated_explanations.parallel import ParallelExecutor, ParallelConfig


class TestExplainParallelRuntime:



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




    def test_worker_init_sets_module_global(self):
        """Test that worker init populates the module-level explain_slice."""
        from calibrated_explanations.core.explain import parallel_runtime
        from calibrated_explanations.core.explain.parallel_runtime import (
            worker_init_from_explainer_spec,
        )

        # We must ensure CalibratedExplainer can be instantiated by the worker init
        # We also mock SequentialExplainExecutor BEFORE init so the captured class is the mock
        with (
            patch(
                "calibrated_explanations.core.calibrated_explainer.CalibratedExplainer"
            ) as mock_ce_cls,
            patch(
                "calibrated_explanations.core.explain.sequential.SequentialExplainExecutor"
            ) as mock_seq_cls,
        ):
            mock_ce_instance = mock_ce_cls.return_value
            # Ensure it doesn't look like Mondrian to avoid shape checks on bins
            mock_ce_instance.is_mondrian = False
            # Ensure other attributes are present if needed
            mock_ce_instance.num_features = 1
            # Ensure condition_source is a string, not a Mock
            mock_ce_instance.condition_source = "observed"

            mock_ce_instance.prediction_orchestrator = MagicMock()

            # Setup return value of predict_internal to avoid unpack error
            mock_ce_instance.prediction_orchestrator.predict_internal.return_value = (
                np.array([0.5]),
                np.array([0.4]),
                np.array([0.6]),
                np.array([1]),
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
                "feature_filter_per_instance_ignore": None,
            }

            result = slice_func(0, 1, state)

            # Verify the result comes from our mock
            assert result == {"results": "mocked"}
            # Verify executor was instantiated and called
            mock_seq_cls.assert_called()
            mock_executor_instance.execute.assert_called_once()

