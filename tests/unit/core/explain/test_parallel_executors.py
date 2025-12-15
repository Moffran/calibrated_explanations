"""Tests for parallel explanation execution strategies.

Tests for instance-parallel and feature-parallel explanation executors
that partition workload and delegate to sequential executor for chunks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from calibrated_explanations.core.explain._shared import ExplainConfig, ExplainRequest

# from calibrated_explanations.core.explain.parallel_feature import FeatureParallelExplainExecutor
from calibrated_explanations.core.explain.parallel_instance import InstanceParallelExplainExecutor


class TestInstanceParallelExecutor:
    """Tests for instance-parallel explanation strategy."""

    def test_should_have_correct_name_and_priority(self):
        """Executor should identify itself with correct name and priority."""
        executor = InstanceParallelExplainExecutor()
        assert executor.name == "instance-parallel"
        assert executor.priority == 30

    def test_should_support_when_executor_enabled_and_enough_instances(self):
        """Executor should support when parallelization is enabled and threshold met."""
        executor = InstanceParallelExplainExecutor()

        # Create mock request with sufficient instances
        request = MagicMock(spec=ExplainRequest)
        request.x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        request.skip_instance_parallel = False

        # Create mock config with executor enabled
        config = MagicMock(spec=ExplainConfig)
        config.executor = MagicMock()
        config.executor.config.enabled = True
        config.granularity = "instance"
        config.min_instances_for_parallel = 2

        assert executor.supports(request, config) is True

    def test_should_not_support_when_executor_disabled(self):
        """Executor should not support when executor is disabled."""
        executor = InstanceParallelExplainExecutor()

        request = MagicMock(spec=ExplainRequest)
        request.x = np.array([[1, 2], [3, 4]])
        request.skip_instance_parallel = False

        config = MagicMock(spec=ExplainConfig)
        config.executor = MagicMock()
        config.executor.config.enabled = False
        config.granularity = "instance"
        config.min_instances_for_parallel = 2

        assert executor.supports(request, config) is False

    def test_should_not_support_when_executor_is_none(self):
        """Executor should not support when no executor is configured."""
        executor = InstanceParallelExplainExecutor()

        request = MagicMock(spec=ExplainRequest)
        request.x = np.array([[1, 2], [3, 4]])
        request.skip_instance_parallel = False

        config = MagicMock(spec=ExplainConfig)
        config.executor = None
        config.granularity = "instance"
        config.min_instances_for_parallel = 2

        assert executor.supports(request, config) is False

    def test_should_not_support_when_wrong_granularity(self):
        """Executor should not support when granularity is not instance-level."""
        executor = InstanceParallelExplainExecutor()

        request = MagicMock(spec=ExplainRequest)
        request.x = np.array([[1, 2], [3, 4]])
        request.skip_instance_parallel = False

        config = MagicMock(spec=ExplainConfig)
        config.executor = MagicMock()
        config.executor.config.enabled = True
        config.granularity = "sequential"  # Wrong granularity
        config.min_instances_for_parallel = 2

        assert executor.supports(request, config) is False

    def test_should_not_support_when_recursion_flag_set(self):
        """Executor should not support when skip_instance_parallel flag is set."""
        executor = InstanceParallelExplainExecutor()

        request = MagicMock(spec=ExplainRequest)
        request.x = np.array([[1, 2], [3, 4]])
        request.skip_instance_parallel = True  # Recursion flag set

        config = MagicMock(spec=ExplainConfig)
        config.executor = MagicMock()
        config.executor.config.enabled = True
        config.granularity = "instance"
        config.min_instances_for_parallel = 2

        assert executor.supports(request, config) is False

    def test_should_not_support_when_below_minimum_instances(self):
        """Executor should not support when instance count is below threshold."""
        executor = InstanceParallelExplainExecutor()

        request = MagicMock(spec=ExplainRequest)
        request.x = np.array([[1, 2]])  # Only 1 instance
        request.skip_instance_parallel = False

        config = MagicMock(spec=ExplainConfig)
        config.executor = MagicMock()
        config.executor.config.enabled = True
        config.granularity = "instance"
        config.min_instances_for_parallel = 4  # Threshold is 4

        assert executor.supports(request, config) is False

    def test_should_handle_zero_dim_array_gracefully(self):
        """Executor should handle zero-dimensional arrays gracefully."""
        executor = InstanceParallelExplainExecutor()

        request = MagicMock(spec=ExplainRequest)
        request.x = np.array(5)  # 0-dimensional array
        request.skip_instance_parallel = False

        config = MagicMock(spec=ExplainConfig)
        config.executor = MagicMock()
        config.executor.config.enabled = True
        config.granularity = "instance"
        config.min_instances_for_parallel = 2

        # Should return False because 0 instances < 2
        assert executor.supports(request, config) is False

    def test_should_initialize_with_sequential_plugin(self):
        """Executor should initialize with a sequential explanation plugin."""
        executor = InstanceParallelExplainExecutor()

        # Sequential plugin should be created and stored
        assert hasattr(executor, "_sequential_plugin")
        assert executor._sequential_plugin is not None


# class TestFeatureParallelExecutor:
#     """Tests for feature-parallel explanation strategy."""

#     def test_should_have_correct_name_and_priority(self):
#         """Executor should identify itself with correct name and priority."""
#         executor = FeatureParallelExplainExecutor()
#         assert executor.name == "feature-parallel"
#         assert executor.priority == 20

#     def test_should_support_when_executor_enabled_and_feature_granularity(self):
#         """Executor should support when executor is enabled and granularity is feature."""
#         executor = FeatureParallelExplainExecutor()

#         # Create mock request (skip_instance_parallel=False means OK)
#         request = MagicMock(spec=ExplainRequest)
#         request.skip_instance_parallel = False  # NOT set

#         # Create mock config with executor enabled
#         config = MagicMock(spec=ExplainConfig)
#         config.executor = MagicMock()
#         config.executor.config.enabled = True
#         config.granularity = "feature"

#         # The logic returns: skip_instance_parallel OR (granularity != "instance")
#         # So if skip_instance_parallel=False and granularity="feature", returns (False or True) = True
#         result = executor.supports(request, config)
#         assert result is True

#     def test_should_not_support_when_executor_disabled(self):
#         """Executor should not support when executor is disabled."""
#         executor = FeatureParallelExplainExecutor()

#         request = MagicMock(spec=ExplainRequest)
#         request.skip_instance_parallel = False

#         config = MagicMock(spec=ExplainConfig)
#         config.executor = MagicMock()
#         config.executor.config.enabled = False
#         config.granularity = "feature"

#         assert executor.supports(request, config) is False

#     def test_should_not_support_when_executor_is_none(self):
#         """Executor should not support when no executor is configured."""
#         executor = FeatureParallelExplainExecutor()

#         request = MagicMock(spec=ExplainRequest)
#         request.skip_instance_parallel = False

#         config = MagicMock(spec=ExplainConfig)
#         config.executor = None
#         config.granularity = "feature"

#         assert executor.supports(request, config) is False

#     def test_should_not_support_when_wrong_granularity(self):
#         """Executor should not support when granularity is not feature-level."""
#         executor = FeatureParallelExplainExecutor()

#         request = MagicMock(spec=ExplainRequest)
#         request.skip_instance_parallel = False

#         config = MagicMock(spec=ExplainConfig)
#         config.executor = MagicMock()
#         config.executor.config.enabled = True
#         config.granularity = "instance"  # Wrong granularity

#         assert executor.supports(request, config) is False

#     def test_should_support_when_skip_instance_parallel_set(self):
#         """Executor should support when skip_instance_parallel is True."""
#         executor = FeatureParallelExplainExecutor()

#         request = MagicMock(spec=ExplainRequest)
#         request.skip_instance_parallel = True  # Set to True

#         config = MagicMock(spec=ExplainConfig)
#         config.executor = MagicMock()
#         config.executor.config.enabled = True
#         config.granularity = "feature"  # Must be "feature" to not fail earlier check

#         # The logic is: skip_instance_parallel OR (granularity != "instance")
#         # With granularity="feature": True OR ("feature" != "instance") = True OR True = True
#         assert executor.supports(request, config) is True
