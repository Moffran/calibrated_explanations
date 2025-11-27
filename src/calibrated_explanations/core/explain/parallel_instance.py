"""Instance-parallel explain executor - parallel execution across instances.

This plugin implements instance-level parallelism, partitioning the input
instances into chunks and processing them in parallel. Each chunk is explained
using the sequential plugin to avoid nested parallelism.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

from ._base import BaseExplainExecutor
from ._helpers import initialize_explanation, slice_bins, slice_threshold
from ._shared import ExplainConfig, ExplainRequest
from .sequential import SequentialExplainExecutor

if TYPE_CHECKING:
    from ...explanations import CalibratedExplanations
    from ..calibrated_explainer import CalibratedExplainer
else:
    CalibratedExplainer = object


class InstanceParallelExplainExecutor(BaseExplainExecutor):
    """Instance-parallel explain execution strategy.

    Partitions test instances into chunks and processes them in parallel,
    with each chunk explained sequentially. This strategy is preferred when
    the number of instances is large relative to the number of features.
    """

    def __init__(self):
        """Initialize the plugin with a sequential plugin for chunk processing."""
        self._sequential_plugin = SequentialExplainExecutor()

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "instance-parallel"

    @property
    def priority(self) -> int:
        """Return plugin priority (highest, checked first)."""
        return 30

    def supports(self, request: ExplainRequest, config: ExplainConfig) -> bool:
        """Return True if instance-level parallelism is enabled and appropriate.

        Requirements:
        - Executor must be available and enabled
        - Granularity must be 'instance'
        - skip_instance_parallel flag must not be set (avoid recursion)
        - Minimum number of instances threshold met
        """
        if config.executor is None:
            return False
        if not config.executor.config.enabled:
            return False
        if config.granularity != "instance":
            return False
        if request.skip_instance_parallel:
            # Prevent recursive instance parallelism
            return False
        # Check minimum instances threshold
        n_instances = request.x.shape[0] if request.x.ndim > 0 else 0
        return n_instances >= config.min_instances_for_parallel

    def execute(
        self,
        request: ExplainRequest,
        config: ExplainConfig,
        explainer: CalibratedExplainer,
    ) -> CalibratedExplanations:
        """Execute instance-parallel explain operation.

        This implementation:
        1. Partitions instances into chunks
        2. Creates per-chunk requests with sliced thresholds/bins
        3. Processes chunks in parallel using the sequential plugin
        4. Combines chunk results into a unified explanation
        """
        x_input = request.x
        features_to_ignore_array = request.features_to_ignore
        executor = config.executor

        n_instances = x_input.shape[0]
        if n_instances == 0:
            # Handle empty input gracefully
            empty_explanation = initialize_explanation(
                explainer,
                x_input,
                request.low_high_percentiles,
                request.threshold,
                request.bins,
                features_to_ignore_array,
            )
            explainer.latest_explanation = empty_explanation
            explainer._last_explanation_mode = explainer._infer_explanation_mode()
            return empty_explanation

        chunk_size = max(1, executor.config.min_batch_size)
        total_start_time = time()

        # Step 1: Partition instances into chunks
        ranges: List[Tuple[int, int, Any, Any]] = []
        for start in range(0, n_instances, chunk_size):
            stop = min(start + chunk_size, n_instances)
            threshold_slice = slice_threshold(request.threshold, start, stop, n_instances)
            bins_slice = slice_bins(request.bins, start, stop)
            ranges.append((start, stop, threshold_slice, bins_slice))

        # Step 2: Handle single chunk case without parallelism
        if len(ranges) == 1:
            start, stop, threshold_slice, bins_slice = ranges[0]
            subset = np.asarray(x_input[start:stop])
            chunk_request = ExplainRequest(
                x=subset,
                threshold=threshold_slice,
                low_high_percentiles=request.low_high_percentiles,
                bins=bins_slice,
                features_to_ignore=features_to_ignore_array,
                use_plugin=False,
                skip_instance_parallel=True,  # Prevent recursive parallelism
            )
            result = self._sequential_plugin.execute(chunk_request, config, explainer)
            explainer.latest_explanation = result
            explainer._last_explanation_mode = explainer._infer_explanation_mode()
            return result

        # Step 3: Build parallel tasks
        tasks: List[Tuple[int, np.ndarray, Any, Any]] = [
            (start, np.asarray(x_input[start:stop]), threshold_slice, bins_slice)
            for start, stop, threshold_slice, bins_slice in ranges
        ]

        # Step 4: Define worker function that invokes sequential plugin
        def _instance_parallel_task(
            task: Tuple[int, np.ndarray, Any, Any],
        ) -> Tuple[int, CalibratedExplanations]:
            """Execute a single instance-chunk explanation task."""
            start_idx, subset, threshold_slice, bins_slice = task
            chunk_request = ExplainRequest(
                x=subset,
                threshold=threshold_slice,
                low_high_percentiles=request.low_high_percentiles,
                bins=bins_slice,
                features_to_ignore=features_to_ignore_array,
                use_plugin=False,
                skip_instance_parallel=True,  # Prevent recursive parallelism
            )
            result = self._sequential_plugin.execute(chunk_request, config, explainer)
            return start_idx, result

        # Step 5: Execute tasks in parallel
        ordered_results = sorted(
            executor.map(_instance_parallel_task, tasks, work_items=n_instances),
            key=lambda item: item[0],
        )

        # Step 6: Combine chunk results into unified explanation
        combined = initialize_explanation(
            explainer,
            x_input,
            request.low_high_percentiles,
            request.threshold,
            request.bins,
            features_to_ignore_array,
        )
        combined.explanations = []
        offset = 0

        for _, chunk_result in ordered_results:
            for explanation in chunk_result.explanations:
                # Update parent references and indices
                explanation.calibrated_explanations = combined
                explanation.index = offset
                explanation.x_test = combined.x_test
                combined.explanations.append(explanation)
                offset += 1

        # Step 7: Finalize metadata
        combined.current_index = combined.start_index
        combined.end_index = len(combined.x_test)
        combined.total_explain_time = time() - total_start_time

        # Update explainer state
        explainer.latest_explanation = combined
        explainer._last_explanation_mode = explainer._infer_explanation_mode()

        return combined


__all__ = ["InstanceParallelExplainExecutor"]
