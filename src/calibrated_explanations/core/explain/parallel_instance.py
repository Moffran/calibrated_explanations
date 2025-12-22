"""Instance-parallel explain executor - parallel execution across instances.

This plugin implements instance-level parallelism, partitioning the input
instances into chunks and processing them in parallel. Each chunk is explained
using the sequential plugin to avoid nested parallelism.
"""

from __future__ import annotations

import contextlib
import os
import warnings
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

MIN_CHUNK_SIZE = 100  # Minimum chunk size to amortize overhead


def _instance_parallel_task(task: Tuple[int, int, dict]) -> Tuple[int, CalibratedExplanations]:
    """Execute a single instance-chunk explanation task.

    The task payload is intentionally small to optimise the hot path when
    shipping work to worker processes. It contains `(start, stop, state)`;
    when a worker harness is present (`parallel_runtime._worker_harness`) we
    delegate to it. Otherwise, rebuild the local request/explainer and run
    the sequential plugin as a fallback.
    """
    start_idx, stop_idx, serialized_state = task

    # If a worker harness is installed (via worker initializer), prefer it.
    # ADR002_ALLOW: optional worker harness is best-effort.
    with contextlib.suppress(Exception):
        import calibrated_explanations.core.explain.parallel_runtime as pr_mod

        harness = getattr(pr_mod, "_worker_harness", None)
        if harness is not None and hasattr(harness, "explain_slice"):
            result = harness.explain_slice(start_idx, stop_idx, serialized_state)
            return start_idx, result

    # Local fallback: unpack full payload and run sequential plugin
    subset = serialized_state.get("subset")
    threshold_slice = serialized_state.get("threshold_slice")
    bins_slice = serialized_state.get("bins_slice")
    low_high_percentiles = serialized_state.get("low_high_percentiles")
    features_to_ignore_array = serialized_state.get("features_to_ignore_array")
    features_to_ignore_per_instance = serialized_state.get("features_to_ignore_per_instance")
    explainer = serialized_state.get("explainer")
    config_state = serialized_state.get("config_state")

    config = ExplainConfig(**config_state)
    config.executor = None

    chunk_request = ExplainRequest(
        x=subset,
        threshold=threshold_slice,
        low_high_percentiles=low_high_percentiles,
        bins=bins_slice,
        features_to_ignore=features_to_ignore_array,
        features_to_ignore_per_instance=features_to_ignore_per_instance,
        use_plugin=False,
        skip_instance_parallel=True,
    )

    plugin = SequentialExplainExecutor()
    result = plugin.execute(chunk_request, config, explainer)
    return start_idx, result


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
        features_to_ignore_per_instance = getattr(request, "features_to_ignore_per_instance", None)

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

        # Determine chunk size: prefer executor config if set, else fallback to ExplainConfig default
        chunk_size = max(1, config.chunk_size)
        if executor:
            if executor.config.instance_chunk_size:
                chunk_size = max(1, executor.config.instance_chunk_size)
            else:
                # Dynamic chunking heuristic:
                # 1. Aim to utilize all workers (n_instances // n_workers)
                # 2. Enforce a minimum chunk size (e.g. 200) to amortize overhead

                # Use a larger minimum chunk to avoid tiny tasks that are dominated by pickling/spawn overhead
                # Use a larger minimum chunk to avoid tiny tasks that are dominated by pickling/spawn overhead
                # Sequential execution is now very fast (~5ms/instance), so we need substantial chunks.
                pass

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
            per_instance_chunk = None
            if features_to_ignore_per_instance is not None:
                per_instance_chunk = features_to_ignore_per_instance[start:stop]
            chunk_request = ExplainRequest(
                x=subset,
                threshold=threshold_slice,
                low_high_percentiles=request.low_high_percentiles,
                bins=bins_slice,
                features_to_ignore=features_to_ignore_array,
                features_to_ignore_per_instance=per_instance_chunk,
                use_plugin=False,
                skip_instance_parallel=True,  # Prevent recursive parallelism
            )
            result = self._sequential_plugin.execute(chunk_request, config, explainer)
            explainer.latest_explanation = result
            explainer._last_explanation_mode = explainer._infer_explanation_mode()
            return result

        # Prepare sanitized config state (exclude executor to avoid pickling issues)
        config_state = {k: v for k, v in config.__dict__.items() if k != "executor"}

        # Step 3: Build parallel tasks
        tasks: List[tuple] = []
        for start, stop, threshold_slice, bins_slice in ranges:
            subset = np.asarray(x_input[start:stop])
            per_instance_chunk = (
                features_to_ignore_per_instance[start:stop]
                if features_to_ignore_per_instance is not None
                else None
            )

            # If a worker initializer is configured for the executor, prefer
            # shipping a minimal, picklable spec instead of the full explainer
            # object to worker processes. This keeps the hot path small and
            # avoids sending unpicklable objects across process boundaries.
            if executor and getattr(executor.config, "worker_initializer", None) is not None:
                # Compact state for worker harness. The harness receives the
                # full explainer spec at pool init time and therefore only
                # needs start/stop and light state here.
                serialized_state = {
                    "subset": subset,
                    "threshold_slice": threshold_slice,
                    "bins_slice": bins_slice,
                    "low_high_percentiles": request.low_high_percentiles,
                    "features_to_ignore_array": features_to_ignore_array,
                    "features_to_ignore_per_instance": per_instance_chunk,
                    "config_state": config_state,
                }
            else:
                # Fallback: include the explainer so in-process workers can
                # reconstruct full execution when no harness is available.
                serialized_state = {
                    "subset": subset,
                    "threshold_slice": threshold_slice,
                    "bins_slice": bins_slice,
                    "low_high_percentiles": request.low_high_percentiles,
                    "features_to_ignore_array": features_to_ignore_array,
                    "features_to_ignore_per_instance": per_instance_chunk,
                    "explainer": explainer,
                    "config_state": config_state,
                }

            tasks.append((start, stop, serialized_state))

        # Step 5: Execute tasks in parallel
        # Note: _instance_parallel_task is now top-level
        active_strategy = getattr(executor, "_active_strategy_name", None) or getattr(
            executor.config, "strategy", "auto"
        )
        if active_strategy == "auto" and hasattr(executor, "_auto_strategy"):
            active_strategy = executor._auto_strategy(work_items=n_instances)

        configured_strategy = getattr(executor.config, "strategy", active_strategy)
        if os.name == "nt" and active_strategy in {"threads", "sequential"}:
            if configured_strategy not in {"threads", "sequential"}:
                warnings.warn(
                    "Instance-parallel execution on Windows is using a thread/sequential fallback. "
                    "Set CE_PARALLEL=processes or CE_PARALLEL=joblib to keep process-based execution.",
                    UserWarning,
                    stacklevel=2,
                )

        # Route through ParallelExecutor.map() only for the real facade so its
        # telemetry (metrics.submitted/completed) stays accurate (used by the
        # evaluation harness). For other executors/test doubles, call the
        # thread strategy directly.
        results = executor.map(_instance_parallel_task, tasks, work_items=n_instances)

        ordered_results = sorted(results, key=lambda item: item[0])

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
