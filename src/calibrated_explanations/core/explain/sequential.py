"""Sequential explain executor - single-threaded execution strategy.

This plugin implements the traditional single-threaded explanation path,
processing features sequentially without parallelism. It serves as the
fallback strategy and reference implementation for behavioralequivalence tests.
"""

from __future__ import annotations

import contextlib
from time import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from ._base import BaseExplainExecutor
from ._helpers import initialize_explanation, merge_feature_result
from ._legacy_explain import explain_predict_step
from ._shared import (
    ExplainConfig,
    ExplainRequest,
    build_feature_tasks,
    finalize_explanation,
)

if TYPE_CHECKING:
    from ...explanations import CalibratedExplanations
    from ..calibrated_explainer import CalibratedExplainer
else:
    CalibratedExplainer = object


class SequentialExplainExecutor(BaseExplainExecutor):
    """Sequential explain execution strategy.

    Processes all test instances and features in a single thread.
    This is the default fallback when parallelism is disabled or unavailable.

    Notes
    -----
    **Memory considerations**: The sequential executor materializes all perturbed
    instances during perturbation generation in `explain_predict_step`. For very
    large datasets (e.g., 1000+ instances × 1000+ features), this can require
    significant memory. Users experiencing memory exhaustion should consider:

    1. Reducing test set size (fewer instances)
    2. Enabling parallelism (which chunks instances internally)
    3. Reducing number of features via feature selection

    A future optimization will implement lazy perturbation generation or instance-
    level chunking in sequential mode to better handle large datasets (tracked as
    a memory efficiency improvement).
    """

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "sequential"

    @property
    def priority(self) -> int:
        """Return plugin priority (lowest, used as fallback)."""
        return 10

    def supports(self, request: ExplainRequest, config: ExplainConfig) -> bool:
        """Return True - sequential plugin always supports any request.

        This is the universal fallback plugin that handles:
        - No executor available
        - Executor disabled
        - Explicit granularity='none'
        - Any configuration that other plugins reject
        """
        # Reference args to satisfy linters (kept for signature compatibility)
        _ = request
        _ = config
        # Sequential always supports - it's the universal fallback
        return True

    def execute(
        self,
        request: ExplainRequest,
        config: ExplainConfig,
        explainer: CalibratedExplainer,
    ) -> CalibratedExplanations:
        """Execute sequential explain operation.

        This implementation mirrors the original CalibratedExplainer.explain
        sequential path (lines 2365-2595) to ensure behavioral parity.
        """
        # Import _feature_task from feature_task module (deferred to avoid circular import)
        from .feature_task import feature_task  # pylint: disable=import-outside-toplevel

        x_input = request.x
        features_to_ignore_array = request.features_to_ignore
        feature_filter_per_instance_ignore = getattr(
            request, "feature_filter_per_instance_ignore", None
        )

        # Track total explanation time
        total_start_time = time()

        # Initialize explanation object
        explanation = initialize_explanation(
            explainer,
            x_input,
            request.low_high_percentiles,
            request.threshold,
            request.bins,
            features_to_ignore_array,
        )

        instance_start_time = time()

        # Step 1: Get predictions for original test instances
        import inspect  # local import to avoid top-level dependency

        predict_fn = explain_predict_step
        call_kwargs: dict = {
            "feature_filter_per_instance_ignore": feature_filter_per_instance_ignore
        }
        # Only include interval_summary if the target function supports it
        with contextlib.suppress(Exception):  # adr002_allow  # pragma: no cover - defensive
            sig = inspect.signature(predict_fn)
            if "interval_summary" in sig.parameters:
                call_kwargs["interval_summary"] = request.interval_summary

        (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
            perturbed_threshold,
            perturbed_bins,
            perturbed_x,
            perturbed_class,
        ) = predict_fn(
            explainer,
            x_input,
            request.threshold,
            request.low_high_percentiles,
            request.bins,
            features_to_ignore_array,
            **call_kwargs,
        )

        # Step 2: Initialize data structures to store feature-level results
        n_instances = x_input.shape[0]
        num_features = config.num_features

        weights_predict = np.zeros((n_instances, num_features))
        weights_low = np.zeros((n_instances, num_features))
        weights_high = np.zeros((n_instances, num_features))
        predict_matrix = np.zeros((n_instances, num_features))
        low_matrix = np.zeros((n_instances, num_features))
        high_matrix = np.zeros((n_instances, num_features))

        rule_values: List[Dict[int, Any]] = [{} for _ in range(n_instances)]
        instance_binned: List[Dict[str, Dict[int, Any]]] = [
            {
                "predict": {},
                "low": {},
                "high": {},
                "current_bin": {},
                "rule_values": {},
                "counts": {},
                "fractions": {},
            }
            for _ in range(n_instances)
        ]

        # Step 3: Build feature task list
        feature_tasks = build_feature_tasks(
            explainer,
            x_input,
            perturbed_feature,
            x_cal,
            features_to_ignore_array,
            config,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            predict,
            low,
            high,
            prediction.get("predict") if isinstance(prediction, dict) else None,
        )

        # Step 4: Process features sequentially
        for task in feature_tasks:
            result = feature_task(task)
            merge_feature_result(
                result,
                weights_predict,
                weights_low,
                weights_high,
                predict_matrix,
                low_matrix,
                high_matrix,
                rule_values,
                instance_binned,
                rule_boundaries,
            )

        # Finalize and return
        return finalize_explanation(
            explanation,
            weights_predict,
            weights_low,
            weights_high,
            predict_matrix,
            low_matrix,
            high_matrix,
            rule_values,
            instance_binned,
            rule_boundaries,
            prediction,
            instance_start_time,
            total_start_time,
            explainer,
        )


__all__ = ["SequentialExplainExecutor"]
