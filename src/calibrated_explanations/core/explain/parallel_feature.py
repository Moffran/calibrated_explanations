"""Feature-parallel explain executor - parallel execution across features.

This plugin implements feature-level parallelism, distributing feature
perturbation tasks across worker processes or threads. It shares most logic
with the sequential plugin but substitutes parallel dispatch for the feature loop.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from ...explanations import CalibratedExplanations
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
    from ..calibrated_explainer import CalibratedExplainer


class FeatureParallelExplainExecutor(BaseExplainExecutor):
    """Feature-parallel explain execution strategy.

    Distributes feature perturbation tasks across an executor's workers,
    enabling parallel computation of feature effects. Falls back to sequential
    processing if the executor is unavailable or disabled.
    """

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "feature-parallel"

    @property
    def priority(self) -> int:
        """Return plugin priority (medium, checked after instance-parallel)."""
        return 20

    def supports(self, request: ExplainRequest, config: ExplainConfig) -> bool:
        """Return True if feature-level parallelism is enabled and appropriate.

        Requirements:
        - Executor must be available and enabled
        - Granularity must be 'feature'
        - Instance parallelism must not be active (no nested parallelism)
        """
        if config.executor is None:
            return False
        if not config.executor.config.enabled:
            return False
        if config.granularity != "feature":
            return False
        # Reject if instance parallel flag is set to avoid nested parallelism
        return request.skip_instance_parallel or config.granularity != "instance"

    def execute(
        self,
        request: ExplainRequest,
        config: ExplainConfig,
        explainer: CalibratedExplainer,
    ) -> CalibratedExplanations:
        """Execute feature-parallel explain operation.

        This implementation mirrors the original sequential path but substitutes
        _explain_parallel_features for the sequential feature loop.
        """
        # Import _feature_task from feature_task module
        from .feature_task import _feature_task  # pylint: disable=import-outside-toplevel

        x_input = request.x
        features_to_ignore_array = request.features_to_ignore
        executor = config.executor

        # Track total explanation time
        total_start_time = time()
        # features_to_ignore handled by build_feature_tasks

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
        ) = explain_predict_step(
            explainer,
            x_input,
            request.threshold,
            request.low_high_percentiles,
            request.bins,
            features_to_ignore_array,
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

        # Step 4: Process features in parallel using executor
        if feature_tasks:
            work_items = max(len(feature_tasks), 1) * max(n_instances, 1)
            results = executor.map(_feature_task, feature_tasks, work_items=work_items)

            # Merge results in sorted order to ensure determinism
            for result in sorted(results, key=lambda item: item[0]):
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


__all__ = ["FeatureParallelExplainExecutor"]
