"""Sequential explain plugin - single-threaded execution strategy.

This plugin implements the traditional single-threaded explanation path,
processing features sequentially without parallelism. It serves as the
fallback strategy and reference implementation for behavioralequivalence tests.
"""

from __future__ import annotations

from collections import defaultdict
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Tuple

import numpy as np

from ...explanations import CalibratedExplanations
from ._base import BaseExplainPlugin
from ._helpers import explain_predict_step, initialize_explanation, merge_feature_result
from ._shared import ExplainConfig, ExplainRequest

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


class SequentialExplainPlugin(BaseExplainPlugin):
    """Sequential explain execution strategy.

    Processes all test instances and features in a single thread.
    This is the default fallback when parallelism is disabled or unavailable.
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
        # Import _feature_task from calibrated_explainer module
        from ..calibrated_explainer import _feature_task

        x_input = request.x
        features_to_ignore_array = request.features_to_ignore

        # Track total explanation time
        total_start_time = time()
        features_to_ignore_set = set(features_to_ignore_array.tolist())

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
        perturbed_feature = np.asarray(perturbed_feature, dtype=object)
        x_cal_np = np.asarray(x_cal)

        # Build feature index map for quick lookup
        if perturbed_feature.size:
            feature_ids = perturbed_feature[:, 0].astype(int)
            feature_index_lists: Dict[int, List[int]] = defaultdict(list)
            for idx, fid in enumerate(feature_ids):
                feature_index_lists[int(fid)].append(idx)
            feature_index_map = {
                fid: np.asarray(indices, dtype=int) for fid, indices in feature_index_lists.items()
            }
        else:
            feature_index_map = {}

        # Get calibration summaries for fast lookups
        categorical_value_counts, numeric_sorted_cache = explainer._get_calibration_summaries(
            x_cal_np
        )

        features_to_ignore_tuple = tuple(int(f) for f in features_to_ignore_set)
        categorical_features_tuple = tuple(int(f) for f in config.categorical_features)
        feature_values_all = config.feature_values
        baseline_predict = prediction["predict"]

        # Build feature tasks
        feature_tasks: List[Tuple[Any, ...]] = []

        for f in range(num_features):
            feature_indices = feature_index_map.get(f)
            lower_boundary = np.array(rule_boundaries[:, f, 0], copy=True)
            upper_boundary = np.array(rule_boundaries[:, f, 1], copy=True)

            # Extract feature-specific value mappings
            if isinstance(lesser_values, Mapping):
                lesser_feature = lesser_values.get(f, {})
            else:
                try:
                    lesser_feature = lesser_values[f]
                except (IndexError, KeyError, TypeError):
                    lesser_feature = {}

            if isinstance(greater_values, Mapping):
                greater_feature = greater_values.get(f, {})
            else:
                try:
                    greater_feature = greater_values[f]
                except (IndexError, KeyError, TypeError):
                    greater_feature = {}

            if isinstance(covered_values, Mapping):
                covered_feature = covered_values.get(f, {})
            else:
                try:
                    covered_feature = covered_values[f]
                except (IndexError, KeyError, TypeError):
                    covered_feature = {}

            value_counts_cache = categorical_value_counts.get(int(f), {})
            numeric_sorted_values = numeric_sorted_cache.get(f)

            # Extract feature columns safely
            if (
                isinstance(x_cal_np, np.ndarray)
                and x_cal_np.ndim >= 2
                and x_cal_np.shape[0]
                and f < x_cal_np.shape[1]
            ):
                x_cal_column = np.asarray(x_cal_np[:, f])
            else:
                x_cal_column = np.empty((0,))
            x_column = np.asarray(x_input[:, f])

            feature_tasks.append(
                (
                    f,
                    x_column,
                    predict,
                    low,
                    high,
                    baseline_predict,
                    features_to_ignore_tuple,
                    categorical_features_tuple,
                    feature_values_all,
                    feature_indices,
                    perturbed_feature,
                    lower_boundary,
                    upper_boundary,
                    lesser_feature,
                    greater_feature,
                    covered_feature,
                    value_counts_cache,
                    numeric_sorted_values,
                    x_cal_column,
                )
            )

        # Step 4: Process features sequentially
        for task in feature_tasks:
            result = _feature_task(task)
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

        # Step 5: Aggregate results into explanation format
        binned_predict: Dict[str, List[Any]] = {
            "predict": [],
            "low": [],
            "high": [],
            "current_bin": [],
            "rule_values": [],
            "counts": [],
            "fractions": [],
        }
        feature_weights: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
        feature_predict: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}

        for i in range(n_instances):
            binned_predict["predict"].append(instance_binned[i]["predict"])
            binned_predict["low"].append(instance_binned[i]["low"])
            binned_predict["high"].append(instance_binned[i]["high"])
            binned_predict["current_bin"].append(instance_binned[i]["current_bin"])
            binned_predict["rule_values"].append(rule_values[i])
            binned_predict["counts"].append(instance_binned[i]["counts"])
            binned_predict["fractions"].append(instance_binned[i]["fractions"])

            feature_weights["predict"].append(weights_predict[i].copy())
            feature_weights["low"].append(weights_low[i].copy())
            feature_weights["high"].append(weights_high[i].copy())

            feature_predict["predict"].append(predict_matrix[i].copy())
            feature_predict["low"].append(low_matrix[i].copy())
            feature_predict["high"].append(high_matrix[i].copy())

        # Step 6: Finalize explanation with timing metadata
        elapsed_time = time() - instance_start_time
        list_instance_time = [elapsed_time / n_instances for _ in range(n_instances)]
        total_time = time() - total_start_time

        explanation = explanation.finalize(
            binned_predict,
            feature_weights,
            feature_predict,
            prediction,
            instance_time=list_instance_time,
            total_time=total_time,
        )

        # Update explainer state
        explainer.latest_explanation = explanation
        explainer._last_explanation_mode = explainer._infer_explanation_mode()

        return explanation


__all__ = ["SequentialExplainPlugin"]
