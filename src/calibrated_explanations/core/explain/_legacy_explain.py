"""Legacy explanation routines preserved for benchmarking and regression tests.

This module captures the behaviour of :meth:`CalibratedExplainer.explain` and
its helper :meth:`CalibratedExplainer._explain_predict_step` before the
performance optimisations introduced in this branch.  The functions provided
here are intentionally verbatim ports of the historical implementations so that
benchmarks and tests can compare the old and new paths without relying on git
history or patch juggling at runtime.

The functions expect a fully initialised :class:`CalibratedExplainer` instance
and delegate to the public helper utilities (``prediction_helpers``) in the
same way as the original methods.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from ...utils import safe_mean
from ._computation import explain_predict_step

if TYPE_CHECKING:
    pass


def explain(
    explainer,
    x,
    threshold=None,
    low_high_percentiles: Sequence[float] = (5, 95),
    bins=None,
    *,
    features_to_ignore: Optional[Sequence[int]] = None,
):
    """Legacy implementation of :meth:`CalibratedExplainer.explain`.

    The function mirrors the historical control flow to enable correctness
    comparisons with the optimised implementation.
    """
    from ._computation import initialize_explanation  # pylint: disable=import-outside-toplevel
    from ._helpers import validate_and_prepare_input  # pylint: disable=import-outside-toplevel
    from .feature_task import assign_weight  # pylint: disable=import-outside-toplevel

    total_time = time()

    features_to_ignore = (
        explainer.features_to_ignore
        if features_to_ignore is None
        else np.union1d(explainer.features_to_ignore, features_to_ignore)
    )

    x = validate_and_prepare_input(explainer, x)
    explanation = initialize_explanation(
        explainer, x, low_high_percentiles, threshold, bins, features_to_ignore
    )

    instance_time = time()

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
        x,
        threshold,
        low_high_percentiles,
        bins,
        features_to_ignore,
    )

    feature_weights: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
    feature_predict: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
    binned_predict: Dict[str, List[Any]] = {
        "predict": [],
        "low": [],
        "high": [],
        "current_bin": [],
        "rule_values": [],
        "counts": [],
        "fractions": [],
    }

    rule_values: Dict[int, Dict[int, Any]] = {}
    instance_weights: Dict[int, Dict[str, np.ndarray]] = {}
    instance_predict: Dict[int, Dict[str, np.ndarray]] = {}
    instance_binned: Dict[int, Dict[str, Dict[int, Any]]] = {}

    for i, instance in enumerate(x):
        rule_values[i] = {}
        instance_weights[i] = {
            "predict": np.zeros(instance.shape[0]),
            "low": np.zeros(instance.shape[0]),
            "high": np.zeros(instance.shape[0]),
        }
        instance_predict[i] = {
            "predict": np.zeros(instance.shape[0]),
            "low": np.zeros(instance.shape[0]),
            "high": np.zeros(instance.shape[0]),
        }
        instance_binned[i] = {
            "predict": {},
            "low": {},
            "high": {},
            "current_bin": {},
            "rule_values": {},
            "counts": {},
            "fractions": {},
        }

    for f in range(explainer.num_features):
        if f in features_to_ignore:
            for i in range(len(x)):
                rule_values[i][f] = (explainer.feature_values[f], x[i, f], x[i, f])
                instance_binned[i]["predict"][f] = predict[i]
                instance_binned[i]["low"][f] = low[i]
                instance_binned[i]["high"][f] = high[i]
            continue

        feature_values = explainer.feature_values[f]
        perturbed_mask = perturbed_feature[:, 0] == f
        perturbed_instances = perturbed_feature[perturbed_mask, 1]
        perturbed = np.unique(perturbed_instances)

        if f in explainer.categorical_features:
            for i in np.unique(perturbed):
                current_bin = -1
                average_predict = np.zeros(len(feature_values))
                low_predict = np.zeros(len(feature_values))
                high_predict = np.zeros(len(feature_values))
                counts = np.zeros(len(feature_values))

                for bin_value, value in enumerate(feature_values):
                    mask = (
                        (perturbed_feature[:, 0] == f)
                        & (perturbed_feature[:, 1] == i)
                        & (perturbed_feature[:, 2] == value)
                    )
                    indices = np.where(mask)[0]

                    if x[i, f] == value:
                        current_bin = bin_value

                    average_predict[bin_value] = predict[indices][0] if indices.size else 0
                    low_predict[bin_value] = low[indices][0] if indices.size else 0
                    high_predict[bin_value] = high[indices][0] if indices.size else 0
                    counts[bin_value] = int(np.sum(x_cal[:, f] == value))

                rule_values[i][f] = (feature_values, x[i, f], x[i, f])
                uncovered = np.setdiff1d(np.arange(len(average_predict)), current_bin)
                total_counts = np.sum(counts[uncovered])
                if total_counts == 0:
                    fractions = np.zeros_like(counts[uncovered])
                else:
                    fractions = counts[uncovered] / total_counts

                instance_binned[i]["predict"][f] = average_predict
                instance_binned[i]["low"][f] = low_predict
                instance_binned[i]["high"][f] = high_predict
                instance_binned[i]["current_bin"][f] = current_bin
                instance_binned[i]["counts"][f] = counts
                instance_binned[i]["fractions"][f] = fractions

                if len(uncovered) == 0:
                    instance_predict[i]["predict"][f] = 0
                    instance_predict[i]["low"][f] = 0
                    instance_predict[i]["high"][f] = 0

                    instance_weights[i]["predict"][f] = 0
                    instance_weights[i]["low"][f] = 0
                    instance_weights[i]["high"][f] = 0
                else:
                    instance_predict[i]["predict"][f] = safe_mean(average_predict[uncovered])
                    instance_predict[i]["low"][f] = safe_mean(low_predict[uncovered])
                    instance_predict[i]["high"][f] = safe_mean(high_predict[uncovered])

                    instance_weights[i]["predict"][f] = assign_weight(
                        instance_predict[i]["predict"][f], prediction["predict"][i]
                    )
                    tmp_low = assign_weight(instance_predict[i]["low"][f], prediction["predict"][i])
                    tmp_high = assign_weight(
                        instance_predict[i]["high"][f], prediction["predict"][i]
                    )
                    instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                    instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])
        else:
            feature_values = np.unique(np.array(x_cal[:, f]))
            lower_boundary = rule_boundaries[:, f, 0]
            upper_boundary = rule_boundaries[:, f, 1]

            avg_predict_map: Dict[int, np.ndarray] = {}
            low_predict_map: Dict[int, np.ndarray] = {}
            high_predict_map: Dict[int, np.ndarray] = {}
            counts_map: Dict[int, np.ndarray] = {}
            rule_value_map: Dict[int, List[np.ndarray]] = {}
            for i in range(len(x)):
                lower_boundary[i] = (
                    lower_boundary[i] if np.any(feature_values < lower_boundary[i]) else -np.inf
                )
                upper_boundary[i] = (
                    upper_boundary[i] if np.any(feature_values > upper_boundary[i]) else np.inf
                )
                num_bins = 1 + (1 if lower_boundary[i] != -np.inf else 0)
                num_bins += 1 if upper_boundary[i] != np.inf else 0
                avg_predict_map[i] = np.zeros(num_bins)
                low_predict_map[i] = np.zeros(num_bins)
                high_predict_map[i] = np.zeros(num_bins)
                counts_map[i] = np.zeros(num_bins)
                rule_value_map[i] = []

            bin_value = np.zeros(len(x), dtype=int)
            current_bin = -np.ones(len(x), dtype=int)

            for j, val in enumerate(np.unique(lower_boundary)):
                if lesser_values[f][j][0].shape[0] == 0:
                    continue
                feature_mask = perturbed_feature[:, 0] == f
                bin_mask = perturbed_feature[:, 2] == j
                lesser_mask = np.asarray(perturbed_feature[:, 3], dtype=object)
                lesser_mask = lesser_mask == True  # noqa: E712 - intentional identity check
                for i in np.where(lower_boundary == val)[0]:
                    instance_mask = perturbed_feature[:, 1] == i
                    mask = feature_mask & bin_mask & instance_mask & lesser_mask
                    index = np.where(mask)[0]

                    avg_predict_map[i][bin_value[i]] = (
                        safe_mean(predict[index]) if index.size else 0
                    )
                    low_predict_map[i][bin_value[i]] = safe_mean(low[index]) if index.size else 0
                    high_predict_map[i][bin_value[i]] = safe_mean(high[index]) if index.size else 0
                    counts_map[i][bin_value[i]] = int(np.sum(x_cal[:, f] < val))
                    rule_value_map[i].append(lesser_values[f][j][0])
                    bin_value[i] += 1

            for j, val in enumerate(np.unique(upper_boundary)):
                if greater_values[f][j][0].shape[0] == 0:
                    continue
                feature_mask = perturbed_feature[:, 0] == f
                bin_mask = perturbed_feature[:, 2] == j
                greater_mask = np.asarray(perturbed_feature[:, 3], dtype=object) == False  # noqa: E712
                for i in np.where(upper_boundary == val)[0]:
                    instance_mask = perturbed_feature[:, 1] == i
                    mask = feature_mask & bin_mask & instance_mask & greater_mask
                    index = np.where(mask)[0]

                    avg_predict_map[i][bin_value[i]] = (
                        safe_mean(predict[index]) if index.size else 0
                    )
                    low_predict_map[i][bin_value[i]] = safe_mean(low[index]) if index.size else 0
                    high_predict_map[i][bin_value[i]] = safe_mean(high[index]) if index.size else 0
                    counts_map[i][bin_value[i]] = int(np.sum(x_cal[:, f] > val))
                    rule_value_map[i].append(greater_values[f][j][0])
                    bin_value[i] += 1

            unique_boundaries = np.unique(
                np.stack([lower_boundary, upper_boundary], axis=1), axis=0
            )
            between_mask = np.equal(perturbed_feature[:, 3], None)
            for i in range(len(x)):
                instance_mask = perturbed_feature[:, 1] == i
                for j, (_lower, _upper) in enumerate(unique_boundaries):
                    feature_mask = perturbed_feature[:, 0] == f
                    bin_mask = perturbed_feature[:, 2] == j
                    mask = feature_mask & instance_mask & bin_mask & between_mask
                    index = np.where(mask)[0]

                    avg_predict_map[i][bin_value[i]] = (
                        safe_mean(predict[index]) if index.size else 0
                    )
                    low_predict_map[i][bin_value[i]] = safe_mean(low[index]) if index.size else 0
                    high_predict_map[i][bin_value[i]] = safe_mean(high[index]) if index.size else 0
                    counts_map[i][bin_value[i]] = int(
                        np.sum((x_cal[:, f] >= _lower) & (x_cal[:, f] <= _upper))
                    )
                    rule_value_map[i].append(covered_values[f][j][0])
                    current_bin[i] = bin_value[i]

            for i in range(len(x)):
                rule_values[i][f] = (rule_value_map[i], x[i, f], x[i, f])

                uncovered = np.setdiff1d(np.arange(len(avg_predict_map[i])), current_bin[i])

                total_counts = np.sum(counts_map[i][uncovered])
                if total_counts == 0:
                    fractions = np.zeros_like(counts_map[i][uncovered])
                else:
                    fractions = counts_map[i][uncovered] / total_counts

                instance_binned[i]["predict"][f] = avg_predict_map[i]
                instance_binned[i]["low"][f] = low_predict_map[i]
                instance_binned[i]["high"][f] = high_predict_map[i]
                instance_binned[i]["current_bin"][f] = current_bin[i]
                instance_binned[i]["counts"][f] = counts_map[i]
                instance_binned[i]["fractions"][f] = fractions

                if len(uncovered) == 0:
                    instance_predict[i]["predict"][f] = 0
                    instance_predict[i]["low"][f] = 0
                    instance_predict[i]["high"][f] = 0

                    instance_weights[i]["predict"][f] = 0
                    instance_weights[i]["low"][f] = 0
                    instance_weights[i]["high"][f] = 0
                else:
                    instance_predict[i]["predict"][f] = safe_mean(avg_predict_map[i][uncovered])
                    instance_predict[i]["low"][f] = safe_mean(low_predict_map[i][uncovered])
                    instance_predict[i]["high"][f] = safe_mean(high_predict_map[i][uncovered])

                    instance_weights[i]["predict"][f] = assign_weight(
                        instance_predict[i]["predict"][f], prediction["predict"][i]
                    )
                    tmp_low = assign_weight(instance_predict[i]["low"][f], prediction["predict"][i])
                    tmp_high = assign_weight(
                        instance_predict[i]["high"][f], prediction["predict"][i]
                    )
                    instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                    instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])

    for i in range(len(x)):
        binned_predict["predict"].append(instance_binned[i]["predict"])
        binned_predict["low"].append(instance_binned[i]["low"])
        binned_predict["high"].append(instance_binned[i]["high"])
        binned_predict["current_bin"].append(instance_binned[i]["current_bin"])
        binned_predict["rule_values"].append(rule_values[i])
        binned_predict["counts"].append(instance_binned[i]["counts"])
        binned_predict["fractions"].append(instance_binned[i]["fractions"])

        feature_weights["predict"].append(instance_weights[i]["predict"])
        feature_weights["low"].append(instance_weights[i]["low"])
        feature_weights["high"].append(instance_weights[i]["high"])
        feature_predict["predict"].append(instance_predict[i]["predict"])
        feature_predict["low"].append(instance_predict[i]["low"])
        feature_predict["high"].append(instance_predict[i]["high"])

    # parity instrumentation removed

    elapsed_time = time() - instance_time
    list_instance_time = [elapsed_time / len(x) for _ in range(len(x))]

    explanation = explanation.finalize(
        binned_predict,
        feature_weights,
        feature_predict,
        prediction,
        instance_time=list_instance_time,
        total_time=total_time,
    )
    explainer.latest_explanation = explanation
    explainer._last_explanation_mode = explainer._infer_explanation_mode()
    # parity instrumentation removed
    return explanation


__all__ = ["explain", "explain_predict_step"]
