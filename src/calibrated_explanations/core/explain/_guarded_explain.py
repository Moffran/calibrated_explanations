# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements, line-too-long
"""Core computation for guarded (in-distribution) explanations.

This module implements :func:`guarded_explain`, the single entry-point called
by :meth:`~calibrated_explanations.CalibratedExplainer.explain_guarded_factual`
and :meth:`~calibrated_explanations.CalibratedExplainer.explore_guarded_alternatives`.

Algorithm outline
-----------------
For each instance *x* and each feature *f*:

1. Use the multi-bin discretiser (``EntropyDiscretizer`` / ``RegressorDiscretizer``
   with ``max_depth=3``) to enumerate leaves for *f*.
2. For each leaf, draw a representative value from the calibration samples
   inside the leaf (median by default), build a perturbed instance
   ``x_pert = x`` with ``x_pert[f] = representative``, and test whether
   ``x_pert`` is in-distribution via :class:`~...utils.distribution_guard.InDistributionGuard`.
3. Collect perturbed instances from all leaves and features, run a single
   batched call to ``prediction_orchestrator.predict_internal``, then scatter
   results back into per-bin prediction records.
4. Optionally merge adjacent conforming bins into wider interval conditions.
5. Wrap results into :class:`~...explanations.guarded_explanation.GuardedExplanations`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


def _merge_adjacent_bins(
    bins,  # list[GuardedBin]
    current_value: float,
    explainer: "CalibratedExplainer",
    x_instance: np.ndarray,
    feature_idx: int,
    threshold: Any,
    low_high_percentiles: tuple,
    mondrian_bins: Any,
    mode: str = "factual",
):
    """Merge runs of adjacent conforming bins into single wider intervals.

    Two bins are *eligible* to be merged when they are numerically adjacent
    (``prev.upper == nxt.lower``) and both are conforming.

    In **factual** mode all adjacent conforming bins are merged freely,
    potentially yielding a single wide interval that spans the factual bin
    together with neighbouring in-distribution regions.

    In **alternative** mode the factual bin acts as a barrier: non-factual
    conforming bins to the left of the factual bin are merged among themselves,
    and non-factual bins to the right are merged among themselves, but the
    factual bin is never merged with non-factual bins.  This preserves the
    semantic distinction between "what the instance currently is" and "what
    could be changed".

    The merged bin's representative value is the median of the merged
    interval's calibration samples; a fresh prediction is obtained for it.

    Parameters
    ----------
    bins : list of GuardedBin
        Ordered (ascending ``lower``) list of bins for the feature.
    current_value : float
        Original instance feature value (determines the factual bin).
    explainer : CalibratedExplainer
        Used to make predictions for merged representative values.
    x_instance : np.ndarray of shape (n_features,)
    feature_idx : int
    threshold, low_high_percentiles, mondrian_bins
        Forwarded unchanged to ``predict_internal``.
    mode : {'factual', 'alternative'}, default='factual'
        Controls how the factual bin interacts with merging.

    Returns
    -------
    merged_bins : list of GuardedBin
    """
    if not bins:
        return bins

    # Sort by lower bound (should already be sorted, but be safe)
    sorted_bins = sorted(bins, key=lambda b: b.lower)

    if mode == "alternative":
        # Split into three segments: left non-factual, factual, right non-factual
        left_bins = [b for b in sorted_bins if not b.is_factual and b.upper <= next(
            (b2.lower for b2 in sorted_bins if b2.is_factual), np.inf
        )]
        factual_bins = [b for b in sorted_bins if b.is_factual]
        right_bins = [b for b in sorted_bins if not b.is_factual and b.lower >= next(
            (b2.upper for b2 in sorted_bins if b2.is_factual), -np.inf
        )]

        merged_left = _merge_run(left_bins, current_value, explainer, x_instance, feature_idx, threshold, low_high_percentiles, mondrian_bins)
        merged_right = _merge_run(right_bins, current_value, explainer, x_instance, feature_idx, threshold, low_high_percentiles, mondrian_bins)
        return merged_left + factual_bins + merged_right
    else:
        # Factual mode: merge all adjacent conforming bins freely
        return _merge_run(sorted_bins, current_value, explainer, x_instance, feature_idx, threshold, low_high_percentiles, mondrian_bins)


def _merge_run(
    bins,
    current_value: float,
    explainer: "CalibratedExplainer",
    x_instance: np.ndarray,
    feature_idx: int,
    threshold: Any,
    low_high_percentiles: tuple,
    mondrian_bins: Any,
) -> list:
    """Merge a single ordered run of bins greedily."""
    if not bins:
        return []

    merged: list = []
    group: list = [bins[0]]

    for nxt in bins[1:]:
        prev = group[-1]
        adjacent = np.isclose(prev.upper, nxt.lower, rtol=0, atol=1e-9) or (
            prev.upper == nxt.lower
        )
        both_conforming = prev.conforming and nxt.conforming
        if adjacent and both_conforming:
            group.append(nxt)
        else:
            merged.append(_finalise_group(group, current_value, explainer, x_instance, feature_idx, threshold, low_high_percentiles, mondrian_bins))
            group = [nxt]

    merged.append(_finalise_group(group, current_value, explainer, x_instance, feature_idx, threshold, low_high_percentiles, mondrian_bins))
    return merged


def _finalise_group(
    group,
    current_value: float,
    explainer: "CalibratedExplainer",
    x_instance: np.ndarray,
    feature_idx: int,
    threshold: Any,
    low_high_percentiles: tuple,
    mondrian_bins: Any,
):
    """Collapse a run of adjacent bins into a single merged GuardedBin."""
    from ...explanations.guarded_explanation import GuardedBin  # local

    if len(group) == 1:
        return group[0]

    lo = group[0].lower
    hi = group[-1].upper
    is_factual = any(b.is_factual for b in group)
    conforming = any(b.conforming for b in group)
    mean_p = float(np.mean([b.p_value for b in group]))

    # Representative: median of calibration values in the merged range
    feat_vals = explainer.x_cal[:, feature_idx]
    if lo == -np.inf and hi == np.inf:
        in_range = feat_vals
    elif lo == -np.inf:
        in_range = feat_vals[feat_vals <= hi]
    elif hi == np.inf:
        in_range = feat_vals[feat_vals > lo]
    else:
        in_range = feat_vals[(feat_vals > lo) & (feat_vals <= hi)]

    if in_range.size == 0:
        representative = (lo if lo != -np.inf else 0.0) if hi == np.inf else (
            hi if lo == -np.inf else (lo + hi) / 2
        )
    else:
        representative = float(np.median(in_range))

    x_pert = x_instance.copy()
    x_pert[feature_idx] = representative
    pred, low, high, _ = explainer.prediction_orchestrator.predict_internal(
        x_pert.reshape(1, -1),
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        bins=mondrian_bins,
    )
    return GuardedBin(
        lower=lo,
        upper=hi,
        representative=representative,
        predict=float(pred[0]),
        low=float(low[0]),
        high=float(high[0]),
        conforming=conforming,
        p_value=mean_p,
        is_factual=is_factual,
        is_merged=True,
    )


def guarded_explain(
    explainer: "CalibratedExplainer",
    x: np.ndarray,
    *,
    mode: str = "factual",
    threshold: Any = None,
    low_high_percentiles: tuple = (5, 95),
    mondrian_bins: Any = None,
    features_to_ignore: Any = None,
    significance: float = 0.1,
    merge_adjacent: bool = False,
    n_neighbors: int = 5,
    leaf_strategy: str = "median",
    normalize_guard: bool = True,
) -> Any:
    """Generate guarded (in-distribution) explanations for a batch of instances.

    Parameters
    ----------
    explainer : CalibratedExplainer
        Fitted and calibrated explainer instance.
    x : np.ndarray of shape (n_instances, n_features)
        Test instances to explain.
    mode : {'factual', 'alternative'}, default='factual'
        * ``'factual'`` – show the rule for the instance's own bin plus
          adjacent conforming bins when *merge_adjacent* is True.
        * ``'alternative'`` – show rules for all conforming non-factual bins.
    threshold : any, optional
        Forwarded to ``prediction_orchestrator.predict_internal``.
    low_high_percentiles : tuple of float, default=(5, 95)
        Forwarded to ``prediction_orchestrator.predict_internal``.
    mondrian_bins : array-like or None
        Mondrian categories; forwarded to the prediction orchestrator.
    features_to_ignore : sequence of int or None
        Feature indices to skip.
    significance : float, default=0.1
        Conformity significance level.  Bins with ``p_value < significance``
        are treated as out-of-distribution.
    merge_adjacent : bool, default=False
        Merge adjacent conforming bins into wider interval conditions.
    n_neighbors : int, default=5
        KNN parameter for :class:`~...utils.distribution_guard.InDistributionGuard`.
    leaf_strategy : {'median', 'percentiles'}, default='median'
        Representative value sampling strategy for the guard.
    normalize_guard : bool, default=True
        Apply per-feature normalisation inside the guard.

    Returns
    -------
    GuardedExplanations
    """
    from ...explanations.guarded_explanation import (  # local to avoid cycles
        GuardedBin,
        GuardedExplanation,
        GuardedExplanations,
        GuardedFeatureExplanation,
    )
    from ...utils.distribution_guard import InDistributionGuard

    x = np.atleast_2d(np.asarray(x))
    n_instances, n_features = x.shape

    # ------------------------------------------------------------------ #
    # Set up multi-bin discretiser (same depth as explore_alternatives)
    # ------------------------------------------------------------------ #
    disc_type = "regressor" if "regression" in explainer.mode else "entropy"
    explainer.set_discretizer(disc_type, features_to_ignore=features_to_ignore)
    discretizer = explainer.discretizer

    # ------------------------------------------------------------------ #
    # Build in-distribution guard from calibration data
    # ------------------------------------------------------------------ #
    guard = InDistributionGuard(
        explainer.x_cal,
        n_neighbors=n_neighbors,
        leaf_strategy=leaf_strategy,
        normalize=normalize_guard,
    )

    categorical_features = set(getattr(explainer, "categorical_features", ()))
    ignore_set = set(features_to_ignore) if features_to_ignore is not None else set()

    # ------------------------------------------------------------------ #
    # Baseline predictions for all instances
    # ------------------------------------------------------------------ #
    base_predict, base_low, base_high, _ = (
        explainer.prediction_orchestrator.predict_internal(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            bins=mondrian_bins,
        )
    )

    # ------------------------------------------------------------------ #
    # Phase 1: Determine representative values & conformity per feature/bin
    # ------------------------------------------------------------------ #
    # Accumulate all perturbed instances for a single batched prediction call.
    perturbed_rows: list[np.ndarray] = []  # each row is a 1D feature vector
    pert_metadata: list[tuple] = []        # (inst_idx, feat_idx, bin_idx, lo, hi, frac, rep, p_val_rep)

    # Cache per-feature bin lists (shared across instances for numerical features)
    feature_disc_bins: dict[int, list] = {}
    for f in range(n_features):
        if f in ignore_set or f in categorical_features:
            continue
        if discretizer is not None:
            feature_disc_bins[f] = discretizer.get_bins_with_cal_indices(f, explainer.x_cal)
        else:
            feature_disc_bins[f] = []

    for inst_idx in range(n_instances):
        x_instance = x[inst_idx]

        for f in range(n_features):
            if f in ignore_set:
                continue

            if f in categorical_features:
                feature_values = getattr(explainer, "feature_values", {}).get(f, [])
                for val in feature_values:
                    x_pert = x_instance.copy()
                    x_pert[f] = val
                    # Guard p-value for this categorical value
                    p_val = float(guard.p_values(x_pert.reshape(1, -1))[0])
                    conforming = p_val >= significance
                    is_factual = bool(val == x_instance[f])
                    perturbed_rows.append(x_pert)
                    pert_metadata.append(
                        (inst_idx, f, "cat", -np.inf, np.inf, float(conforming), float(val), p_val, is_factual)
                    )
            else:
                disc_bins = feature_disc_bins.get(f, [])
                for b_idx, (lo, hi, _cal_indices) in enumerate(disc_bins):
                    frac, candidates = guard.leaf_conforming_fraction(
                        feature_idx=f,
                        lower=lo,
                        upper=hi,
                        x_instance=x_instance,
                        significance=significance,
                    )
                    # Representative: median of candidates (or heuristic if leaf empty)
                    if candidates.size > 0:
                        representative = float(np.median(candidates))
                    else:
                        # Leaf has no calibration samples – use midpoint heuristic
                        if lo == -np.inf and hi == np.inf:
                            representative = float(x_instance[f])
                        elif lo == -np.inf:
                            representative = float(hi)
                        elif hi == np.inf:
                            representative = float(lo)
                        else:
                            representative = float((lo + hi) / 2)

                    # Conformity: leaf passes if majority of sampled candidates are in-dist
                    conforming = frac >= 0.5

                    # Determine if this is the factual bin
                    cur = float(x_instance[f])
                    if lo == -np.inf and hi == np.inf:
                        is_factual = True
                    elif lo == -np.inf:
                        is_factual = cur <= hi
                    elif hi == np.inf:
                        is_factual = cur > lo
                    else:
                        is_factual = (cur > lo) and (cur <= hi)

                    # p-value for the representative (used for per-bin reporting)
                    x_pert = x_instance.copy()
                    x_pert[f] = representative
                    p_val = float(guard.p_values(x_pert.reshape(1, -1))[0])

                    perturbed_rows.append(x_pert)
                    pert_metadata.append(
                        (inst_idx, f, b_idx, lo, hi, float(frac), representative, p_val, is_factual)
                    )

    # ------------------------------------------------------------------ #
    # Phase 2: Batched calibrated predictions for all perturbed instances
    # ------------------------------------------------------------------ #
    if perturbed_rows:
        x_perturbed = np.vstack([r.reshape(1, -1) for r in perturbed_rows])

        # Replicate mondrian_bins if provided (same bin as the source instance)
        pert_mbins: Any = None
        if mondrian_bins is not None:
            import numpy as _np  # noqa: PLC0415 – local use only
            if hasattr(mondrian_bins, "__len__"):
                # Select the bin corresponding to each perturbed row's instance index
                inst_indices = [m[0] for m in pert_metadata]
                try:
                    pert_mbins = _np.asarray(mondrian_bins)[inst_indices]
                except (IndexError, TypeError):
                    pert_mbins = None
            else:
                pert_mbins = mondrian_bins

        pert_predict, pert_low, pert_high, _ = (
            explainer.prediction_orchestrator.predict_internal(
                x_perturbed,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=pert_mbins,
            )
        )
    else:
        pert_predict = np.array([])
        pert_low = np.array([])
        pert_high = np.array([])

    # ------------------------------------------------------------------ #
    # Phase 3: Scatter predictions back into per-instance structures
    # ------------------------------------------------------------------ #
    # Group pert_metadata by instance_idx
    inst_feat_bins: dict[int, dict[int, list[GuardedBin]]] = {
        i: {} for i in range(n_instances)
    }

    for row_idx, meta in enumerate(pert_metadata):
        inst_idx, feat_idx, b_idx, lo, hi, frac, representative, p_val, is_factual = meta
        conforming = frac >= 0.5 if b_idx != "cat" else bool(frac)  # frac == float(conforming) for cat

        gbin = GuardedBin(
            lower=lo,
            upper=hi,
            representative=representative,
            predict=float(pert_predict[row_idx]),
            low=float(pert_low[row_idx]),
            high=float(pert_high[row_idx]),
            conforming=conforming,
            p_value=p_val,
            is_factual=bool(is_factual),
            is_merged=False,
        )
        feat_dict = inst_feat_bins[inst_idx]
        if feat_idx not in feat_dict:
            feat_dict[feat_idx] = []
        feat_dict[feat_idx].append(gbin)

    # ------------------------------------------------------------------ #
    # Phase 4: Assemble GuardedExplanation objects
    # ------------------------------------------------------------------ #
    feature_names = list(getattr(explainer, "feature_names", None) or [str(i) for i in range(n_features)])

    all_explanations: list[GuardedExplanation] = []

    for inst_idx in range(n_instances):
        x_instance = x[inst_idx]
        feat_dict = inst_feat_bins[inst_idx]
        feature_explanations: list[GuardedFeatureExplanation] = []

        for f in range(n_features):
            if f in ignore_set:
                continue
            if f not in feat_dict:
                continue

            bins_for_feature = feat_dict[f]
            is_cat = f in categorical_features
            fname = feature_names[f] if f < len(feature_names) else str(f)

            # Sort numerical bins by lower bound for display / merging
            if not is_cat:
                bins_for_feature = sorted(bins_for_feature, key=lambda b: b.lower)
                if merge_adjacent and len(bins_for_feature) > 1:
                    bins_for_feature = _merge_adjacent_bins(
                        bins_for_feature,
                        current_value=float(x_instance[f]),
                        explainer=explainer,
                        x_instance=x_instance,
                        feature_idx=f,
                        threshold=threshold,
                        low_high_percentiles=low_high_percentiles,
                        mondrian_bins=mondrian_bins,
                        mode=mode,
                    )

            feature_explanations.append(
                GuardedFeatureExplanation(
                    feature_idx=f,
                    feature_name=fname,
                    current_value=x_instance[f],
                    bins=bins_for_feature,
                    is_categorical=is_cat,
                )
            )

        all_explanations.append(
            GuardedExplanation(
                instance_idx=inst_idx,
                instance=x_instance.copy(),
                feature_explanations=feature_explanations,
                baseline_predict=float(base_predict[inst_idx]),
                baseline_low=float(base_low[inst_idx]),
                baseline_high=float(base_high[inst_idx]),
                mode=mode,
            )
        )

    return GuardedExplanations(all_explanations, mode)
