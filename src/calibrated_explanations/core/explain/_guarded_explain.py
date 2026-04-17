# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements, line-too-long
"""Core computation for guarded explanations.

This module implements :func:`guarded_explain`, the single entry-point called
by :meth:`~calibrated_explanations.CalibratedExplainer.explain_guarded_factual`
and :meth:`~calibrated_explanations.CalibratedExplainer.explore_guarded_alternatives`.

Algorithm outline
-----------------
For each instance *x* and each feature *f*:

1. Use the multi-bin discretiser (``EntropyDiscretizer`` / ``RegressorDiscretizer``
   with ``max_depth=3``) to enumerate leaves for *f*.
2. For each non-empty numerical leaf, compute the median of calibration samples
   inside the leaf as the representative value.  Guard this representative
   point via a single KNN-based conformity probe at the raw ``significance``
   threshold.  Categorical leaves are guarded at the candidate value directly.
   Empty leaves are rejected unconditionally.
3. Collect perturbed instances from all leaves and features, run a single
   batched call to ``prediction_orchestrator.predict_internal``, then scatter
   results back into per-bin prediction records.
4. Optionally merge adjacent conforming bins into wider interval conditions.
   Merged intervals are re-tested via a single median probe; merges that
   fail the conformity check are skipped.
5. Wrap results into CE-compatible explanation containers and guarded
   subclasses. These containers reuse standard helper surfaces (plotting,
   narratives, conjunctions, reject integration) via compatibility shims, but
   guarded outputs are not contracted to be semantically identical to standard
   CE internals or perturbation math.
6. Audit counts and emitted rules refer to the shipped guard-rule decision for
   each interval candidate. A conforming candidate does not certify that every
   point inside the emitted interval would also pass the guard.

This is therefore a CE-compatible guarded extension rather than "standard CE
with fewer perturbations."
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ...utils.exceptions import ConfigurationError, NotFittedError, ValidationError

if TYPE_CHECKING:
    from ...utils.distribution_guard import InDistributionGuard
    from ..calibrated_explainer import CalibratedExplainer


_LOGGER = logging.getLogger(__name__)


def _require_guarded_calibration_alignment(explainer: "CalibratedExplainer") -> None:
    """Fail fast unless guarded checks and interval predictions share calibration features.

    Raises
    ------
    NotFittedError
        If the explainer has not been initialized (``initialized`` is falsy). This
        preserves the standard uninitialized-explainer error contract so callers
        catching ``NotFittedError`` are not silently broken.
    ConfigurationError
        If the explainer is in fast mode.  Fast interval calibrators are trained on
        per-feature blends of ``scaled_x_cal`` / ``fast_x_cal``, not on
        ``explainer.x_cal`` directly, so the alignment precondition below cannot
        be reliably enforced.  Guarded entrypoints are not supported for fast
        explainers (ADR-032 decision 7).
    ValidationError
        If the prediction orchestrator does not expose calibration-feature
        provenance, or if the recorded snapshot diverges from ``explainer.x_cal``.
    """
    # Preserve the standard error contract: an uninitialized explainer must raise
    # NotFittedError, not a calibration-alignment error.  Check this before
    # anything else so callers who catch NotFittedError are not broken.
    if not getattr(explainer, "initialized", False):
        raise NotFittedError("The learner must be initialized before calling guarded entrypoints.")

    # Fast explainers use per-feature blends of scaled_x_cal / fast_x_cal for
    # their interval calibrators, while InDistributionGuard always references
    # explainer.x_cal.  These two distributions cannot be aligned, so the ADR-032
    # exchangeability precondition cannot be reliably enforced for fast explainers.
    # Hard-fail immediately — this is not subject to configuration or opt-out.
    is_fast_fn = getattr(explainer, "is_fast", None)
    if callable(is_fast_fn) and is_fast_fn():
        raise ConfigurationError(
            "Guarded explanations are not supported for fast explainers. "
            "Use a standard (non-fast) explainer for guarded entrypoints.",
        )

    orchestrator = getattr(explainer, "prediction_orchestrator", None)
    if orchestrator is None or not hasattr(orchestrator, "get_interval_calibration_features"):
        raise ValidationError(
            "Guarded explanations require the prediction backend to expose the interval "
            "calibration features used for the active learner. Recalibrate the explainer "
            "before calling guarded entrypoints.",
        )

    backend_cal = orchestrator.get_interval_calibration_features()
    if backend_cal is None:
        raise ValidationError(
            "Guarded explanations require interval calibration features for the active "
            "prediction backend, but none were available. Recalibrate the explainer before "
            "calling guarded entrypoints.",
        )

    try:
        explainer_cal = np.asarray(explainer.x_cal)
        backend_cal = np.asarray(backend_cal)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValidationError(
            "Guarded explanations could not validate calibration-feature alignment for the "
            "prediction backend. Recalibrate the explainer before calling guarded "
            "entrypoints.",
        ) from exc

    if explainer_cal.shape != backend_cal.shape or not np.array_equal(explainer_cal, backend_cal):
        raise ValidationError(
            "Guarded explanations require the prediction backend to use the same calibration "
            "features as explainer.x_cal. Recalibrate the explainer so guarded filtering and "
            "interval predictions share the same calibration data.",
            details={
                "explainer_x_cal_shape": tuple(explainer_cal.shape),
                "backend_x_cal_shape": tuple(backend_cal.shape),
            },
        )


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
    guard: Optional["InDistributionGuard"] = None,
    adjusted_sig: Optional[float] = None,
):
    """Merge runs of adjacent conforming bins into single wider intervals.

    Two bins are *eligible* to be merged when they are numerically adjacent
    (``prev.upper == nxt.lower``) and both are conforming.

    In **factual** mode merging is anchored on the emitted factual rule. The
    factual bin is expanded iteratively by attempting to absorb the immediate
    left and right adjacent conforming bins until no further adjacent bin can be
    included. This permits partial factual expansion when the full conforming
    run would fail the merged guard re-check.

    In **alternative** mode the factual bin acts as a barrier: non-factual
    conforming bins to the left of the factual bin are merged among themselves,
    and non-factual bins to the right are merged among themselves, but the
    factual bin is never merged with non-factual bins.  This preserves the
    semantic distinction between "what the instance currently is" and "what
    could be changed".

    The merged bin's representative value is the median of the merged
    interval's calibration samples; a fresh prediction is obtained for it.
    The merged representative is re-tested via the guard.  If it fails the
    conformity check the merge is skipped and the original bins are kept.

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
    guard : InDistributionGuard or None
        Used to re-test merged representative conformity.
    adjusted_sig : float or None
        Effective per-feature significance level used for conformity checks.

    Returns
    -------
    merged_bins : list of GuardedBin
    """
    if not bins:
        return bins

    # Sort by lower bound (should already be sorted, but be safe)
    sorted_bins = sorted(bins, key=lambda b: b.lower)

    merge_kwargs = {
        "guard": guard,
        "adjusted_sig": adjusted_sig,
    }

    if mode == "alternative":
        # Split into three segments: left non-factual, factual, right non-factual
        left_bins = [
            b
            for b in sorted_bins
            if not b.is_factual
            and b.upper <= next((b2.lower for b2 in sorted_bins if b2.is_factual), np.inf)
        ]
        factual_bins = [b for b in sorted_bins if b.is_factual]
        right_bins = [
            b
            for b in sorted_bins
            if not b.is_factual
            and b.lower >= next((b2.upper for b2 in sorted_bins if b2.is_factual), -np.inf)
        ]

        merged_left = _merge_run(
            left_bins,
            current_value,
            explainer,
            x_instance,
            feature_idx,
            threshold,
            low_high_percentiles,
            mondrian_bins,
            **merge_kwargs,
        )
        merged_right = _merge_run(
            right_bins,
            current_value,
            explainer,
            x_instance,
            feature_idx,
            threshold,
            low_high_percentiles,
            mondrian_bins,
            **merge_kwargs,
        )
        return merged_left + factual_bins + merged_right
    else:
        return _merge_factual_anchor(
            sorted_bins,
            current_value,
            explainer,
            x_instance,
            feature_idx,
            threshold,
            low_high_percentiles,
            mondrian_bins,
            **merge_kwargs,
        )


def _guarded_bin_dedupe_key(gbin) -> tuple[Any, ...]:
    """Return a deterministic structural key for guarded-bin deduplication.

    The key intentionally excludes internal object identity and keeps only
    emitted interval bounds plus prediction payloads and guard-role flags.
    """
    return (
        gbin.lower,
        gbin.upper,
        gbin.predict,
        gbin.low,
        gbin.high,
        gbin.conforming,
        gbin.is_factual,
    )


def _dedupe_guarded_bins(bins):
    """Drop exact duplicate guarded bins while preserving first-seen order.

    This function is used in the pre-explanation guarded pipeline stage so
    explanation classes consume finalized bins and do not perform merge/dedupe
    corrections during rule creation.
    """
    deduped = []
    seen: set[tuple[Any, ...]] = set()
    for gbin in bins:
        key = _guarded_bin_dedupe_key(gbin)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gbin)
    return deduped


def _merge_run(
    bins,
    current_value: float,
    explainer: "CalibratedExplainer",
    x_instance: np.ndarray,
    feature_idx: int,
    threshold: Any,
    low_high_percentiles: tuple,
    mondrian_bins: Any,
    guard: Optional["InDistributionGuard"] = None,
    adjusted_sig: Optional[float] = None,
) -> list:
    """Merge a single ordered run of bins greedily."""
    if not bins:
        return []

    merged: list = []
    group: list = [bins[0]]

    for nxt in bins[1:]:
        prev = group[-1]
        adjacent = np.isclose(prev.upper, nxt.lower, rtol=0, atol=1e-9) or (prev.upper == nxt.lower)
        both_conforming = prev.conforming and nxt.conforming
        if adjacent and both_conforming:
            group.append(nxt)
        else:
            result = _finalise_group(
                group,
                current_value,
                explainer,
                x_instance,
                feature_idx,
                threshold,
                low_high_percentiles,
                mondrian_bins,
                guard=guard,
                adjusted_sig=adjusted_sig,
            )
            if isinstance(result, list):
                merged.extend(result)
            else:
                merged.append(result)
            group = [nxt]

    result = _finalise_group(
        group,
        current_value,
        explainer,
        x_instance,
        feature_idx,
        threshold,
        low_high_percentiles,
        mondrian_bins,
        guard=guard,
        adjusted_sig=adjusted_sig,
    )
    if isinstance(result, list):
        merged.extend(result)
    else:
        merged.append(result)
    return merged


def _merge_factual_anchor(
    bins,
    current_value: float,
    explainer: "CalibratedExplainer",
    x_instance: np.ndarray,
    feature_idx: int,
    threshold: Any,
    low_high_percentiles: tuple,
    mondrian_bins: Any,
    guard: Optional["InDistributionGuard"] = None,
    adjusted_sig: Optional[float] = None,
) -> list:
    """Expand the factual emitted rule across adjacent conforming bins."""
    if not bins:
        return []

    factual_idx = next((idx for idx, gbin in enumerate(bins) if gbin.is_factual), None)
    if factual_idx is None:
        return _merge_run(
            bins,
            current_value,
            explainer,
            x_instance,
            feature_idx,
            threshold,
            low_high_percentiles,
            mondrian_bins,
            guard=guard,
            adjusted_sig=adjusted_sig,
        )

    merged_group = [bins[factual_idx]]
    left_idx = factual_idx - 1
    right_idx = factual_idx + 1

    while True:
        expanded = False

        if left_idx >= 0 and bins[left_idx].conforming:
            candidate = [bins[left_idx], *merged_group]
            result = _finalise_group(
                candidate,
                current_value,
                explainer,
                x_instance,
                feature_idx,
                threshold,
                low_high_percentiles,
                mondrian_bins,
                guard=guard,
                adjusted_sig=adjusted_sig,
            )
            if not isinstance(result, list):
                merged_group = [result]
                left_idx -= 1
                expanded = True

        if right_idx < len(bins) and bins[right_idx].conforming:
            candidate = [*merged_group, bins[right_idx]]
            result = _finalise_group(
                candidate,
                current_value,
                explainer,
                x_instance,
                feature_idx,
                threshold,
                low_high_percentiles,
                mondrian_bins,
                guard=guard,
                adjusted_sig=adjusted_sig,
            )
            if not isinstance(result, list):
                merged_group = [result]
                right_idx += 1
                expanded = True

        if not expanded:
            break

    prefix = _merge_run(
        bins[: left_idx + 1],
        current_value,
        explainer,
        x_instance,
        feature_idx,
        threshold,
        low_high_percentiles,
        mondrian_bins,
        guard=guard,
        adjusted_sig=adjusted_sig,
    )
    suffix = _merge_run(
        bins[right_idx:],
        current_value,
        explainer,
        x_instance,
        feature_idx,
        threshold,
        low_high_percentiles,
        mondrian_bins,
        guard=guard,
        adjusted_sig=adjusted_sig,
    )
    return prefix + merged_group + suffix


def _finalise_group(
    group,
    current_value: float,
    explainer: "CalibratedExplainer",
    x_instance: np.ndarray,
    feature_idx: int,
    threshold: Any,
    low_high_percentiles: tuple,
    mondrian_bins: Any,
    guard: Optional["InDistributionGuard"] = None,
    adjusted_sig: Optional[float] = None,
):
    """Collapse a run of adjacent bins into a single merged GuardedBin.

    If a guard is provided the merged representative is re-tested.  When the
    re-test fails the merge is skipped and the original (unmerged) bin list is
    returned instead.
    """
    from ...explanations.guarded_explanation import GuardedBin  # local

    if len(group) == 1:
        return group[0]

    lo = group[0].lower
    hi = group[-1].upper
    is_factual = any(b.is_factual for b in group)

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
        # Empty merged range: reject merge, return original unmerged bins
        return group

    representative = float(np.median(in_range))

    # Re-test merged range via single median probe
    if guard is not None and adjusted_sig is not None:
        x_pert = x_instance.copy()
        x_pert[feature_idx] = representative
        p_val = float(guard.p_values(x_pert.reshape(1, -1))[0])
        conforming = p_val >= adjusted_sig
        if not conforming:
            # Merge failed — return original unmerged bins
            return group
    else:
        p_val = float(np.mean([b.p_value for b in group]))
        conforming = True

    x_pert = x_instance.copy()
    x_pert[feature_idx] = representative

    pred, low, high, _ = explainer.prediction_orchestrator.predict_internal(
        x_pert.reshape(1, -1),
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        bins=mondrian_bins,
    )

    m_predict = float(pred[0])
    m_low = float(low[0])
    m_high = float(high[0])
    if not np.isfinite(m_low):
        m_low = m_predict
    if not np.isfinite(m_high):
        m_high = m_predict
    if not np.isfinite(m_predict):
        m_predict = 0.5 * (m_low + m_high)

    # Enforce invariants for merged bin
    epsilon = 1e-9
    if not (m_low - epsilon <= m_high + epsilon):
        if "regression" not in getattr(explainer, "mode", ""):
            raise ValidationError(
                "Prediction interval invariant violated (merged bin): low > high",
                details={"feat": feature_idx, "lo": m_low, "hi": m_high},
            )
        warnings.warn(
            "Prediction interval invariant violated (merged bin): low > high; reordering bounds.",
            UserWarning,
            stacklevel=2,
        )
        m_low, m_high = (m_high, m_low) if m_low > m_high else (m_low, m_high)
    if not (m_low - epsilon <= m_predict <= m_high + epsilon):
        m_predict = min(max(m_predict, m_low), m_high)

    return GuardedBin(
        lower=lo,
        upper=hi,
        representative=representative,
        predict=m_predict,
        low=m_low,
        high=m_high,
        conforming=conforming,
        p_value=p_val,
        is_factual=is_factual,
        is_merged=True,
    )


def merge_adjacent_bins_for_testing(*args: Any, **kwargs: Any) -> list:
    """Return merged guarded bins without importing a private helper in tests."""
    return _merge_adjacent_bins(*args, **kwargs)


def finalise_guarded_group_for_testing(*args: Any, **kwargs: Any) -> list:
    """Return finalized guarded bins without importing a private helper in tests."""
    return _finalise_group(*args, **kwargs)


def guarded_explain(
    explainer: "CalibratedExplainer",
    x: np.ndarray,
    *,
    mode: str = "factual",
    threshold: Any = None,
    low_high_percentiles: tuple = (5, 95),
    mondrian_bins: Any = None,
    features_to_ignore: Any = None,
    per_instance_features_to_ignore: Any = None,
    significance: float = 0.1,
    merge_adjacent: bool = False,
    n_neighbors: int = 5,
    normalize_guard: bool = True,
    verbose: bool = False,
) -> Any:
    """Generate representative-point guarded explanations for a batch of instances.

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
        Conformity significance level. A larger value yields a stricter
        test (fewer bins accepted as in-distribution), because the
        conforming predicate compares the shipped decision p-value against
        ``significance``.
    merge_adjacent : bool, default=False
        Merge adjacent conforming bins into wider interval conditions.
        Merged representatives are re-tested via the guard; merges that fail
        conformity are skipped. This is a heuristic compaction step and can
        emit interval rules standard factual CE would not produce.
    n_neighbors : int, default=5
        KNN parameter for :class:`~...utils.distribution_guard.InDistributionGuard`.
    normalize_guard : bool, default=True
        Apply per-feature normalisation inside the guard.
    verbose : bool, default=False
        When True, emit UserWarnings for degraded/diagnostic situations.

    Returns
    -------
    CalibratedExplanations
        A CE-compatible container whose ``.explanations`` list holds
        :class:`GuardedFactualExplanation` or
        :class:`GuardedAlternativeExplanation` objects. Guarded conformity is
        assessed on representative perturbed points for interval candidates,
        not on every point within an emitted interval.
    """
    from ...explanations.explanations import (  # local to avoid cycles
        AlternativeExplanations,
        CalibratedExplanations,
    )
    from ...explanations.guarded_explanation import (
        GuardedAlternativeExplanation,
        GuardedBin,
        GuardedFactualExplanation,
    )
    from ...utils.distribution_guard import InDistributionGuard

    x = np.atleast_2d(np.asarray(x))
    n_instances, n_features = x.shape
    if not (0 < float(significance) <= 1):
        raise ValidationError(
            "significance must be in the interval (0, 1]",
            details={"significance": significance},
        )
    if int(n_neighbors) < 1:
        raise ValidationError(
            "n_neighbors must be >= 1",
            details={"n_neighbors": n_neighbors},
        )
    _require_guarded_calibration_alignment(explainer)

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
        normalize=normalize_guard,
        categorical_features=sorted(getattr(explainer, "categorical_features", ()) or ()),
    )

    categorical_features = set(getattr(explainer, "categorical_features", ()))
    ignore_set = set(features_to_ignore) if features_to_ignore is not None else set()
    per_instance_ignore = per_instance_features_to_ignore

    total_bins_tested = 0
    total_bins_conforming = 0
    total_bins_nonconforming = 0
    total_factual_bins_nonconforming = 0

    # ------------------------------------------------------------------ #
    # Threshold Validation (Major Issue 1d)
    # ------------------------------------------------------------------ #
    if threshold is not None and not np.isscalar(threshold) and not isinstance(threshold, tuple):
        t_arr = np.asarray(threshold)
        if t_arr.ndim > 0 and len(t_arr) != n_instances:
            raise ValidationError(
                f"Threshold array length ({len(t_arr)}) must match n_instances ({n_instances})"
            )

    # ------------------------------------------------------------------ #
    # Baseline predictions for all instances
    # ------------------------------------------------------------------ #
    base_predict, base_low, base_high, predicted_class = (
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
    # Metadata: (inst_idx, feat_idx, bin_idx, lo, hi, p_val, rep, adjusted_sig, is_factual)
    pert_metadata: list[tuple] = []

    # Cache per-feature bin lists (shared across instances for numerical features).
    feature_disc_bins: dict[int, list] = {}
    for f in range(n_features):
        if f in ignore_set:
            continue
        if f not in categorical_features:
            if discretizer is not None:
                feature_disc_bins[f] = discretizer.get_bins_with_cal_indices(f, explainer.x_cal)
            else:
                feature_disc_bins[f] = []

    for inst_idx in range(n_instances):
        x_instance = x[inst_idx]

        inst_ignore = set(ignore_set)
        if per_instance_ignore is not None and inst_idx < len(per_instance_ignore):
            inst_ignore.update(per_instance_ignore[inst_idx])

        for f in range(n_features):
            if f in inst_ignore:
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

                    total_bins_tested += 1
                    if conforming:
                        total_bins_conforming += 1
                    else:
                        total_bins_nonconforming += 1
                        if is_factual:
                            total_factual_bins_nonconforming += 1

                    perturbed_rows.append(x_pert)
                    pert_metadata.append(
                        (
                            inst_idx,
                            f,
                            "cat",
                            -np.inf,
                            np.inf,
                            p_val,
                            val,  # Removed float(val) cast to prevent string category crashes
                            significance,
                            is_factual,
                        )
                    )
            else:
                disc_bins = feature_disc_bins.get(f, [])
                for b_idx, (lo, hi, cal_indices) in enumerate(disc_bins):
                    feat_vals = (
                        explainer.x_cal[cal_indices, f] if len(cal_indices) > 0 else np.array([])
                    )

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

                    if feat_vals.size == 0:
                        # Empty bin: reject unconditionally
                        if verbose:
                            warnings.warn(
                                f"Feature {f}, bin ({lo}, {hi}] has no calibration "
                                "samples; rejecting bin unconditionally",
                                UserWarning,
                                stacklevel=2,
                            )
                        if lo == -np.inf and hi == np.inf:
                            representative = float(x_instance[f])
                        elif lo == -np.inf:
                            representative = float(hi)
                        elif hi == np.inf:
                            representative = float(lo)
                        else:
                            representative = float((lo + hi) / 2)
                        p_val = 0.0
                        x_pert = x_instance.copy()
                        x_pert[f] = representative

                    else:
                        # Non-empty bin: single median probe
                        representative = float(np.median(feat_vals))
                        x_pert = x_instance.copy()
                        x_pert[f] = representative
                        p_val = float(guard.p_values(x_pert.reshape(1, -1))[0])

                    conforming = p_val >= significance

                    total_bins_tested += 1
                    if conforming:
                        total_bins_conforming += 1
                    else:
                        total_bins_nonconforming += 1
                        if is_factual:
                            total_factual_bins_nonconforming += 1

                    perturbed_rows.append(x_pert)
                    pert_metadata.append(
                        (
                            inst_idx,
                            f,
                            b_idx,
                            lo,
                            hi,
                            p_val,
                            representative,
                            significance,
                            is_factual,
                        )
                    )

    # ------------------------------------------------------------------ #
    # Phase 2: Batched calibrated predictions for all perturbed instances
    # ------------------------------------------------------------------ #
    if perturbed_rows:
        x_perturbed = np.vstack([r.reshape(1, -1) for r in perturbed_rows])

        # Replicate per-instance thresholds if provided (probabilistic regression).
        # For scalar/tuple thresholds, pass through unchanged.
        pert_threshold: Any = threshold
        if (
            threshold is not None
            and not np.isscalar(threshold)
            and not isinstance(threshold, tuple)
            and hasattr(threshold, "__len__")
        ):
            inst_indices = [m[0] for m in pert_metadata]
            if len(threshold) != n_instances:
                raise ValidationError(
                    f"Threshold array length ({len(threshold)}) must match n_instances ({n_instances})",
                    details={"n_instances": n_instances, "len_threshold": len(threshold)},
                )
            pert_threshold = np.asarray(threshold)[inst_indices]

        # Replicate mondrian_bins if provided (same bin as the source instance)
        pert_mbins: Any = None
        if mondrian_bins is not None:
            if hasattr(mondrian_bins, "__len__"):
                # Select the bin corresponding to each perturbed row's instance index
                inst_indices = [m[0] for m in pert_metadata]
                if len(mondrian_bins) != n_instances:
                    raise ValidationError(
                        f"Mondrian bins array length ({len(mondrian_bins)}) must match n_instances ({n_instances})",
                        details={"n_instances": n_instances, "len_mondrian": len(mondrian_bins)},
                    )
                pert_mbins = np.asarray(mondrian_bins)[inst_indices]
            else:
                pert_mbins = mondrian_bins

        pert_predict, pert_low, pert_high, _ = explainer.prediction_orchestrator.predict_internal(
            x_perturbed,
            threshold=pert_threshold,
            low_high_percentiles=low_high_percentiles,
            bins=pert_mbins,
        )
    else:
        pert_predict = np.array([])
        pert_low = np.array([])
        pert_high = np.array([])

    if len(pert_predict) != len(pert_metadata):
        raise ValidationError(
            "Guarded perturbed prediction batch length mismatch",
            details={
                "n_predicted": int(len(pert_predict)),
                "n_expected": int(len(pert_metadata)),
            },
        )

    # ------------------------------------------------------------------ #
    # Phase 3: Scatter predictions back into per-instance structures
    # ------------------------------------------------------------------ #
    # Group pert_metadata by instance_idx
    # Sort metadata by inst_idx to maintain scatter alignment (Issue 1e)
    # This ensures that we can index into pert_predict/low/high correctly
    # even if multiple instances contributed to the batch.

    # Actually, pert_metadata was populated in a nested loop (inst then feat then bin).
    # row_idx in pert_metadata ALWAYS matches the row index in x_perturbed and thus pert_predict.
    # The error "IndexError: index 1 is out of bounds for axis 0 with size 1"
    # suggests that len(pert_metadata) > len(pert_predict).
    # This happens if some rows were appended to perturbed_rows but not predicted.

    inst_feat_bins: dict[int, dict[int, list[GuardedBin]]] = {i: {} for i in range(n_instances)}

    for row_idx, meta in enumerate(pert_metadata):
        inst_idx, feat_idx, b_idx, lo, hi, p_val, representative, adj_sig, is_factual = meta
        conforming = p_val >= adj_sig

        # Validate interval invariant: low <= predict <= high
        # Robust indexing: pert_predict may be empty if no rows were perturbed
        if row_idx >= len(pert_predict):
            raise ValidationError(
                "Missing prediction for perturbed row in guarded_explain batch",
                details={
                    "row_idx": row_idx,
                    "n_predicted": int(len(pert_predict)),
                    "n_metadata": int(len(pert_metadata)),
                },
            )

        g_predict = float(pert_predict[row_idx])
        g_low = float(pert_low[row_idx])
        g_high = float(pert_high[row_idx])
        if not np.isfinite(g_low):
            g_low = g_predict
        if not np.isfinite(g_high):
            g_high = g_predict
        if not np.isfinite(g_predict):
            g_predict = 0.5 * (g_low + g_high)

        # Apply epsilon coercion for flotation precision drift (Issue 1a)
        # This keeps the invariant g_low <= g_predict <= g_high true
        # for nearly-on-boundary values.
        epsilon = 1e-8
        if g_predict < g_low and (g_low - g_predict) < epsilon:
            g_predict = g_low
        elif g_predict > g_high and (g_predict - g_high) < epsilon:
            g_predict = g_high

        # Allow small floating point tolerance matching ADR-021 / CalibratedExplanation check
        epsilon = 1e-9
        if not (g_low - epsilon <= g_high + epsilon):
            if "regression" not in getattr(explainer, "mode", ""):
                raise ValidationError(
                    "Prediction interval invariant violated: low > high",
                    details={
                        "inst_idx": inst_idx,
                        "feat_idx": feat_idx,
                        "bin": b_idx,
                        "low": g_low,
                        "high": g_high,
                    },
                )
            warnings.warn(
                "Prediction interval invariant violated: low > high; reordering bounds.",
                UserWarning,
                stacklevel=2,
            )
            g_low, g_high = (g_high, g_low) if g_low > g_high else (g_low, g_high)

        if not (g_low - epsilon <= g_predict <= g_high + epsilon):
            g_predict = min(max(g_predict, g_low), g_high)

        gbin = GuardedBin(
            lower=lo,
            upper=hi,
            representative=representative,
            predict=g_predict,
            low=g_low,
            high=g_high,
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
    # Phase 4: Assemble explanation objects inside CalibratedExplanations
    # ------------------------------------------------------------------ #
    feature_names = list(
        getattr(explainer, "feature_names", None) or [str(i) for i in range(n_features)]
    )

    # Create the CalibratedExplanations container
    container = CalibratedExplanations(
        explainer,
        x,
        threshold,
        mondrian_bins,
        features_to_ignore=list(ignore_set) if ignore_set else None,
    )
    container.low_high_percentiles = low_high_percentiles

    # Track instances where all candidates were removed (emit summary warning later)
    _all_removed_instances: list[int] = []

    # Minimal binned payload for CE helper compatibility shims.
    # For example, add_new_rule_condition expects binned['rule_values'][feature][0][0].
    binned_predict: dict[str, list[Any]] = {"rule_values": []}

    # Build prediction dict (shared across all instances, indexed by [inst_idx])
    is_mc = (
        explainer.is_multiclass() if callable(explainer.is_multiclass) else explainer.is_multiclass
    )
    is_regression = (
        "regression" in explainer.mode
        if isinstance(explainer.mode, str)
        else getattr(explainer, "mode", "classification") == "regression"
    )

    prediction = {
        "predict": base_predict,
        "low": base_low,
        "high": base_high,
        "classes": (predicted_class if is_mc else np.ones(base_predict.shape)),
    }

    # Add probabilities for classification if missing
    if not is_regression and "prob" not in prediction:
        # Standard CE convention: Classification MUST include 'prob' equivalent to 'predict'
        # which is the probability of the predicted class (or matrix for some plugins).
        prediction["prob"] = base_predict

    # Populate feature weights/predictions so CE helper surfaces can operate on
    # guarded outputs without claiming metric identity with standard CE.
    feature_predict: dict[str, list[np.ndarray]] = {"predict": [], "low": [], "high": []}
    feature_weights: dict[str, list[np.ndarray]] = {"predict": [], "low": [], "high": []}

    # Select the explanation subclass based on mode
    expl_class = GuardedFactualExplanation if mode == "factual" else GuardedAlternativeExplanation

    for inst_idx in range(n_instances):
        x_instance = x[inst_idx]
        feat_dict = inst_feat_bins[inst_idx]

        f_predict = {
            "predict": np.zeros(n_features),
            "low": np.zeros(n_features),
            "high": np.zeros(n_features),
        }
        f_weights = {
            "predict": np.zeros(n_features),
            "low": np.zeros(n_features),
            "high": np.zeros(n_features),
        }

        threshold_i: Any = threshold
        if (
            threshold is not None
            and not np.isscalar(threshold)
            and not isinstance(threshold, tuple)
            and hasattr(threshold, "__len__")
            and inst_idx < len(threshold)
        ):
            threshold_i = np.asarray(threshold)[inst_idx]

        inst_ignore = set(ignore_set)
        if per_instance_ignore is not None and inst_idx < len(per_instance_ignore):
            inst_ignore.update(per_instance_ignore[inst_idx])

        # Build per-instance guarded_bins dict with optional merging
        guarded_bins: dict[int, list] = {}
        for f in range(n_features):
            if f in inst_ignore:
                continue
            if f not in feat_dict:
                continue

            bins_for_feature = feat_dict[f]
            is_cat = f in categorical_features

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
                        threshold=threshold_i,
                        low_high_percentiles=low_high_percentiles,
                        mondrian_bins=mondrian_bins,
                        mode=mode,
                        guard=guard,
                        adjusted_sig=significance,
                    )

            bins_for_feature = _dedupe_guarded_bins(bins_for_feature)

            guarded_bins[f] = bins_for_feature

            # Factual Weight/Predict Population for this instance/feature
            perturb_bins = [b for b in bins_for_feature if not b.is_factual and b.conforming]
            if not perturb_bins:
                perturb_bins = [b for b in bins_for_feature if b.conforming]

            if perturb_bins:
                p_preds = [b.predict for b in perturb_bins]
                p_lows = [b.low for b in perturb_bins]
                p_highs = [b.high for b in perturb_bins]

                avg_p = float(np.mean(p_preds))
                avg_l = float(np.mean(p_lows))
                avg_h = float(np.mean(p_highs))

                f_predict["predict"][f] = avg_p
                f_predict["low"][f] = avg_l
                f_predict["high"][f] = avg_h

                f_weights["predict"][f] = float(base_predict[inst_idx] - avg_p)
                f_weights["low"][f] = float(base_predict[inst_idx] - avg_h)
                f_weights["high"][f] = float(base_predict[inst_idx] - avg_l)

        if guarded_bins and not any(b.conforming for bins in guarded_bins.values() for b in bins):
            _all_removed_instances.append(inst_idx)

        for key in ["predict", "low", "high"]:
            feature_predict[key].append(f_predict[key])
            feature_weights[key].append(f_weights[key])

        # Build a per-instance rule_values mapping keyed by feature index.
        # Each entry matches the legacy tuple shape: (sampled_values, x, x)
        rule_values_entry: dict[int, Any] = {}
        for f, bins_for_feature in guarded_bins.items():
            sampled_values = [b.representative for b in bins_for_feature]
            rule_values_entry[int(f)] = (sampled_values, x_instance[f], x_instance[f])

        binned_predict["rule_values"].append(rule_values_entry)

        expl = expl_class(
            container,
            index=inst_idx,
            x=x_instance,
            binned=binned_predict,
            feature_weights=feature_weights,
            feature_predict=feature_predict,
            prediction=prediction,
            y_threshold=threshold,
            instance_bin=mondrian_bins[inst_idx] if mondrian_bins is not None else None,
            condition_source=getattr(explainer, "condition_source", "prediction"),
            guarded_bins=guarded_bins,
            feature_names=feature_names,
            categorical_features=categorical_features,
            verbose=verbose,
        )
        container.explanations.append(expl)

    container.feature_weights = feature_weights
    container.feature_predict = feature_predict

    if _all_removed_instances:
        n_removed = len(_all_removed_instances)
        if n_removed <= 5:
            idx_desc = ", ".join(str(i) for i in _all_removed_instances)
        else:
            first_three = ", ".join(str(i) for i in _all_removed_instances[:3])
            idx_desc = f"{first_three}, ... ({n_removed} total)"
        warnings.warn(
            f"All interval candidates were removed by the guard for {n_removed} "
            f"of {n_instances} instances (indices: {idx_desc}). "
            "No rules will be emitted for these instances.",
            UserWarning,
            stacklevel=2,
        )

    _LOGGER.info(
        "guarded_explain completed",
        extra={
            "mode": mode,
            "n_instances": int(n_instances),
            "n_features": int(n_features),
            "n_perturbed": int(len(perturbed_rows)),
            "bins_tested": int(total_bins_tested),
            "bins_conforming": int(total_bins_conforming),
            "bins_nonconforming": int(total_bins_nonconforming),
            "factual_bins_nonconforming": int(total_factual_bins_nonconforming),
        },
    )

    if mode == "alternative":
        return AlternativeExplanations.from_collection(container)

    return container
