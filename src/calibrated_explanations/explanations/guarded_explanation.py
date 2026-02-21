# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
"""Guarded explanation classes for in-distribution calibrated explanations.

These classes represent explanations produced by
:meth:`~calibrated_explanations.CalibratedExplainer.explain_guarded_factual` and
:meth:`~calibrated_explanations.CalibratedExplainer.explore_guarded_alternatives`.

Unlike standard explanations, guarded explanations:

* Use multi-bin (``max_depth=3``) discretisers for *both* factual and
  alternative modes, yielding **interval rule conditions** such as
  ``"0.50 < age <= 1.50"`` rather than simple threshold splits.
* Prune leaves whose representative perturbations are classified as
  out-of-distribution by the in-distribution guard.
* Optionally merge adjacent conforming leaves into wider intervals
  (``merge_adjacent=True``).

The main entry-points are :class:`GuardedFactualExplanation` (subclass of
:class:`~.explanation.FactualExplanation`) and
:class:`GuardedAlternativeExplanation` (subclass of
:class:`~.explanation.AlternativeExplanation`).  Both integrate with the
existing ``CalibratedExplanations`` container and inherit ``plot()``,
``list_rules()``, and ``build_rules_payload()`` from their parents.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ._conjunctions import ConjunctionState
from .explanation import AlternativeExplanation, FactualExplanation


@dataclass
class GuardedBin:
    """A single leaf (bin) from the multi-bin discretiser.

    Attributes
    ----------
    lower : float
        Lower bound of the bin's interval (``-np.inf`` for the first bin).
    upper : float
        Upper bound of the bin's interval (``+np.inf`` for the last bin).
    representative : float
        The representative feature value used for prediction (typically the
        median of calibration samples in the leaf).
    predict : float
        Calibrated prediction for the perturbed instance constructed by
        substituting *representative* for the feature while keeping all other
        feature values at their original levels.
    low : float
        Lower bound of the calibrated prediction interval.
    high : float
        Upper bound of the calibrated prediction interval.
    conforming : bool
        ``True`` when the conformal p-value for the representative value
        exceeds the configured significance threshold.
    p_value : float
        Conformal p-value for the representative value.
    is_factual : bool
        ``True`` when the original instance's feature value falls in this bin.
    is_merged : bool
        ``True`` when this bin was created by merging two or more adjacent
        conforming bins.
    """

    lower: float
    upper: float
    representative: float
    predict: float
    low: float
    high: float
    conforming: bool
    p_value: float
    is_factual: bool
    is_merged: bool = False


def _guarded_condition_str(
    feature_name: str,
    gbin: GuardedBin,
    is_categorical: bool = False,
) -> str:
    """Build a human-readable interval condition string for a GuardedBin.

    Examples
    --------
    * ``"age <= 30.00"``  (first bin, ``lower == -inf``)
    * ``"30.00 < age <= 50.00"``  (interior bin)
    * ``"age > 50.00"``  (last bin, ``upper == inf``)
    * ``"colour = 'red'"``  (categorical feature)
    """
    if is_categorical:
        return f"{feature_name} = {gbin.representative}"
    if gbin.lower == -np.inf and gbin.upper == np.inf:
        return f"{feature_name} (any value)"
    if gbin.lower == -np.inf:
        return f"{feature_name} <= {gbin.upper:.4g}"
    if gbin.upper == np.inf:
        return f"{feature_name} > {gbin.lower:.4g}"
    return f"{gbin.lower:.4g} < {feature_name} <= {gbin.upper:.4g}"


def _to_python(value: Any) -> Any:
    """Convert numpy scalars to native Python objects for stable JSON-like payloads."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_name(feature_idx: int, feature_names: List[str]) -> str:
    """Return display name for a feature index."""
    return feature_names[feature_idx] if feature_idx < len(feature_names) else str(feature_idx)


def _sorted_guarded_bins(guarded_bins: Dict[int, List[GuardedBin]]) -> list[tuple[int, GuardedBin]]:
    """Return deterministic flattened guarded bins sorted by feature and bounds."""
    flattened: list[tuple[int, GuardedBin]] = []
    for feat in sorted(guarded_bins):
        bins = sorted(
            guarded_bins[feat],
            key=lambda b: (
                float(b.lower) if b.lower != -np.inf else float("-inf"),
                float(b.upper) if b.upper != np.inf else float("inf"),
                float(b.representative) if isinstance(b.representative, (int, float, np.number)) else 0.0,
            ),
        )
        for gbin in bins:
            flattened.append((feat, gbin))
    return flattened


class GuardedFactualExplanation(FactualExplanation):
    """Guarded factual explanation using interval bins with conformity filtering.

    Inherits from :class:`~.explanation.FactualExplanation` so that
    ``plot()``, ``list_rules()``, ``build_rules_payload()``, and
    ``_rules_with_impact()`` work without modification.

    Parameters
    ----------
    calibrated_explanations : CalibratedExplanations
        Parent container (must have a valid ``get_explainer()``).
    guarded_bins : dict[int, list[GuardedBin]]
        Per-feature list of guarded bins (keyed by feature index).
    feature_names : list[str]
        Feature names for condition string generation.
    categorical_features : set[int]
        Set of categorical feature indices.

    All other parameters are forwarded to :class:`FactualExplanation`.
    """

    def __init__(
        self,
        calibrated_explanations,
        index,
        x,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=None,
        instance_bin=None,
        condition_source: str = "prediction",
        *,
        guarded_bins: Optional[Dict[int, List[GuardedBin]]] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[set] = None,
        verbose: bool = False,
    ):
        # Store guarded data BEFORE super().__init__ which calls get_rules()
        self._guarded_bins = guarded_bins or {}
        self._guarded_feature_names = feature_names or []
        self._guarded_categorical = categorical_features or set()
        self._guarded_verbose = bool(verbose)
        super().__init__(
            calibrated_explanations,
            index,
            x,
            binned,
            feature_weights,
            feature_predict,
            prediction,
            y_threshold,
            instance_bin,
            condition_source,
        )

    def _check_preconditions(self):
        """Skip discretizer type check (guarded uses multi-bin by design)."""

    def define_conditions(self):
        """Generate interval condition strings from guarded bins."""
        self.conditions = []
        num_features = len(self.x_test)
        for f in range(num_features):  # pylint: disable=invalid-name
            if f not in self._guarded_bins:
                self.conditions.append("")
                continue
            bins = self._guarded_bins[f]
            factual_bin = next((b for b in bins if b.is_factual), None)
            if factual_bin is not None:
                name = (
                    self._guarded_feature_names[f]
                    if f < len(self._guarded_feature_names)
                    else str(f)
                )
                self.conditions.append(
                    _guarded_condition_str(name, factual_bin, f in self._guarded_categorical)
                )
            else:
                self.conditions.append("")
        return self.conditions

    def get_rules(self):
        """Create factual rules with CE-compatible payload semantics."""
        if (
            getattr(self, "has_conjunctive_rules", False)
            and getattr(self, "conjunctive_rules", None) is not None
        ):
            return self.conjunctive_rules

        instance = np.array(self.x_test, copy=True)
        state_helper = ConjunctionState(None)
        state_helper.state["classes"] = self.prediction.get("classes", [])

        state_helper.set_base_prediction(
            self.prediction["predict"], self.prediction["low"], self.prediction["high"]
        )
        rules = self.define_conditions()
        ignored = self.ignored_features_for_instance()
        explainer = self.get_explainer()
        categorical_features = set(getattr(explainer, "categorical_features", ()) or ())
        categorical_labels = getattr(explainer, "categorical_labels", None)

        for f, _ in enumerate(instance):  # pylint: disable=invalid-name
            if f in ignored:
                continue
            bins = self._guarded_bins.get(f, [])
            factual_bin = next((b for b in bins if b.is_factual), None)
            if factual_bin is None or not factual_bin.conforming:
                name = (
                    self._guarded_feature_names[f]
                    if f < len(self._guarded_feature_names)
                    else str(f)
                )
                if self._guarded_verbose:
                    warnings.warn(
                        f"Dropping non-conforming factual bin for feature '{name}' "
                        f"(p_value={factual_bin.p_value:.4f}).",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            if self.prediction["predict"] == self.feature_predict["predict"][f]:
                continue

            if not rules[f]:
                continue

            if f in categorical_features:
                if categorical_labels is not None:
                    value_str = categorical_labels[f][int(instance[f])]
                else:
                    value_str = str(instance[f])
            else:
                value_str = str(np.around(instance[f], decimals=2))

            instance_predict = self.feature_predict["predict"][f]
            instance_low = self.feature_predict["low"][f]
            instance_high = self.feature_predict["high"][f]
            prediction = self.prediction["predict"]

            w = prediction - instance_predict
            w_low = prediction - instance_high if instance_high != np.inf else -np.inf
            w_high = prediction - instance_low if instance_low != -np.inf else np.inf
            sampled_values = [factual_bin.representative]
            with np.errstate(all="ignore"):
                try:
                    sampled_values = self.binned["rule_values"][f][0][-1]
                except (KeyError, IndexError, TypeError):
                    sampled_values = [factual_bin.representative]

            state_helper.add_rule(
                predict=self.prediction["predict"],
                low=self.prediction["low"],
                high=self.prediction["high"],
                base_predict=instance_predict,
                value=value_str,
                feature=f,
                sampled_values=sampled_values,
                feature_value=self.x_test[f],
                rule_text=rules[f],
                is_conjunctive=False,
                weight=w,
                weight_low=w_low,
                weight_high=w_high,
            )

        self.rules = state_helper.get_state()
        self.has_rules = True
        return self.rules

    def get_guarded_audit(self) -> Dict[str, Any]:
        """Return interval-level guarded audit data for this factual explanation."""
        ignored = self.ignored_features_for_instance()
        feature_names = list(self._guarded_feature_names or [])
        intervals: list[dict[str, Any]] = []

        for feat, gbin in _sorted_guarded_bins(self._guarded_bins):
            name = _feature_name(feat, feature_names)
            condition = _guarded_condition_str(name, gbin, feat in self._guarded_categorical)

            if feat in ignored:
                emitted = False
                reason = "ignored_feature"
            elif not gbin.conforming:
                emitted = False
                reason = "removed_guard"
            elif not gbin.is_factual:
                emitted = False
                reason = "design_excluded"
            elif self.prediction["predict"] == self.feature_predict["predict"][feat]:
                emitted = False
                reason = "zero_impact"
            elif not condition:
                emitted = False
                reason = "design_excluded"
            else:
                emitted = True
                reason = "emitted"

            intervals.append(
                {
                    "feature": int(feat),
                    "feature_name": str(name),
                    "lower": _to_python(gbin.lower),
                    "upper": _to_python(gbin.upper),
                    "representative": _to_python(gbin.representative),
                    "p_value": _to_python(gbin.p_value),
                    "conforming": bool(gbin.conforming),
                    "is_factual": bool(gbin.is_factual),
                    "is_merged": bool(gbin.is_merged),
                    "emitted": bool(emitted),
                    "emission_reason": reason,
                    "condition": condition,
                    "predict": _to_python(gbin.predict),
                    "low": _to_python(gbin.low),
                    "high": _to_python(gbin.high),
                }
            )

        removed_guard = sum(1 for rec in intervals if not rec["conforming"])
        summary = {
            "intervals_tested": int(len(intervals)),
            "intervals_conforming": int(sum(1 for rec in intervals if rec["conforming"])),
            "intervals_removed_guard": int(removed_guard),
            "intervals_emitted": int(sum(1 for rec in intervals if rec["emitted"])),
            "features_with_any_removed_guard": int(
                len({rec["feature"] for rec in intervals if rec["emission_reason"] == "removed_guard"})
            ),
        }

        return {
            "mode": "factual",
            "instance_index": int(self.index),
            "summary": summary,
            "intervals": intervals,
        }


class GuardedAlternativeExplanation(AlternativeExplanation):
    """Guarded alternative explanation using interval bins with conformity filtering.

    Inherits from :class:`~.explanation.AlternativeExplanation` so that
    ``plot()``, ``list_rules()``, ``build_rules_payload()``, and filtering
    methods (``super_explanations``, ``counter_explanations``, etc.) work.

    Parameters
    ----------
    calibrated_explanations : CalibratedExplanations
        Parent container (must have a valid ``get_explainer()``).
    guarded_bins : dict[int, list[GuardedBin]]
        Per-feature list of guarded bins (keyed by feature index).
    feature_names : list[str]
        Feature names for condition string generation.
    categorical_features : set[int]
        Set of categorical feature indices.

    All other parameters are forwarded to :class:`AlternativeExplanation`.
    """

    def __init__(
        self,
        calibrated_explanations,
        index,
        x,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        y_threshold=None,
        instance_bin=None,
        condition_source: str = "prediction",
        *,
        guarded_bins: Optional[Dict[int, List[GuardedBin]]] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[set] = None,
        verbose: bool = False,
    ):
        # Store guarded data BEFORE super().__init__ which calls get_rules()
        self._guarded_bins = guarded_bins or {}
        self._guarded_feature_names = feature_names or []
        self._guarded_categorical = categorical_features or set()
        self._guarded_verbose = bool(verbose)
        super().__init__(
            calibrated_explanations,
            index,
            x,
            binned,
            feature_weights,
            feature_predict,
            prediction,
            y_threshold,
            instance_bin,
            condition_source,
        )

    def _check_preconditions(self):
        """Skip discretizer type check (guarded uses multi-bin by design)."""

    def define_conditions(self):
        """Generate interval condition strings from guarded bins."""
        self.conditions = []
        num_features = len(self.x_test)
        for f in range(num_features):  # pylint: disable=invalid-name
            if f not in self._guarded_bins:
                self.conditions.append("")
                continue
            # For alternatives, show all conforming non-factual bin conditions
            alt_bins = [b for b in self._guarded_bins[f] if not b.is_factual and b.conforming]
            if alt_bins:
                name = (
                    self._guarded_feature_names[f]
                    if f < len(self._guarded_feature_names)
                    else str(f)
                )
                conditions = [
                    _guarded_condition_str(name, b, f in self._guarded_categorical)
                    for b in alt_bins
                ]
                self.conditions.append(" | ".join(conditions))
            else:
                self.conditions.append("")
        return self.conditions

    def get_rules(self):
        """Build rules from conforming non-factual guarded bins.

        Returns
        -------
        dict
            Rule state dict compatible with ``ConjunctionState.get_state()``.
        """
        if (
            getattr(self, "has_conjunctive_rules", False)
            and getattr(self, "conjunctive_rules", None) is not None
        ):
            return self.conjunctive_rules

        state_helper = ConjunctionState(None)
        state_helper.state["classes"] = self.prediction.get("classes", [])
        base_predict = self.prediction["predict"]
        base_low = self.prediction["low"]
        base_high = self.prediction["high"]
        state_helper.set_base_prediction(base_predict, base_low, base_high)

        for f, bins in self._guarded_bins.items():  # pylint: disable=invalid-name
            is_cat = f in self._guarded_categorical
            name = (
                self._guarded_feature_names[f] if f < len(self._guarded_feature_names) else str(f)
            )

            # Only include conforming, non-factual bins
            alt_bins = [b for b in bins if not b.is_factual and b.conforming]

            if is_cat:
                value_str = str(self.x_test[f])
            else:
                value_str = str(np.around(self.x_test[f], decimals=2))

            for gbin in alt_bins:
                # Skip if prediction is identical to baseline
                if gbin.predict == base_predict and gbin.low == base_low and gbin.high == base_high:
                    continue

                condition = _guarded_condition_str(name, gbin, is_cat)

                state_helper.add_rule(
                    predict=gbin.predict,
                    low=gbin.low,
                    high=gbin.high,
                    base_predict=base_predict,
                    value=value_str,
                    feature=f,
                    sampled_values=[gbin.representative],
                    feature_value=self.x_test[f],
                    rule_text=condition,
                    is_conjunctive=False,
                )

        self.rules = state_helper.get_state()
        self.has_rules = True
        return self.rules

    def get_guarded_audit(self) -> Dict[str, Any]:
        """Return interval-level guarded audit data for this alternative explanation."""
        ignored = self.ignored_features_for_instance()
        feature_names = list(self._guarded_feature_names or [])
        base_predict = self.prediction["predict"]
        base_low = self.prediction["low"]
        base_high = self.prediction["high"]
        intervals: list[dict[str, Any]] = []

        for feat, gbin in _sorted_guarded_bins(self._guarded_bins):
            name = _feature_name(feat, feature_names)
            condition = _guarded_condition_str(name, gbin, feat in self._guarded_categorical)

            if feat in ignored:
                emitted = False
                reason = "ignored_feature"
            elif not gbin.conforming:
                emitted = False
                reason = "removed_guard"
            elif gbin.is_factual:
                emitted = False
                reason = "design_excluded"
            elif gbin.predict == base_predict and gbin.low == base_low and gbin.high == base_high:
                emitted = False
                reason = "baseline_equal"
            else:
                emitted = True
                reason = "emitted"

            intervals.append(
                {
                    "feature": int(feat),
                    "feature_name": str(name),
                    "lower": _to_python(gbin.lower),
                    "upper": _to_python(gbin.upper),
                    "representative": _to_python(gbin.representative),
                    "p_value": _to_python(gbin.p_value),
                    "conforming": bool(gbin.conforming),
                    "is_factual": bool(gbin.is_factual),
                    "is_merged": bool(gbin.is_merged),
                    "emitted": bool(emitted),
                    "emission_reason": reason,
                    "condition": condition,
                    "predict": _to_python(gbin.predict),
                    "low": _to_python(gbin.low),
                    "high": _to_python(gbin.high),
                }
            )

        removed_guard = sum(1 for rec in intervals if not rec["conforming"])
        summary = {
            "intervals_tested": int(len(intervals)),
            "intervals_conforming": int(sum(1 for rec in intervals if rec["conforming"])),
            "intervals_removed_guard": int(removed_guard),
            "intervals_emitted": int(sum(1 for rec in intervals if rec["emitted"])),
            "features_with_any_removed_guard": int(
                len({rec["feature"] for rec in intervals if rec["emission_reason"] == "removed_guard"})
            ),
        }

        return {
            "mode": "alternative",
            "instance_index": int(self.index),
            "summary": summary,
            "intervals": intervals,
        }
