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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        ``True`` when the majority of sampled representative values within
        this bin produce in-distribution perturbed instances.
    p_value : float
        Conformal p-value for the representative value (or mean p-value when
        multiple candidates were sampled).
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


@dataclass
class GuardedFeatureExplanation:
    """Guarded explanation for a single feature of a single instance.

    Attributes
    ----------
    feature_idx : int
    feature_name : str
    current_value : Any
        Observed value of the feature in the original instance.
    bins : list of :class:`GuardedBin`
        All bins produced by the discretiser (including non-conforming ones).
    is_categorical : bool
        ``True`` when the feature is categorical (no interval merging).
    """

    feature_idx: int
    feature_name: str
    current_value: Any
    bins: list[GuardedBin] = field(default_factory=list)
    is_categorical: bool = False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def factual_bin(self) -> GuardedBin | None:
        """Return the bin containing the original feature value, or None."""
        for b in self.bins:
            if b.is_factual:
                return b
        return None

    def conforming_bins(self) -> list[GuardedBin]:
        """Return all conforming bins (including factual)."""
        return [b for b in self.bins if b.conforming]

    def alternative_bins(self) -> list[GuardedBin]:
        """Return conforming non-factual bins (i.e., reachable alternatives)."""
        return [b for b in self.bins if not b.is_factual and b.conforming]

    def condition_str(self, b: GuardedBin) -> str:
        """Return a human-readable interval condition string for bin *b*.

        Examples
        --------
        * ``"age <= 30.00"``  (first bin, ``lower == -inf``)
        * ``"30.00 < age <= 50.00"``  (interior bin)
        * ``"age > 50.00"``  (last bin, ``upper == inf``)
        * ``"colour == red"``  (categorical feature)
        """
        name = self.feature_name
        if self.is_categorical:
            return f"{name} == {b.representative!r}"
        if b.lower == -np.inf and b.upper == np.inf:
            return f"{name} (any value)"
        if b.lower == -np.inf:
            return f"{name} <= {b.upper:.4g}"
        if b.upper == np.inf:
            return f"{name} > {b.lower:.4g}"
        return f"{b.lower:.4g} < {name} <= {b.upper:.4g}"


@dataclass
class GuardedExplanation:
    """Guarded explanation for a single test instance.

    Attributes
    ----------
    instance_idx : int
        Index of the instance within the original batch passed to the
        explain method.
    instance : np.ndarray of shape (n_features,)
        Original feature vector.
    feature_explanations : list of :class:`GuardedFeatureExplanation`
    baseline_predict : float
        Calibrated prediction for the original (unperturbed) instance.
    baseline_low : float
        Lower bound of the calibrated prediction interval for the original
        instance.
    baseline_high : float
        Upper bound of the calibrated prediction interval.
    mode : str
        One of ``'factual'`` or ``'alternative'``.
    """

    instance_idx: int
    instance: np.ndarray
    feature_explanations: list[GuardedFeatureExplanation]
    baseline_predict: float
    baseline_low: float
    baseline_high: float
    mode: str  # 'factual' or 'alternative'

    def as_rules(self, mode: str | None = None) -> list[dict]:
        """Return a list of rule dicts suitable for display or further processing.

        Parameters
        ----------
        mode : str or None
            ``'factual'`` returns the rule for the instance's own bin;
            ``'alternative'`` returns rules for all conforming non-factual bins.
            When ``None``, the explanation's own *mode* attribute is used.

        Returns
        -------
        rules : list of dict
            Each dict contains:

            ``'feature'``
                Feature name.
            ``'condition'``
                Interval condition string (e.g. ``"30 < age <= 50"``).
            ``'predict'``, ``'low'``, ``'high'``
                Calibrated prediction and interval for the rule.
            ``'weight'``
                ``predict - baseline_predict``.
            ``'conforming'``
                Whether the bin passed the in-distribution test.
            ``'p_value'``
                Conformal p-value of the representative value.
            ``'is_merged'``
                Whether the interval was created by adjacent merging.
        """
        m = mode or self.mode
        rules: list[dict] = []
        for fe in self.feature_explanations:
            if m == "factual":
                b = fe.factual_bin()
                if b is not None:
                    rules.append(self._rule_dict(fe, b))
            else:  # alternative
                for b in fe.alternative_bins():
                    rules.append(self._rule_dict(fe, b))
        return rules

    def _rule_dict(self, fe: GuardedFeatureExplanation, b: GuardedBin) -> dict:
        return {
            "feature": fe.feature_name,
            "condition": fe.condition_str(b),
            "predict": b.predict,
            "low": b.low,
            "high": b.high,
            "weight": b.predict - self.baseline_predict,
            "conforming": b.conforming,
            "p_value": b.p_value,
            "is_merged": b.is_merged,
        }

    def __repr__(self) -> str:  # pragma: no cover
        rules = self.as_rules()
        return (
            f"GuardedExplanation(instance={self.instance_idx}, "
            f"mode={self.mode!r}, "
            f"baseline={self.baseline_predict:.4f}, "
            f"n_rules={len(rules)})"
        )


class GuardedExplanations:
    """An ordered collection of :class:`GuardedExplanation` objects.

    This class is the return type of
    :meth:`~calibrated_explanations.CalibratedExplainer.explain_guarded_factual`
    and
    :meth:`~calibrated_explanations.CalibratedExplainer.explore_guarded_alternatives`.

    Parameters
    ----------
    explanations : list of :class:`GuardedExplanation`
    mode : str
        ``'factual'`` or ``'alternative'``.
    """

    def __init__(self, explanations: list[GuardedExplanation], mode: str) -> None:
        self.explanations = explanations
        self.mode = mode

    def __iter__(self):
        return iter(self.explanations)

    def __len__(self) -> int:
        return len(self.explanations)

    def __getitem__(self, idx):
        return self.explanations[idx]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GuardedExplanations(mode={self.mode!r}, n_instances={len(self)})"
        )
