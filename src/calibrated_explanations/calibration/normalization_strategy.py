"""Multiclass interval normalization strategies for Venn-Abers OvR calibration."""

from __future__ import annotations

from enum import Enum
from typing import Any


class NormalizationStrategy(Enum):
    """Strategy for normalizing multiclass OvR Venn-Abers probability intervals.

    Independent binary OvR calibrators produce per-class intervals that carry
    no joint constraint.  A normalization step is needed to make the outputs
    useful as a multiclass probability distribution.  Four strategies are
    available; they trade off probabilistic coherence of the *bounds* against
    preservation of class-wise *interval widths* (a natural difficulty signal):

    - SCALE (default): scale both bounds and point estimates uniformly by
      ``1/S`` where ``S = Σ_c p_c`` (the raw row sum of point estimates), so
      that point estimates sum to 1, the relative width ordering across classes
      is preserved, and bounds remain consistent with the scaled point estimates.

    - SIMPLEX: preserve raw VA bounds; normalize only the point estimates to
      sum to 1.  Interval widths reflect the calibrator's per-class uncertainty
      and differ across classes, but the displayed bounds are not rescaled to
      match the normalized point estimates.

    - COHERENCE: enforce the additive coherence constraint
      ``h_c + Σ_{k≠c} l_k = 1`` for every class c by adjusting upper bounds
      while preserving lower bounds.  A mathematical consequence is that all
      interval widths become equal (``D_c = 1 − Σ_k l_k``), erasing the
      per-class difficulty signal.  Point estimates are then simplex-normalized.

    - NONE: return raw, un-normalized VA outputs.  Neither the bounds nor the
      point estimates are guaranteed to satisfy any joint constraint.  Useful
      for diagnostics and research.

    Notes
    -----
    SCALE and SIMPLEX both preserve the relative ordering of interval widths
    across classes.  They differ in whether the *bounds* are also rescaled:

    - Under SCALE the displayed ``[l_c, h_c]`` are the raw VA outputs divided
      by ``S``, keeping them consistent with the normalized ``p_c``.
    - Under SIMPLEX the displayed ``[l_c, h_c]`` are exactly the raw VA
      outputs; only the scalar ``p_c`` is rescaled.

    COHERENCE forces equal interval widths — an undesirable side effect when
    visualizing which classes carry more uncertainty.

    References
    ----------
    See the companion paper (Section: Contribution) and the incompatibility
    theorem discussed in the Future Work section for a formal proof that
    COHERENCE and varying widths are mutually exclusive.
    """

    SIMPLEX = "simplex"
    COHERENCE = "coherence"
    NONE = "none"
    SCALE = "scale"


def coerce_normalization_strategy(value: Any) -> NormalizationStrategy:
    """Return a validated NormalizationStrategy, defaulting to SIMPLEX.

    Accepts a ``NormalizationStrategy`` member, its string value
    (case-insensitive), or a legacy bool (``True`` → COHERENCE with a
    deprecation warning, ``False`` → NONE with a deprecation warning).
    Any unrecognised value falls back to SCALE.

    Parameters
    ----------
    value : Any
        The raw parameter value supplied by the caller.

    Returns
    -------
    NormalizationStrategy
        The resolved strategy.
    """
    if isinstance(value, NormalizationStrategy):
        return value
    if isinstance(value, bool):
        from calibrated_explanations.utils.deprecations import (
            deprecate,  # pylint: disable=import-outside-toplevel
        )

        if value:
            deprecate(
                "Passing normalize=True is deprecated and will be removed in v1.0.0. "
                "Use normalization=NormalizationStrategy.COHERENCE (or 'coherence') instead. "
                "The new default is NormalizationStrategy.SIMPLEX.",
                key="normalize_true_bool",
                stacklevel=3,
            )
            return NormalizationStrategy.COHERENCE
        deprecate(
            "Passing normalize=False is deprecated and will be removed in v1.0.0. "
            "Use normalization=NormalizationStrategy.NONE (or 'none') instead.",
            key="normalize_false_bool",
            stacklevel=3,
        )
        return NormalizationStrategy.NONE
    if isinstance(value, str):
        try:
            return NormalizationStrategy(value.lower())
        except ValueError:
            pass
    return NormalizationStrategy.SCALE


__all__ = ["NormalizationStrategy", "coerce_normalization_strategy"]
