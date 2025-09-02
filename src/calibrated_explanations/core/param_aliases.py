"""Parameter alias canonicalization utilities (Phase 1B).

Provides helpers to normalize user-facing keyword arguments across the codebase
to a canonical set, reducing argument drift and easing future refactors.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .exceptions import ValidationError

# Global alias map: alias -> canonical
_ALIASES: Mapping[str, str] = {
    # naming variants
    "percentiles": "low_high_percentiles",
    "lowhigh": "low_high_percentiles",
    "low_high": "low_high_percentiles",
    "interval_percentiles": "low_high_percentiles",
    "thresholds": "threshold",
    "random_state": "seed",
    "fast_mode": "fast",
    "noise": "noise_type",
    "scale": "scale_factor",
    "severity_level": "severity",
    "ignore_features": "features_to_ignore",
    "features_ignore": "features_to_ignore",
    # british/american spelling
    "discretiser": "discretizer",
    # categorical config
    "cat_features": "categorical_features",
    "cat_labels": "categorical_labels",
    "class_names": "class_labels",
    # mondrian categories
    "mondrian_categories": "bins",
    "mondrian": "bins",
    "mc": "bins",
}


def canonicalize_params(
    params: Dict[str, Any], *, raise_on_conflict: bool = False
) -> Dict[str, Any]:
    """Return a new dict with parameter keys normalized to canonical names.

    - If both alias and canonical keys are present with different values and
      ``raise_on_conflict`` is True, raises ValidationError.
    - Prefers canonical key when both present and values match or conflict resolution disabled.
    """
    if not params:
        return {}
    out: Dict[str, Any] = dict(params)
    for alias, canon in _ALIASES.items():
        if alias in out:
            if canon in out:
                if raise_on_conflict and out[canon] != out[alias]:
                    raise ValidationError(
                        f"Conflicting values for '{canon}' and alias '{alias}'."
                    )
                # Prefer canonical, drop alias
                out.pop(alias, None)
            else:
                out[canon] = out.pop(alias)
    return out


__all__ = ["canonicalize_params"]

