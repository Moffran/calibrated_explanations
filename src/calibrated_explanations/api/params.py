"""Parameter canonicalization utilities (ADR-002).

This module centralizes lightweight argument normalization and consistency
checks, enabling downstream validators and plugins to rely on a stable
parameter contract.

Notes
-----
- Alias mapping enables backward compatibility without behavior drift.
- Combination validation enforces ADR-compliant constraints (e.g., conflicting
  parameter exclusivity).
- Canonicalization only maps known aliases to canonical keys when the
  canonical key is not already provided.

See ADR-002 for context.
"""

from __future__ import annotations

from typing import Any

from ..utils.exceptions import ConfigurationError

# Removed aliases (v0.11.0).
REMOVED_ALIAS_MAP: dict[str, str] = {
    "alpha": "low_high_percentiles",
    "alphas": "low_high_percentiles",
    "n_jobs": "parallel_workers",
}

# Kept for API compatibility; no active alias mapping remains after v0.11.0.
ALIAS_MAP: dict[str, str] = {}

# Parameter combinations that are mutually exclusive or conflicting.
EXCLUSIVE_PARAM_GROUPS: list[tuple[str, ...]] = [
    # Example: threshold and confidence_level are alternative ways to specify coverage.
    # Callers should choose one or the other, not both.
    ("threshold", "confidence_level"),
]


def canonicalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs with known aliases mapped to canonical keys.

    Notes
    -----
    - If an alias exists in ``kwargs`` and the canonical key is absent, copy the
      value to the canonical key.
    - If both alias and canonical are present, keep the canonical value and do not
      overwrite it.
    - Always preserve original keys; we do not delete aliases to avoid
      any chance of behavior drift. Callers should read canonical keys first.
    - Unknown keys are left untouched.
    """
    return dict(kwargs)


def reject_removed_aliases(kwargs: dict[str, Any]) -> None:
    """Reject aliases removed in v0.11.0 with actionable migration guidance."""
    used = {alias: canonical for alias, canonical in REMOVED_ALIAS_MAP.items() if alias in kwargs}
    if not used:
        return
    formatted = ", ".join(f"'{alias}' -> '{canonical}'" for alias, canonical in used.items())
    raise ConfigurationError(
        "Deprecated parameter aliases were removed in v0.11.0. "
        f"Use canonical names instead: {formatted}.",
        details={
            "removed_aliases": list(used.keys()),
            "canonical_replacements": used,
            "removed_in": "v0.11.0",
        },
    )


def validate_param_combination(kwargs: dict[str, Any]) -> None:
    """Perform basic consistency checks for parameter combinations (ADR-002).

    Enforces mutual exclusivity and conflict constraints. Raises
    ``ConfigurationError`` when violations are detected.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to validate.

    Raises
    ------
    ConfigurationError
        When conflicting parameter combinations are detected.
    """
    # Check mutual exclusivity groups
    for param_group in EXCLUSIVE_PARAM_GROUPS:
        present = [p for p in param_group if p in kwargs and kwargs[p] is not None]
        if len(present) > 1:
            raise ConfigurationError(
                f"Parameters {present} are mutually exclusive; specify at most one.",
                details={
                    "conflict": param_group,
                    "provided": present,
                    "requirement": "choose one or none",
                },
            )


__all__ = [
    "ALIAS_MAP",
    "REMOVED_ALIAS_MAP",
    "canonicalize_kwargs",
    "reject_removed_aliases",
    "validate_param_combination",
]


def warn_on_aliases(kwargs: dict[str, Any]) -> None:
    """Compatibility wrapper for the removed alias guard.

    Notes
    -----
    - `warn_on_aliases` historically emitted deprecation warnings.
    - Since aliases were removed in v0.11.0, this now fails fast.
    """
    reject_removed_aliases(kwargs)


__all__.append("warn_on_aliases")
