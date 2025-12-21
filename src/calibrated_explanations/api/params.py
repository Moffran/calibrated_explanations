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

# Minimal, conservative alias map.
ALIAS_MAP: dict[str, str] = {
    # Example: users sometimes pass statistical alpha(s) instead of percentiles.
    # Mapping stays syntactic only in 1B; no semantic conversion performed.
    "alpha": "low_high_percentiles",
    "alphas": "low_high_percentiles",
    # Parallelism terminology normalization (future-facing; not wired yet).
    "n_jobs": "parallel_workers",
}

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
    out = dict(kwargs)
    for alias, canonical in ALIAS_MAP.items():
        if alias in kwargs and canonical not in kwargs:
            out[canonical] = kwargs[alias]
    return out


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


__all__ = ["ALIAS_MAP", "canonicalize_kwargs", "validate_param_combination"]


def warn_on_aliases(kwargs: dict[str, Any]) -> None:
    """Emit deprecation warnings when known alias keys are used.

    Notes
    -----
    - No behavior change; only a `DeprecationWarning` to guide users.
    """
    import warnings as _warnings

    from ..utils import deprecate_alias

    for alias, canonical in ALIAS_MAP.items():
        if alias in kwargs:
            # Emit a lightweight UserWarning so outer `warnings.catch_warnings`
            # contexts (used in unit tests) can observe the deprecation even
            # when pytest's `pytest.warns` context manager intercepts
            # DeprecationWarning emissions.
            _warnings.warn(
                "Parameter or alias '" + alias + "' is deprecated; use '" + canonical + "'",
                UserWarning,
                stacklevel=3,
            )
            deprecate_alias(alias, canonical, stacklevel=3)


__all__.append("warn_on_aliases")
