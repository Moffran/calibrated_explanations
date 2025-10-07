"""Parameter canonicalization utilities (Phase 1B).

This module centralizes lightweight argument normalization and basic
combination checks without changing external behavior.

Notes
-----
- No deprecation warnings or behavior changes here in Phase 1B.
- Unknown parameters are preserved as-is.
- Canonicalization only maps known aliases to canonical keys when the
  canonical key is not already provided.

See ADR-002 for context.
"""

from __future__ import annotations

import warnings
from typing import Any

# Minimal, conservative alias map. Extend in Phase 2.
ALIAS_MAP: dict[str, str] = {
    # Example: users sometimes pass statistical alpha(s) instead of percentiles.
    # Mapping stays syntactic only in 1B; no semantic conversion performed.
    "alpha": "low_high_percentiles",
    "alphas": "low_high_percentiles",
    # Parallelism terminology normalization (future-facing; not wired yet).
    "n_jobs": "parallel_workers",
}


def canonicalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs with known aliases mapped to canonical keys.

    Rules
    -----
    - If an alias exists in ``kwargs`` and the canonical key is absent, copy the
      value to the canonical key.
    - If both alias and canonical are present, keep the canonical value and do not
      overwrite it.
    - Always preserve original keys; we do not delete aliases in Phase 1B to avoid
      any chance of behavior drift. Callers should read canonical keys first.
    - Unknown keys are left untouched.
    """
    out = dict(kwargs)
    for alias, canonical in ALIAS_MAP.items():
        if alias in kwargs and canonical not in kwargs:
            out[canonical] = kwargs[alias]
    return out


def validate_param_combination(kwargs: dict[str, Any]) -> None:
    """Perform basic consistency checks for parameter combinations.

    Phase 1B keeps this intentionally minimal to avoid changing existing behavior.
    Add stricter checks in Phase 2 alongside clearer user-facing messaging.
    """
    # Example placeholder: keep for future expansion. No-op for now.
    return None


__all__ = ["ALIAS_MAP", "canonicalize_kwargs", "validate_param_combination"]


def warn_on_aliases(kwargs: dict[str, Any]) -> None:
    """Emit deprecation warnings when known alias keys are used.

    Notes
    -----
    - No behavior change; only a `DeprecationWarning` to guide users.
    - Intended to be called at public boundaries in Phase 2.
    """
    for alias, canonical in ALIAS_MAP.items():
        if alias in kwargs:
            warnings.warn(
                f"Parameter '{alias}' is deprecated; use '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )


__all__.append("warn_on_aliases")
