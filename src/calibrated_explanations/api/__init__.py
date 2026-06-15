"""Stable API surface for orchestrator-friendly helpers (ADR-001 Stage 5)."""

from __future__ import annotations

from .config import ExplainerBuilder, ExplainerConfig
from .decorators import experimental
from .params import (
    canonicalize_kwargs,
    reject_removed_aliases,
    validate_param_combination,
    warn_on_aliases,
)
from .quick import quick_explain

__all__ = [
    "ExplainerBuilder",
    "ExplainerConfig",
    "experimental",
    "canonicalize_kwargs",
    "reject_removed_aliases",
    "validate_param_combination",
    "warn_on_aliases",
    "quick_explain",
]
