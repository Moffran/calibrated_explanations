"""Stable API surface for orchestrator-friendly helpers (ADR-001 Stage 5)."""

from __future__ import annotations

from .config import ExplainerBuilder, ExplainerConfig
from .params import canonicalize_kwargs, validate_param_combination, warn_on_aliases
from .quick import quick_explain

__all__ = [
    "ExplainerBuilder",
    "ExplainerConfig",
    "canonicalize_kwargs",
    "validate_param_combination",
    "warn_on_aliases",
    "quick_explain",
]
