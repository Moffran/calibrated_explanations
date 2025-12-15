"""Explain executor system for calibrated explanations.

This package provides a plugin-based architecture for explain execution strategies:
- Sequential: single-threaded feature-by-feature processing
- Feature-parallel: parallel processing across features
- Instance-parallel: parallel processing across instances

The plugin system replaces branching logic in CalibratedExplainer.explain,
providing clean separation between orchestration and execution strategies.
"""

from __future__ import annotations

import warnings as _warnings

from ._base import BaseExplainExecutor
from ._shared import ExplainConfig, ExplainRequest, ExplainResponse
from .orchestrator import ExplanationOrchestrator
from .parallel_instance import InstanceParallelExplainExecutor
from .sequential import SequentialExplainExecutor


def explain(*args, **kwargs):
    """Forward calls to the legacy explain implementation (deprecated)."""
    _warnings.warn(
        "calibrated_explanations.core.explain.explain is deprecated; use CalibratedExplainer.explain_factual instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ._legacy_explain import explain as _legacy_explain

    return _legacy_explain(*args, **kwargs)


__all__ = [
    "explain",
    "BaseExplainExecutor",
    "ExplainConfig",
    "ExplainRequest",
    "ExplainResponse",
    "ExplanationOrchestrator",
    "InstanceParallelExplainExecutor",
    "SequentialExplainExecutor",
]
