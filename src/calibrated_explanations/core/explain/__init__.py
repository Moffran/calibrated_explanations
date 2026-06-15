"""Explain executor system for calibrated explanations.

This package provides a plugin-based architecture for explain execution strategies:
- Sequential: single-threaded feature-by-feature processing
- Feature-parallel: parallel processing across features
- Instance-parallel: parallel processing across instances

The plugin system replaces branching logic in CalibratedExplainer.explain,
providing clean separation between orchestration and execution strategies.
"""

from __future__ import annotations

from ._base import BaseExplainExecutor
from ._computation import discretize, rule_boundaries
from ._legacy_explain import explain as legacy_explain
from ._shared import ExplainConfig, ExplainRequest, ExplainResponse
from .orchestrator import ExplanationOrchestrator
from .parallel_instance import InstanceParallelExplainExecutor
from .sequential import SequentialExplainExecutor


def explain(*args, **kwargs):
    """Forward calls to the legacy explain implementation (deprecated)."""
    from calibrated_explanations.utils.deprecations import (
        deprecate,  # pylint: disable=import-outside-toplevel
    )

    deprecate(
        "calibrated_explanations.core.explain.explain is deprecated and will be removed in v1.0.0; "
        "use CalibratedExplainer.explain_factual instead.",
        key="core.explain.explain",
        stacklevel=2,
        raise_on_error=False,
    )
    from ._legacy_explain import explain as _legacy_explain

    return _legacy_explain(*args, **kwargs)


__all__ = [
    "explain",
    "legacy_explain",
    "discretize",
    "rule_boundaries",
    "BaseExplainExecutor",
    "ExplainConfig",
    "ExplainRequest",
    "ExplainResponse",
    "ExplanationOrchestrator",
    "InstanceParallelExplainExecutor",
    "SequentialExplainExecutor",
]
