"""Explain plugin system for calibrated explanations.

This package provides a plugin-based architecture for explain execution strategies:
- Sequential: single-threaded feature-by-feature processing
- Feature-parallel: parallel processing across features
- Instance-parallel: parallel processing across instances

The plugin system replaces branching logic in CalibratedExplainer.explain,
providing clean separation between orchestration and execution strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ._base import BaseExplainPlugin
from ._shared import ExplainConfig, ExplainRequest, ExplainResponse
from .orchestrator import ExplanationOrchestrator
from .parallel_feature import FeatureParallelExplainPlugin
from .parallel_instance import InstanceParallelExplainPlugin
from .sequential import SequentialExplainPlugin

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


__all__ = [
    "BaseExplainPlugin",
    "ExplainConfig",
    "ExplainRequest",
    "ExplainResponse",
    "ExplanationOrchestrator",
    "FeatureParallelExplainPlugin",
    "InstanceParallelExplainPlugin",
    "SequentialExplainPlugin",
]
