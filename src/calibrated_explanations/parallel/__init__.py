"""Parallel execution helpers for the calibrated explanations runtime.

This package provides strategy selection and graceful fallbacks for parallel
and sequential execution. See ADR-004 for design details.

Part of ADR-001: Core Decomposition Boundaries (Stage 1b).
"""

from .parallel import (
    ParallelConfig,
    ParallelExecutor,
    ParallelMetrics,
)

__all__ = [
    "ParallelConfig",
    "ParallelExecutor",
    "ParallelMetrics",
]
