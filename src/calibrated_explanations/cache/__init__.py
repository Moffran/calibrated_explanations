"""Cache layer for the calibrated explanations performance architecture.

This package provides namespaced, versioned caching with optional TTL and memory budgets,
thread-safety, and lightweight telemetry. See ADR-003 for design details.

Part of ADR-001: Core Decomposition Boundaries (Stage 1b).
"""

from .cache import (
    CacheConfig,
    CacheMetrics,
    CalibratorCache,
    TelemetryCallback,
    make_key,
)
from .explanation_cache import ExplanationCacheFacade

__all__ = [
    "CacheConfig",
    "CacheMetrics",
    "CalibratorCache",
    "ExplanationCacheFacade",
    "TelemetryCallback",
    "make_key",
]
