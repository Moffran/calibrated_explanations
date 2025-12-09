"""Cache entry points (ADR-001 Stage 5 API tightening).

Only the stable cache interfaces are exposed from the package root to prevent
callers from depending on backend-specific helpers. This keeps the public
surface lintable by the import graph checker.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .cache import (
        CacheConfig,
        CacheMetrics,
        CalibratorCache,
        LRUCache,
        TelemetryCallback,
        _default_size_estimator,
        _hash_part,
        make_key,
    )
    from .explanation_cache import ExplanationCacheFacade


__all__ = (
    "CacheConfig",
    "CacheMetrics",
    "CalibratorCache",
    "ExplanationCacheFacade",
    "LRUCache",
    "TelemetryCallback",
    "_default_size_estimator",
    "_hash_part",
    "make_key",
)

_NAME_TO_MODULE = {
    "CacheConfig": ("cache", "CacheConfig"),
    "CacheMetrics": ("cache", "CacheMetrics"),
    "CalibratorCache": ("cache", "CalibratorCache"),
    "LRUCache": ("cache", "LRUCache"),
    "TelemetryCallback": ("cache", "TelemetryCallback"),
    "_default_size_estimator": ("cache", "_default_size_estimator"),
    "_hash_part": ("cache", "_hash_part"),
    "make_key": ("cache", "make_key"),
    "ExplanationCacheFacade": ("explanation_cache", "ExplanationCacheFacade"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose cache interfaces from the package root."""
    if name not in __all__:
        raise AttributeError(name)

    module_name, attr_name = _NAME_TO_MODULE[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
