"""Parallel execution entry points (ADR-001 Stage 5 API tightening).

Only the stable executor façade and configuration helpers are re-exported from
the package root. Lower-level chunking utilities remain internal.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .parallel import ParallelConfig, ParallelExecutor, ParallelMetrics


__all__ = (
    "ParallelConfig",
    "ParallelExecutor",
    "ParallelMetrics",
)

_NAME_TO_MODULE = {
    "ParallelConfig": ("parallel", "ParallelConfig"),
    "ParallelExecutor": ("parallel", "ParallelExecutor"),
    "ParallelMetrics": ("parallel", "ParallelMetrics"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose the sanctioned parallel API surface."""

    if name not in __all__:
        raise AttributeError(name)

    module_name, attr_name = _NAME_TO_MODULE[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
