"""Deprecated shim forwarding to :mod:`calibrated_explanations.parallel`.

This module previously contained the parallel execution facade. Per ADR-001
the canonical implementation now lives in ``calibrated_explanations.parallel``.

Compatibility notes
-------------------
* Importing this module emits a :class:`DeprecationWarning`.
* The shim will be removed after v1.1.0 once downstream callers migrate.
* All attributes are forwarded to :mod:`calibrated_explanations.parallel.parallel`
  to preserve monkeypatch targets used by existing tests and plugins.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..parallel import parallel as _parallel

warnings.warn(
    "The 'calibrated_explanations.perf.parallel' module is deprecated. "
    "Use 'calibrated_explanations.parallel' instead. "
    "This shim will be removed after v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export canonical symbols
ParallelConfig = _parallel.ParallelConfig
ParallelExecutor = _parallel.ParallelExecutor
ParallelMetrics = _parallel.ParallelMetrics

# Expose implementation details so legacy monkeypatch targets continue to work
os = _parallel.os
ThreadPoolExecutor = _parallel.ThreadPoolExecutor
ProcessPoolExecutor = _parallel.ProcessPoolExecutor
_JoblibParallel = _parallel._JoblibParallel
_joblib_delayed = _parallel._joblib_delayed


def __getattr__(name: str) -> Any:  # pragma: no cover - thin shim
    return getattr(_parallel, name)


__all__ = ["ParallelConfig", "ParallelExecutor", "ParallelMetrics"]
