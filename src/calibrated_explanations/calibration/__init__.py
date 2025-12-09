"""Calibration entry points (ADR-001 Stage 5 API tightening).

This package exposes only the sanctioned calibration primitives through the
package root to keep the public surface small and lintable. Callers should
avoid importing from internal modules such as ``interval_learner`` directly;
all supported functionality is available from here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .interval_learner import (
        assign_threshold,
        initialize_interval_learner,
        initialize_interval_learner_for_fast_explainer,
        update_interval_learner,
    )
    from .interval_regressor import IntervalRegressor
    from .state import CalibrationState
    from .summaries import (
        get_calibration_summaries,
        invalidate_calibration_summaries,
    )
    from .venn_abers import VennAbers


__all__ = (
    "CalibrationState",
    "IntervalRegressor",
    "VennAbers",
    "assign_threshold",
    "get_calibration_summaries",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "invalidate_calibration_summaries",
    "update_interval_learner",
)

_NAME_TO_MODULE = {
    "CalibrationState": ("state", "CalibrationState"),
    "IntervalRegressor": ("interval_regressor", "IntervalRegressor"),
    "VennAbers": ("venn_abers", "VennAbers"),
    "assign_threshold": ("interval_learner", "assign_threshold"),
    "get_calibration_summaries": ("summaries", "get_calibration_summaries"),
    "initialize_interval_learner": (
        "interval_learner",
        "initialize_interval_learner",
    ),
    "initialize_interval_learner_for_fast_explainer": (
        "interval_learner",
        "initialize_interval_learner_for_fast_explainer",
    ),
    "invalidate_calibration_summaries": (
        "summaries",
        "invalidate_calibration_summaries",
    ),
    "update_interval_learner": ("interval_learner", "update_interval_learner"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load sanctioned calibration symbols.

    The lazy indirection keeps import time light while ensuring callers only use
    the package root as the public boundary, per ADR-001 Stage 5 guidance.
    """
    if name not in __all__:
        raise AttributeError(name)

    module_name, attr_name = _NAME_TO_MODULE[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
