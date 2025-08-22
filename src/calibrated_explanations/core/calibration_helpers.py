"""Phase 1A calibration delegators (scaffolding).

These helpers wrap internal interval-learner setup/update and threshold
assignment to prepare for a later mechanical extraction.
"""

from __future__ import annotations

from typing import Any


def assign_threshold(explainer: Any, threshold):  # pragma: no cover - thin wrapper
    return explainer.assign_threshold(threshold)


def initialize_interval_learner(explainer: Any) -> None:  # pragma: no cover
    return explainer._CalibratedExplainer__initialize_interval_learner()  # noqa: SLF001


def update_interval_learner(explainer: Any, xs, ys, bins=None) -> None:  # pragma: no cover
    return explainer._CalibratedExplainer__update_interval_learner(xs, ys, bins=bins)  # noqa: SLF001


__all__ = [
    "assign_threshold",
    "initialize_interval_learner",
    "update_interval_learner",
]
