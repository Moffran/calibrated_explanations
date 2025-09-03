"""Configuration primitives for calibrated_explanations (Phase 2 scaffolding).

This module introduces a light-weight configuration dataclass and a builder
to simplify constructing explainers with validated options. In this phase,
no wiring to core classes is performed to avoid behavior changes; consumers
may import and use these types for future-facing code.

See ACTION_PLAN Phase 2 and ADR-009 for preprocessing-related fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TaskLiteral = Literal["classification", "regression", "auto"]


@dataclass
class ExplainerConfig:
    """Configuration for building an explainer wrapper.

    Notes
    -----
    - Fields included here are future-facing; not all are used in Phase 2 start.
    - Keep defaults aligned with existing behavior to prevent drift when adopted.
    """

    model: Any
    task: TaskLiteral = "auto"
    # Calibration / explanation knobs (subset; extend later as needed)
    low_high_percentiles: tuple[int, int] = (5, 95)
    threshold: float | None = None  # for probabilistic regression use-cases

    # Preprocessing (ADR-009)
    preprocessor: Any | None = None
    auto_encode: bool | Literal["auto"] = "auto"
    unseen_category_policy: Literal["ignore", "error"] = "error"

    # Parallelism placeholder (not wired yet)
    parallel_workers: int | None = None


class ExplainerBuilder:
    """Fluent helper to assemble an :class:`ExplainerConfig`.

    In a later step this builder can produce a configured `WrapCalibratedExplainer`.
    For now it only creates the config to avoid runtime changes in Phase 2 start.
    """

    def __init__(self, model: Any) -> None:
        self._cfg = ExplainerConfig(model=model)

    # Simple fluent setters
    def task(self, task: TaskLiteral) -> ExplainerBuilder:
        self._cfg.task = task
        return self

    def low_high_percentiles(self, p: tuple[int, int]) -> ExplainerBuilder:
        self._cfg.low_high_percentiles = p
        return self

    def threshold(self, t: float | None) -> ExplainerBuilder:
        self._cfg.threshold = t
        return self

    def preprocessor(self, pre: Any | None) -> ExplainerBuilder:
        self._cfg.preprocessor = pre
        return self

    def auto_encode(self, flag: bool | Literal["auto"]) -> ExplainerBuilder:
        self._cfg.auto_encode = flag
        return self

    def unseen_category_policy(self, policy: Literal["ignore", "error"]) -> ExplainerBuilder:
        self._cfg.unseen_category_policy = policy
        return self

    def parallel_workers(self, n: int | None) -> ExplainerBuilder:
        self._cfg.parallel_workers = n
        return self

    def build_config(self) -> ExplainerConfig:
        """Return the assembled configuration (no side effects)."""
        return self._cfg


__all__ = [
    "ExplainerConfig",
    "ExplainerBuilder",
    "TaskLiteral",
]
