"""Configuration primitives for calibrated_explanations (Phase 2 scaffolding).

This module introduces a light-weight configuration dataclass and a builder
to simplify constructing explainers with validated options. In this phase,
no wiring to core classes is performed to avoid behavior changes; consumers
may import and use these types for future-facing code.

See `RELEASE_PLAN_v1` milestone targets and ADR-009 for preprocessing-related fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..perf import from_config as _perf_from_config

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

    # Performance feature flags (ADR-003/ADR-004) – disabled by default
    perf_cache_enabled: bool = False
    perf_cache_max_items: int = 128
    perf_parallel_enabled: bool = False
    perf_parallel_backend: Literal["auto", "sequential", "joblib"] = "auto"


class ExplainerBuilder:
    """Fluent helper to assemble an :class:`ExplainerConfig`.

    In a later step this builder can produce a configured `WrapCalibratedExplainer`.
    For now it only creates the config to avoid runtime changes in Phase 2 start.
    """

    def __init__(self, model: Any) -> None:
        self._cfg = ExplainerConfig(model=model)

    # Simple fluent setters
    def task(self, task: TaskLiteral) -> ExplainerBuilder:
        """Set the task type for the explainer configuration.

        Parameters
        ----------
        task : {"classification", "regression", "auto"}
            Desired task literal to store in the configuration.
        """
        self._cfg.task = task
        return self

    def low_high_percentiles(self, p: tuple[int, int]) -> ExplainerBuilder:
        """Update the percentile pair for interval explanations.

        Parameters
        ----------
        p : tuple of int
            Inclusive lower and upper percentiles used for interval computation.
        """
        self._cfg.low_high_percentiles = p
        return self

    def threshold(self, t: float | None) -> ExplainerBuilder:
        """Store a regression-style threshold value on the configuration.

        Parameters
        ----------
        t : float or None
            Threshold applied when producing probabilistic regression outputs.
        """
        self._cfg.threshold = t
        return self

    def preprocessor(self, pre: Any | None) -> ExplainerBuilder:
        """Attach an optional preprocessing object to the configuration.

        Parameters
        ----------
        pre : Any or None
            Preprocessor applied to inputs prior to fitting or calibration.
        """
        self._cfg.preprocessor = pre
        return self

    def auto_encode(self, flag: bool | Literal["auto"]) -> ExplainerBuilder:
        """Toggle automatic categorical encoding behavior.

        Parameters
        ----------
        flag : bool or "auto"
            Whether to auto-encode categorical inputs when preprocessing.
        """
        self._cfg.auto_encode = flag
        return self

    def unseen_category_policy(self, policy: Literal["ignore", "error"]) -> ExplainerBuilder:
        """Select the strategy for handling unseen categorical values.

        Parameters
        ----------
        policy : {"ignore", "error"}
            Policy to apply when encountering unseen categories at inference time.
        """
        self._cfg.unseen_category_policy = policy
        return self

    def parallel_workers(self, n: int | None) -> ExplainerBuilder:
        """Configure the desired number of parallel worker processes.

        Parameters
        ----------
        n : int or None
            Worker count for parallel execution; ``None`` leaves the default in place.
        """
        self._cfg.parallel_workers = n
        return self

    # Perf flags (feature-flagged; no behavior change when off)
    def perf_cache(self, enabled: bool, *, max_items: int | None = None) -> ExplainerBuilder:
        """Enable or disable the performance cache options.

        Parameters
        ----------
        enabled : bool
            Flag indicating whether caching primitives should be provisioned.
        max_items : int, optional
            Maximum number of cached entries when caching is enabled.
        """
        self._cfg.perf_cache_enabled = enabled
        if max_items is not None:
            self._cfg.perf_cache_max_items = max_items
        return self

    def perf_parallel(
        self, enabled: bool, *, backend: Literal["auto", "sequential", "joblib"] | None = None
    ) -> ExplainerBuilder:
        """Configure the parallel backend used for performance operations.

        Parameters
        ----------
        enabled : bool
            Whether parallel primitives should be created.
        backend : {"auto", "sequential", "joblib"}, optional
            Explicit backend selection overriding the default when provided.
        """
        self._cfg.perf_parallel_enabled = enabled
        if backend is not None:
            self._cfg.perf_parallel_backend = backend
        return self

    def build_config(self) -> ExplainerConfig:
        """Return the assembled configuration (no side effects)."""
        # attach a perf factory convenience object when building config so later
        # consumers can opt-in to perf primitives consistently. This does not
        # change behavior unless the factory is used.
        try:
            # stash a lightweight factory on the config for downstream wiring
            self._cfg._perf_factory = _perf_from_config(self._cfg)  # type: ignore[attr-defined]
        except Exception:
            # be conservative: do not fail config building if perf factory creation fails
            self._cfg._perf_factory = None  # type: ignore[attr-defined]
        return self._cfg


__all__ = [
    "ExplainerConfig",
    "ExplainerBuilder",
    "TaskLiteral",
]
