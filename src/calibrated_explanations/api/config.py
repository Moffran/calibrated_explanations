"""Configuration primitives for calibrated_explanations.

This module provides a configuration dataclass and a fluent builder for
constructing explainers with validated options. See `ADR-009` for
preprocessing-related fields and `ADR-034` §7 for env-var precedence rules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

from ..utils.exceptions import ConfigurationError

# Backward-compatible patch point used by tests. When set to a callable, build_config
# uses it instead of the internal factory builder.
_perf_from_config = None


@dataclass
class ExplainerConfig:
    """Configuration for building an explainer wrapper.

    Notes
    -----
    Fields wired by ``from_config()``
        ``model``, ``preprocessor``, ``auto_encode``, ``unseen_category_policy``,
        ``categorical_features``, ``missing_value_policy``; performance primitives
        (cache, parallel executor) via the perf factory; internal feature-filter config.

    Fields applied at explain-time
        ``threshold`` and ``low_high_percentiles`` are forwarded to
        ``explain_factual`` / ``explore_alternatives`` via ``kwargs.setdefault()``.

    ``WrapCalibratedExplainer`` auto-detects task from the fitted model;
    there is no ``task`` field. ``perf_parallel_workers`` is the governed
    parallel-worker count (``CE_PARALLEL`` env var takes precedence —
    see ADR-034 §7).
    """

    model: Any
    # Calibration / explanation knobs (subset; extend later as needed)
    low_high_percentiles: tuple[int, int] = (5, 95)
    threshold: float | None = None  # for probabilistic regression use-cases

    # Preprocessing (ADR-009)
    preprocessor: Any | None = None
    auto_encode: bool | Literal["auto"] = "auto"
    unseen_category_policy: Literal["ignore", "error"] = "error"
    categorical_features: tuple[int, ...] = ()
    missing_value_policy: Literal["category", "error"] = "category"

    # Performance feature flags (ADR-003/ADR-004) - disabled by default
    perf_cache_enabled: bool = False
    perf_cache_max_items: int = 512
    perf_cache_max_bytes: int | None = 32 * 1024 * 1024
    perf_cache_namespace: str = "calibrator"
    perf_cache_version: str = "v1"
    perf_cache_ttl: float | None = None
    perf_parallel_enabled: bool = False
    perf_parallel_backend: Literal["auto", "sequential", "joblib", "threads", "processes"] = "auto"
    perf_parallel_workers: int | None = None
    perf_parallel_min_batch: int = 8
    perf_parallel_min_instances: int | None = None
    perf_parallel_tiny_workload: int | None = None
    perf_parallel_granularity: Literal["instance"] = "instance"
    perf_telemetry: Any | None = None

    # Internal FAST-based feature filtering (disabled by default)
    perf_feature_filter_enabled: bool = False
    perf_feature_filter_per_instance_top_k: int = 8

    @property
    def perf_factory(self):
        """Factory for performance telemetry."""
        return getattr(self, "_perf_factory", None)


class ExplainerBuilder:
    """Fluent helper to assemble an :class:`ExplainerConfig`.

    In a later step this builder can produce a configured `WrapCalibratedExplainer`.
    """

    def __init__(self, model: Any) -> None:
        """Store base model reference and seed configuration defaults."""
        self._cfg = ExplainerConfig(model=model)

    # Simple fluent setters
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

    def categorical_features(self, features: tuple[int, ...] | list[int]) -> ExplainerBuilder:
        """Select categorical feature indices for built-in auto-encoding.

        Parameters
        ----------
        features : sequence of int
            Zero-based column indices to force through the categorical encoder.
        """
        self._cfg.categorical_features = tuple(int(feature) for feature in features)
        return self

    def missing_value_policy(self, policy: Literal["category", "error"]) -> ExplainerBuilder:
        """Select the handling strategy for missing categorical values.

        Parameters
        ----------
        policy : {"category", "error"}
            Whether missing categorical values become a deterministic category
            or raise a validation error.
        """
        self._cfg.missing_value_policy = policy
        return self

    # Perf flags (feature-flagged; no behavior change when off)
    def perf_cache(
        self,
        enabled: bool,
        *,
        max_items: int | None = None,
        max_bytes: int | None = None,
        namespace: str | None = None,
        version: str | None = None,
        ttl: float | None = None,
    ) -> ExplainerBuilder:
        """Enable or disable the performance cache options.

        Parameters
        ----------
        enabled : bool
            Flag indicating whether caching primitives should be provisioned.
        max_items : int, optional
            Maximum number of cached entries when caching is enabled.

        Notes
        -----
        ``CE_CACHE`` env var takes precedence over ``enabled`` because
        ``CacheConfig.from_env()`` is applied after builder construction
        inside ``_build_perf_factory()`` (ADR-034 §7).
        """
        self._cfg.perf_cache_enabled = enabled
        if max_items is not None:
            self._cfg.perf_cache_max_items = max_items
        if max_bytes is not None:
            self._cfg.perf_cache_max_bytes = max_bytes
        if namespace is not None:
            self._cfg.perf_cache_namespace = namespace
        if version is not None:
            self._cfg.perf_cache_version = version
        if ttl is not None:
            self._cfg.perf_cache_ttl = ttl
        return self

    def perf_parallel(
        self,
        enabled: bool,
        *,
        backend: Literal["auto", "sequential", "joblib", "threads", "processes"] | None = None,
        workers: int | None = None,
        min_batch: int | None = None,
        min_instances: int | None = None,
        tiny_workload: int | None = None,
        granularity: Literal["instance"] | None = None,
    ) -> ExplainerBuilder:
        """Configure the parallel backend used for performance operations.

        Parameters
        ----------
        enabled : bool
            Whether parallel primitives should be created.
        backend : {"auto", "sequential", "joblib"}, optional
            Explicit backend selection overriding the default when provided.

        Notes
        -----
        ``CE_PARALLEL`` env var takes precedence over ``enabled`` because
        ``ParallelConfig.from_env()`` is applied after builder construction
        inside ``_build_perf_factory()`` (ADR-034 §7).
        """
        self._cfg.perf_parallel_enabled = enabled
        if backend is not None:
            self._cfg.perf_parallel_backend = backend
        if workers is not None:
            self._cfg.perf_parallel_workers = workers
        if min_batch is not None:
            self._cfg.perf_parallel_min_batch = min_batch
        if min_instances is not None:
            self._cfg.perf_parallel_min_instances = min_instances
        if tiny_workload is not None:
            self._cfg.perf_parallel_tiny_workload = tiny_workload
        if granularity is not None:
            if granularity != "instance":
                raise ConfigurationError(
                    "perf_parallel_granularity='feature' is not supported. Use 'instance'.",
                    details={
                        "param": "perf_parallel_granularity",
                        "received": granularity,
                        "allowed": ["instance"],
                    },
                )
            self._cfg.perf_parallel_granularity = granularity
        return self

    def perf_telemetry(self, callback: Any | None) -> ExplainerBuilder:
        """Register a telemetry callback shared by cache and parallel executors."""
        self._cfg.perf_telemetry = callback
        return self

    def perf_feature_filter(
        self,
        enabled: bool,
        *,
        per_instance_top_k: int | None = None,
    ) -> ExplainerBuilder:
        """Configure internal FAST-based feature filtering.

        Parameters
        ----------
        enabled : bool
            Flag indicating whether the internal FAST-based feature filter is enabled.
        per_instance_top_k : int, optional
            Maximum number of features to keep per instance based on FAST weights.
        """
        self._cfg.perf_feature_filter_enabled = enabled
        if per_instance_top_k is not None:
            self._cfg.perf_feature_filter_per_instance_top_k = max(1, int(per_instance_top_k))
        return self

    def build_config(self) -> ExplainerConfig:
        """Return the assembled configuration (no side effects).

        Raises
        ------
        ConfigurationError
            If the performance cache or parallel executor configuration
            (builder flags merged with ``CE_CACHE``/``CE_PARALLEL``) cannot be
            resolved. ``details`` name the ``capability``, the ``source`` that
            failed and the underlying ``cause``.
        """
        # Warn if callers injected removed fields via direct attribute assignment
        # (task / parallel_workers were removed in v0.11.3 — see ADR-034 §7).
        for _removed in ("task", "parallel_workers"):
            if hasattr(self._cfg, _removed):
                warnings.warn(
                    f"ExplainerConfig.{_removed} is not applicable and has been removed "
                    f"(v0.11.3). The field has no effect and will be ignored. "
                    f"Remove it from your configuration.",
                    UserWarning,
                    stacklevel=2,
                )
        # Attach the perf factory so from_config() can provision cache/parallel
        # primitives. Cache and parallel are opt-in only (ADR-003/ADR-004), so a
        # failure here means an explicit request cannot be honoured: fail closed
        # instead of silently degrading to no cache / sequential execution.
        factory_builder = _perf_from_config or _build_perf_factory
        try:
            self._cfg._perf_factory = factory_builder(self._cfg)  # type: ignore[attr-defined]
        except ConfigurationError:
            raise
        except Exception as exc:  # adr002_allow: re-raised as ConfigurationError
            raise _perf_initialization_error("perf_primitives", "factory", exc) from exc
        return self._cfg


_PERF_CAPABILITY_LABELS = {
    "cache": "performance cache",
    "parallel": "parallel executor",
    "perf_primitives": "performance primitives",
}


def _perf_initialization_error(
    capability: str,
    source: str,
    exc: Exception,
    *,
    requested_by: str | None = None,
) -> ConfigurationError:
    """Build the fail-closed error for a cache/parallel initialization failure.

    Parameters
    ----------
    capability : {"cache", "parallel", "perf_primitives"}
        The performance capability that could not be initialized.
    source : str
        Where the failure arose: ``"config"`` (builder/``ExplainerConfig``
        values), ``"CE_CACHE"``/``"CE_PARALLEL"`` (environment overrides) or
        ``"factory"`` (primitive construction).
    exc : Exception
        The underlying cause.
    requested_by : str, optional
        Activation source recorded by the perf factory, when known.
    """
    details: dict[str, Any] = {
        "capability": capability,
        "source": source,
        "cause": f"{type(exc).__name__}: {exc}",
    }
    if requested_by is not None:
        details["requested_by"] = requested_by
    label = _PERF_CAPABILITY_LABELS[capability]
    return ConfigurationError(f"Cannot initialize the {label} ({source}): {exc}", details=details)


def _activation_source(requested_in_config: bool, enabled: bool) -> str | None:
    """Classify how a perf capability ended up enabled or disabled.

    Returns ``"config"`` or ``"env"`` for the layer that enabled it,
    ``"disabled_by_env"`` when a configured request was switched off by the
    environment, and ``None`` when it was never requested.
    """
    if enabled:
        return "config" if requested_in_config else "env"
    return "disabled_by_env" if requested_in_config else None


class _ConfigPerfFactory:
    """Internal cache/parallel primitive builder for config-based wrapper wiring.

    ``activation`` records, per capability (``"cache"``/``"parallel"``), whether
    it was explicitly activated and by which layer (see ``_activation_source``).
    There is no implicit activation path: both capabilities are opt-in only.
    """

    def __init__(
        self,
        cache_cfg: Any,
        parallel_cfg: Any,
        activation: dict[str, str | None] | None = None,
    ) -> None:
        self._cache_cfg = cache_cfg
        self._parallel_cfg = parallel_cfg
        self.activation: dict[str, str | None] = dict(activation or {})

    def make_cache(self) -> Any:
        """Build a cache backend based on the stored configuration."""
        from ..cache import CalibratorCache

        return CalibratorCache(self._cache_cfg)

    def make_parallel_executor(self, cache: Any | None = None) -> Any:
        """Create a parallel executor wired to the stored parallel configuration."""
        from ..parallel import ParallelExecutor

        return ParallelExecutor(self._parallel_cfg, cache=cache)

    def make_parallel_backend(self, cache: Any | None = None) -> Any:
        """Alias for :meth:`make_parallel_executor`."""
        return self.make_parallel_executor(cache=cache)


def _build_perf_factory(cfg: Any) -> _ConfigPerfFactory:
    """Create perf primitives from config without using removed perf root facade.

    Raises
    ------
    ConfigurationError
        If the configured values or the ``CE_CACHE``/``CE_PARALLEL`` overrides
        cannot be resolved (see :func:`_perf_initialization_error`).
    """
    from ..cache import CacheConfig
    from ..parallel import ParallelConfig

    cache_requested = bool(getattr(cfg, "perf_cache_enabled", False))
    cache_cfg = CacheConfig(
        enabled=cache_requested,
        namespace=getattr(cfg, "perf_cache_namespace", "calibrator"),
        version=getattr(cfg, "perf_cache_version", "v1"),
        max_items=getattr(cfg, "perf_cache_max_items", 512),
        max_bytes=getattr(cfg, "perf_cache_max_bytes", 32 * 1024 * 1024),
        ttl_seconds=getattr(cfg, "perf_cache_ttl", None),
        telemetry=getattr(cfg, "perf_telemetry", None),
    )
    try:
        cache_cfg = CacheConfig.from_env(cache_cfg)
    except Exception as exc:  # adr002_allow: re-raised as ConfigurationError
        raise _perf_initialization_error("cache", "CE_CACHE", exc) from exc

    parallel_requested = bool(getattr(cfg, "perf_parallel_enabled", False))
    try:
        parallel_cfg = ParallelConfig(
            enabled=parallel_requested,
            strategy=getattr(cfg, "perf_parallel_backend", "sequential"),
            max_workers=getattr(cfg, "perf_parallel_workers", None),
            min_batch_size=getattr(cfg, "perf_parallel_min_batch", 8),
            min_instances_for_parallel=getattr(cfg, "perf_parallel_min_instances", None),
            tiny_workload_threshold=getattr(cfg, "perf_parallel_tiny_workload", None),
            granularity=getattr(cfg, "perf_parallel_granularity", "instance"),
            telemetry=getattr(cfg, "perf_telemetry", None),
        )
    except Exception as exc:  # adr002_allow: re-raised as ConfigurationError
        raise _perf_initialization_error("parallel", "config", exc) from exc
    try:
        parallel_cfg = ParallelConfig.from_env(parallel_cfg)
    except Exception as exc:  # adr002_allow: re-raised as ConfigurationError
        raise _perf_initialization_error("parallel", "CE_PARALLEL", exc) from exc

    activation = {
        "cache": _activation_source(cache_requested, cache_cfg.enabled),
        "parallel": _activation_source(parallel_requested, parallel_cfg.enabled),
    }
    return _ConfigPerfFactory(cache_cfg=cache_cfg, parallel_cfg=parallel_cfg, activation=activation)


__all__ = [
    "ExplainerConfig",
    "ExplainerBuilder",
]
