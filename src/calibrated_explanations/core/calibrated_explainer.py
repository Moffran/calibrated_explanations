"""Calibrated Explanations for Black-Box Predictions (calibrated-explanations).

The calibrated explanations explanation method is based on the paper
"Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

Calibrated explanations are a way to explain the predictions of a black-box learner
using Venn-Abers predictors (classification & regression) or
conformal predictive systems (regression).
"""

# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-lines, too-many-positional-arguments, too-many-public-methods
from __future__ import annotations

import copy
import sys
import contextlib
from time import time
from typing import TYPE_CHECKING

import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from ..explanations import AlternativeExplanations, CalibratedExplanations
    from ..plugins.manager import PluginManager

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _tomllib = None  # type: ignore[assignment]

# Core imports (no cross-sibling dependencies)
from ..calibration.interval_wrappers import is_fast_interval_collection
from ..utils import check_is_fitted, convert_targets_to_numeric, safe_isinstance

from ..utils.exceptions import (
    DataShapeError,
    ValidationError,
)
from .reject.policy import RejectPolicy
from .prediction.interval_summary import IntervalSummary, coerce_interval_summary

# Lazy imports deferred to avoid cross-sibling coupling
# These are imported inside methods/properties where used
# - perf (CalibratorCache, ParallelExecutor) - lazy in __init__
# - plotting (_plot_global) - lazy in plotting methods
# - explanations (AlternativeExplanations, CalibratedExplanations) - lazy as needed
# - integrations (LimeHelper, ShapHelper) - lazy in __init__
# - api.params (canonicalize_kwargs, etc.) - lazy in param handling
# - plugins (IntervalCalibratorContext, PluginManager, LegacyPredictBridge) - lazy in __init__
# - utils.discretizers (EntropyDiscretizer, RegressorDiscretizer) - lazy in validation


class CalibratedExplainer:
    """The :class:`.CalibratedExplainer` class is used for explaining machine learning learners with calibrated predictions.

    The calibrated explanations are based on the paper
    "Calibrated Explanations for Black-Box Predictions"
    by Helena Löfström, Tuwe Löfström, Ulf Johansson and Cecilia Sönströd.

    Calibrated explanations provides a way to explain the predictions of a black-box learner
    using Venn-Abers predictors (classification) or
    conformal predictive systems (regression).
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(
        self,
        learner,
        x_cal,
        y_cal,
        mode="classification",
        feature_names=None,
        categorical_features=None,
        categorical_labels=None,
        class_labels=None,
        bins=None,
        difficulty_estimator=None,
        **kwargs,
    ) -> None:
        """Initialize the explainer with calibration data and metadata.

        Parameters
        ----------
        learner : Any
            Predictive learner that must already expose ``fit``/``predict`` and,
            for classification, ``predict_proba``.
        x_cal : array-like of shape (n_calibration_samples, n_features)
            Calibration feature matrix used to fit interval calibrators.
        y_cal : array-like of shape (n_calibration_samples,)
            Calibration targets paired with ``x_cal``.
        mode : {"classification", "regression"}, default="classification"
            Operating mode controlling which calibrators/plugins are used.
        feature_names : Sequence[str] or None, optional
            Optional list of human-readable feature names.
        categorical_features : Sequence[int] or None, optional
            Indices describing which features should be treated as categorical.
        categorical_labels : Mapping[int, Mapping[int, str]] or None, optional
            Optional mapping translating categorical feature values to labels.
        class_labels : Mapping[int, str] or None, optional
            Optional mapping translating class indices to display labels.
        bins : array-like or None, optional
            Pre-computed Mondrian categories for fast explanations.
        difficulty_estimator : Any or None, optional
            Optional crepes ``DifficultyEstimator`` instance for regression tasks.
        **kwargs : Any
            Advanced configuration flags preserved for backward compatibility.

        Notes
        -----
        Minimal lifecycle logging is available at INFO level. To enable, run::

            import logging
            logging.getLogger("calibrated_explanations").setLevel(logging.INFO)
        """
        perf_cache = kwargs.pop("perf_cache", None)
        perf_parallel = kwargs.pop("perf_parallel", None)

        init_time = time()
        self.__initialized = False
        preprocessor_metadata = kwargs.pop("preprocessor_metadata", None)
        if isinstance(preprocessor_metadata, Mapping):
            self._preprocessor_metadata: Dict[str, Any] | None = dict(preprocessor_metadata)
        else:
            self._preprocessor_metadata = None
        check_is_fitted(learner)
        self.learner = learner
        self.predict_function = kwargs.get("predict_function")
        if self.predict_function is None:
            self.predict_function = (
                learner.predict_proba if mode == "classification" else learner.predict
            )
        # Optionally suppress or convert low-level crepes errors into clearer messages.
        # Caller can pass suppress_crepes_errors=True via kwargs to avoid raising on
        # crepes broadcasting/shape errors (useful for synthetic tiny datasets).
        self.suppress_crepes_errors = bool(kwargs.get("suppress_crepes_errors", False))
        self.oob = kwargs.get("oob", False)
        self._categorical_value_counts_cache: Dict[int, Dict[Any, int]] | None = None
        self._numeric_sorted_cache: Dict[int, np.ndarray] | None = None
        self._calibration_summary_shape: Tuple[int, int] | None = None
        if self.oob:
            if mode == "classification":
                y_oob_proba = self.learner.oob_decision_function_
                if (
                    len(y_oob_proba.shape) == 1 or y_oob_proba.shape[1] == 1
                ):  # Binary classification
                    y_oob = (y_oob_proba > 0.5).astype(np.dtype(y_cal.dtype))
                else:  # Multiclass classification
                    y_oob = np.argmax(y_oob_proba, axis=1)
                    if safe_isinstance(y_cal, "pandas.core.arrays.categorical.Categorical"):
                        y_oob = y_cal.categories[y_oob]
                    else:
                        y_oob = y_oob.astype(np.dtype(y_cal.dtype))
            else:
                y_oob = self.learner.oob_prediction_
            if len(x_cal) != len(y_oob):
                raise DataShapeError(
                    "The length of the out-of-bag predictions does not match the length of X_cal."
                )
            y_cal = y_oob
        self.x_cal = x_cal
        self.y_cal = y_cal

        # Initialize RNG with seed
        from ..utils import set_rng_seed  # pylint: disable=import-outside-toplevel

        seed = kwargs.get("seed", 42)
        self.seed = seed
        self.rng = set_rng_seed(seed)

        self.sample_percentiles = kwargs.get("sample_percentiles", [25, 50, 75])
        self.verbose = kwargs.get("verbose", False)
        self.bins = bins
        self.interval_summary = coerce_interval_summary(
            kwargs.get("interval_summary", IntervalSummary.REGULARIZED_MEAN)
        )

        self.__fast = kwargs.get("fast", False)
        self.__noise_type = kwargs.get("noise_type", "uniform")
        self.__scale_factor = kwargs.get("scale_factor", 5)
        self.__severity = kwargs.get("severity", 1)
        self.condition_source = kwargs.get("condition_source", "observed")
        if self.condition_source not in {"observed", "prediction"}:
            raise ValidationError(
                "condition_source must be either 'observed' or 'prediction'",
                details={
                    "param": "condition_source",
                    "value": self.condition_source,
                    "allowed": ("observed", "prediction"),
                },
            )

        self.categorical_labels = categorical_labels
        self.class_labels = class_labels
        if categorical_features is None:
            if categorical_labels is not None:
                categorical_features = categorical_labels.keys()
            else:
                categorical_features = []
        self.categorical_features = list(categorical_features)
        self._invalidate_calibration_summaries()
        self.features_to_ignore = kwargs.get("features_to_ignore", [])

        # Identify constant calibration features that can be ignored downstream
        from .calibration_helpers import identify_constant_features  # pylint: disable=import-outside-toplevel

        self.features_to_ignore = identify_constant_features(self.x_cal)

        if feature_names is None:
            feature_names = (
                self._X_cal[0].keys()
                if isinstance(self._X_cal[0], dict)
                else [str(i) for i in range(self.num_features)]
            )
        self._feature_names = list(feature_names)

        if mode == "classification":
            if any(isinstance(val, str) for val in self.y_cal) or any(
                isinstance(val, (np.str_, np.object_)) for val in self.y_cal
            ):
                self.y_cal_numeric, self.label_map = convert_targets_to_numeric(self.y_cal)
                self.y_cal = self.y_cal_numeric  # save to _y_cal to avoid append
                if self.class_labels is None:
                    self.class_labels = {v: k for k, v in self.label_map.items()}
            else:
                self.label_map = None
                if self.class_labels is None:
                    self.class_labels = {int(label): str(label) for label in np.unique(self.y_cal)}
        else:
            self.label_map = None
            self.class_labels = None

        self.discretizer: Any = None
        self.discretized_X_cal: Optional[np.ndarray] = None
        # Predeclare attributes for fast mode to satisfy type checkers
        self.fast_x_cal: Optional[np.ndarray] = None
        self.scaled_x_cal: Optional[np.ndarray] = None
        self.scaled_y_cal: Optional[np.ndarray] = None

        self.feature_values: Dict[int, List[Any]] = {}
        self.feature_frequencies: Dict[int, np.ndarray] = {}

        # Lazy import helper integrations (deferred from module level)
        from ..integrations import LimeHelper, ShapHelper

        self.latest_explanation: Optional[CalibratedExplanations] = None
        self._lime_helper = LimeHelper(self)
        self._shap_helper = ShapHelper(self)
        self.reject = kwargs.get("reject", False)
        # Optional default reject policy for explainer-level defaults
        from .reject.policy import RejectPolicy as _RejectPolicy

        self.default_reject_policy = kwargs.get("default_reject_policy", _RejectPolicy.NONE)

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.set_mode(str.lower(mode), initialize=False)

        # Lazy import orchestrator and plugin management (deferred from module level)
        from ..plugins.manager import PluginManager
        from ..plugins.builtins import LegacyPredictBridge
        from ..cache import CalibratorCache

        # Initialize plugin manager (SINGLE SOURCE OF TRUTH for plugin management)
        # PluginManager handles ALL plugin initialization including:
        # - Reading pyproject.toml configurations
        # - Setting up plugin overrides from kwargs
        # - Creating and initializing orchestrators
        # - Building plugin fallback chains
        self.plugin_manager = PluginManager(self)
        self.plugin_manager.initialize_from_kwargs(kwargs)
        self.plugin_manager.initialize_orchestrators()

        # Initialize interval learner after orchestrators are ready
        self.prediction_orchestrator.interval_registry.initialize()

        self.perf_cache: CalibratorCache[Any] | None = perf_cache

        # Initialize parallel executor (ADR-004: Honor CE_PARALLEL overrides)
        self._perf_parallel: Any | None = self._resolve_parallel_executor(perf_parallel)

        # Orchestrator references are now accessed via properties that delegate to PluginManager
        # No direct assignment needed - properties handle the delegation

        # Reject learner initialization
        self.reject_learner = (
            self.initialize_reject_learner() if kwargs.get("reject", False) else None
        )

        self._predict_bridge = LegacyPredictBridge(self)

        self.init_time = time() - init_time

    # TODO: Needs to be
    def __deepcopy__(self, memo):
        """Safely deepcopy the explainer, handling circular references."""
        if id(self) in memo:
            return memo[id(self)]
        # Create a shallow copy without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Manually copy attributes
        # Some attributes are runtime helpers or refer back into the explainer
        # (plugin manager, parallel executor, caches, integration helpers, etc.).
        # Deep-copying these can cause recursion or try to copy unpicklable objects.
        # Shallow-copy them instead to preserve references and avoid recursion.
        shallow_copy_keys = {
            "_plugin_manager",
            "_perf_parallel",
            "perf_cache",
            "_lime_helper",
            "_shap_helper",
            "_predict_bridge",
            "latest_explanation",
            "learner",
            "predict_function",
            "rng",
        }

        for k, v in self.__dict__.items():
            if k in shallow_copy_keys:
                # ADR002_ALLOW: swallowing to keep deepcopy best-effort.
                with contextlib.suppress(Exception):
                    setattr(result, k, v)
                continue

            try:
                setattr(result, k, copy.deepcopy(v, memo))
            except (
                Exception
            ):  # ADR002_ALLOW: fallback to shallow copy when deepcopy fails.  # pragma: no cover
                # Fallback: if deepcopy fails for any reason, keep original reference.
                # ADR002_ALLOW: ignore attributes that cannot be copied.
                with contextlib.suppress(Exception):
                    setattr(result, k, v)

        return result

    def __getstate__(self):
        """Exclude runtime helpers when pickling."""
        state = self.__dict__.copy()
        state["perf_cache"] = None
        state["_perf_parallel"] = None
        return state

    def __setstate__(self, state):
        """Restore state after pickling without restoring helpers."""
        self.__dict__.update(state)

    def require_plugin_manager(self) -> PluginManager:
        """Return the plugin manager or raise if the explainer is not initialized.

        Returns
        -------
        PluginManager
            The active plugin manager instance.

        Raises
        ------
        NotFittedError
            If the plugin manager is not initialized.
        """
        from ..utils.exceptions import NotFittedError

        manager = getattr(self, "_plugin_manager", None)
        if manager is None:
            raise NotFittedError(
                "PluginManager is not initialized. Instantiate CalibratedExplainer via __init__.",
                details={
                    "state": "uninitialized",
                    "reason": "plugin_manager_missing",
                    "required_method": "__init__",
                },
            )
        return manager

    def _resolve_parallel_executor(self, explicit_executor: Any | None) -> Any | None:
        """Resolve the parallel executor honoring overrides and environment config."""
        return self.resolve_parallel_executor(explicit_executor)

    def resolve_parallel_executor(self, explicit_executor: Any | None) -> Any | None:
        """Resolve the parallel executor honoring overrides and environment config."""
        from ..parallel import ParallelConfig, ParallelExecutor

        if explicit_executor is not None:
            return explicit_executor

        env_config = ParallelConfig.from_env()
        if env_config.enabled:
            return ParallelExecutor(env_config)

        return None

    # ------------------------------------------------------------------
    # Parallel pool lifecycle helpers
    # ------------------------------------------------------------------
    def initialize_pool(self, n_workers: int | None = None, *, pool_at_init: bool = False) -> None:
        """Create a `ParallelExecutor` for this explainer.

        Parameters
        ----------
        n_workers: int | None
            Optional maximum worker count to enforce.
        pool_at_init: bool
            If True, enter the pool immediately so worker processes are
            spawned at initialization time (useful for warm-up and
            initializer-based harness installation).
        """
        from ..parallel import ParallelConfig, ParallelExecutor

        if getattr(self, "_perf_parallel", None) is not None:
            return

        cfg = ParallelConfig.from_env()
        cfg.enabled = True
        if n_workers is not None:
            cfg.max_workers = n_workers

        # If requested, set up a worker initializer that will receive a
        # compact explainer spec. Keep the spec deliberately small and
        # picklable.
        if pool_at_init:
            # ADR002_ALLOW: optional initializer wiring should not block.
            with contextlib.suppress(Exception):
                import calibrated_explanations.core.explain.parallel_runtime as pr_mod

                # Build a picklable compact spec containing only the data
                # required to rehydrate an explainer in worker processes.
                # Attempt to include a picklable learner payload. If the
                # learner is not picklable, fall back to omitting it so the
                # worker initializer must handle a missing learner case.
                learner_bytes = None
                try:
                    import pickle  # nosec B403

                    learner_bytes = pickle.dumps(getattr(self, "learner", None))
                except (
                    Exception
                ):  # ADR002_ALLOW: learner pickling best-effort fallback.  # pragma: no cover
                    learner_bytes = None

                spec = {
                    "learner_bytes": learner_bytes,
                    "x_cal": getattr(self, "x_cal", None),
                    "y_cal": getattr(self, "y_cal", None),
                    "mode": getattr(self, "mode", None),
                    "num_features": getattr(self, "num_features", None),
                    "bins": getattr(self, "bins", None),
                    "sample_percentiles": getattr(self, "sample_percentiles", None),
                }
                cfg.worker_initializer = pr_mod.worker_init_from_explainer_spec
                cfg.worker_init_args = (spec,)

        self._perf_parallel = ParallelExecutor(cfg)
        if pool_at_init:
            # Enter context to spawn worker pool now
            self._perf_parallel.__enter__()

    def close(self) -> None:
        """Shutdown any provisioned parallel pool and release resources."""
        perf = getattr(self, "_perf_parallel", None)
        if perf is None:
            return
        try:
            perf.__exit__(None, None, None)
        finally:
            self._perf_parallel = None

    def __enter__(self) -> "CalibratedExplainer":
        """Context manager entry; create and enter a worker pool."""
        self.initialize_pool(pool_at_init=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; close any provisioned pool."""
        self.close()

    def infer_explanation_mode(self) -> str:
        """Infer the explanation mode from runtime state."""
        # Lazy import discretizers (deferred from module level)
        from ..utils import EntropyDiscretizer, RegressorDiscretizer

        # Check discretizer type to infer mode
        discretizer = self.discretizer if hasattr(self, "discretizer") else None
        if discretizer is not None and isinstance(
            discretizer, (EntropyDiscretizer, RegressorDiscretizer)
        ):
            return "alternative"
        # All other discretizers (Binary*, or None) indicate factual
        return "factual"

    # ===================================================================
    # Delegation methods for orchestrator operations
    # ===================================================================
    # These methods delegate to PluginManager and orchestrators.
    # PluginManager is the single source of truth for plugin defaults and chains.
    # Tests that call these directly MUST initialize PluginManager properly.

    @property
    def prediction_orchestrator(self) -> Any:
        """Return the PredictionOrchestrator provisioned by the PluginManager."""
        return self.require_plugin_manager().prediction_orchestrator

    @prediction_orchestrator.setter
    def prediction_orchestrator(self, value: Any) -> None:
        """Set the PredictionOrchestrator."""
        self.require_plugin_manager().prediction_orchestrator = value

    @prediction_orchestrator.deleter
    def prediction_orchestrator(self) -> None:
        """Delete the PredictionOrchestrator."""
        del self.require_plugin_manager().prediction_orchestrator

    @property
    def explanation_orchestrator(self) -> Any:
        """Return the ExplanationOrchestrator provisioned by the PluginManager."""
        return self.require_plugin_manager().explanation_orchestrator

    @explanation_orchestrator.setter
    def explanation_orchestrator(self, value: Any) -> None:
        """Set the ExplanationOrchestrator."""
        self.require_plugin_manager().explanation_orchestrator = value

    @explanation_orchestrator.deleter
    def explanation_orchestrator(self) -> None:
        """Delete the ExplanationOrchestrator."""
        del self.require_plugin_manager().explanation_orchestrator

    @property
    def reject_orchestrator(self) -> Any:
        """Return the RejectOrchestrator provisioned by the PluginManager."""
        return self.require_plugin_manager().reject_orchestrator

    @reject_orchestrator.setter
    def reject_orchestrator(self, value: Any) -> None:
        """Set the RejectOrchestrator."""
        self.require_plugin_manager().reject_orchestrator = value

    @reject_orchestrator.deleter
    def reject_orchestrator(self) -> None:
        """Delete the RejectOrchestrator."""
        del self.require_plugin_manager().reject_orchestrator

    def build_plot_style_chain(self) -> Tuple[str, ...]:
        """Return the plot style chain.

        This is the public replacement for the legacy internal helper. It delegates
        to :class:`PluginManager` to construct the chain when available and
        otherwise returns an empty tuple for minimal explainer stubs used in tests.
        """
        return self.plugin_manager.build_plot_chain()

    def instantiate_plugin(self, prototype: Any) -> Any:
        """Delegate to ExplanationOrchestrator."""
        return self.plugin_manager.explanation_orchestrator.instantiate_plugin(prototype)

    def build_instance_telemetry_payload(self, explanations: Any) -> Dict[str, Any]:
        """Delegate to ExplanationOrchestrator."""
        return self.explanation_orchestrator.build_instance_telemetry_payload(explanations)

    def _invoke_explanation_plugin(self, *args, **kwargs) -> Any:
        """Invoke the explanation plugin with the given parameters."""
        return self.invoke_explanation_plugin(*args, **kwargs)

    def invoke_explanation_plugin(
        self,
        mode: str,
        x: Any,
        threshold: Any,
        low_high_percentiles: Any,
        bins: Any,
        features_to_ignore: Any,
        *,
        extras: Mapping[str, Any] | None = None,
        reject_policy: Any | None = None,
    ) -> Any:
        """Delegate to ExplanationOrchestrator."""
        # Reject integration (ADR-029):
        # - default remains RejectPolicy.NONE (no reject)
        # - per-call reject_policy overrides the explainer-level default_reject_policy
        # Backward compatibility:
        # - do not pass reject_policy=None / RejectPolicy.NONE through to orchestrator calls
        from .reject.policy import RejectPolicy

        candidate_policy = reject_policy
        if candidate_policy is None:
            candidate_policy = getattr(self, "default_reject_policy", RejectPolicy.NONE)

        try:
            effective_policy = RejectPolicy(candidate_policy)
        except Exception:
            effective_policy = RejectPolicy.NONE

        if effective_policy is RejectPolicy.NONE:
            return self.explanation_orchestrator.invoke(
                mode,
                x,
                threshold,
                low_high_percentiles,
                bins,
                features_to_ignore,
                extras=extras,
            )

        # Policy enabled: ensure reject orchestration and delegate via RejectOrchestrator
        def _explain_fn(x_subset, **kw):
            return self.explanation_orchestrator.invoke(
                mode,
                x_subset,
                threshold,
                low_high_percentiles,
                kw.get("bins", bins),
                features_to_ignore,
                extras=extras,
                _ce_skip_reject=True,
            )

        # Implicitly enable reject orchestration
        try:
            # ensure plugin manager has set up orchestrators
            _ = self.reject_orchestrator
        except Exception:
            # fallback: initialize via plugin manager if available
            try:
                self.plugin_manager.initialize_orchestrators()
            except Exception:
                pass

        return self.reject_orchestrator.apply_policy(effective_policy, x, explain_fn=_explain_fn, bins=bins)

    def ensure_interval_runtime_state(self) -> None:
        """Delegate to PredictionOrchestrator."""
        return self.prediction_orchestrator.ensure_interval_runtime_state()

    def gather_interval_hints(self, *, fast: bool) -> Tuple[str, ...]:
        """Delegate to PredictionOrchestrator."""
        return self.prediction_orchestrator.gather_interval_hints(fast=fast)

    # ===================================================================
    # Backward-compatibility properties for plugin state (via PluginManager)
    # ===================================================================
    # These properties delegate to PluginManager for backward compatibility
    # with code that accesses plugin state directly from explainer.

    @property
    def _interval_plugin_hints(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_hints

    @_interval_plugin_hints.setter
    def _interval_plugin_hints(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_hints = value

    @_interval_plugin_hints.deleter
    def _interval_plugin_hints(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_plugin_hints

    @property
    def _interval_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_fallbacks

    @_interval_plugin_fallbacks.setter
    def _interval_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_fallbacks = value

    @_interval_plugin_fallbacks.deleter
    def _interval_plugin_fallbacks(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_plugin_fallbacks

    @property
    def _interval_preferred_identifier(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_preferred_identifier

    @_interval_preferred_identifier.setter
    def _interval_preferred_identifier(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_preferred_identifier = value

    @_interval_preferred_identifier.deleter
    def _interval_preferred_identifier(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_preferred_identifier

    @property
    def _telemetry_interval_sources(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        return self.plugin_manager.telemetry_interval_sources

    @_telemetry_interval_sources.setter
    def _telemetry_interval_sources(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.telemetry_interval_sources = value

    @_telemetry_interval_sources.deleter
    def _telemetry_interval_sources(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.telemetry_interval_sources

    @property
    def _interval_plugin_identifiers(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_identifiers

    @_interval_plugin_identifiers.setter
    def _interval_plugin_identifiers(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_identifiers = value

    @_interval_plugin_identifiers.deleter
    def _interval_plugin_identifiers(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_plugin_identifiers

    @property
    def _interval_context_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_context_metadata

    @_interval_context_metadata.setter
    def _interval_context_metadata(self, value: Dict[str, Dict[str, Any]]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_context_metadata = value

    @_interval_context_metadata.deleter
    def _interval_context_metadata(self) -> None:
        """Delegate to PluginManager."""
        del self.plugin_manager.interval_context_metadata

    @property
    def plot_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Return the plot plugin fallback configuration.

        Returns
        -------
        Dict[str, Tuple[str, ...]]
            Mapping of mode to fallback identifiers.
        """
        return self.plugin_manager.plot_plugin_fallbacks

    @plot_plugin_fallbacks.setter
    def plot_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Set the plot plugin fallback configuration."""
        self.plugin_manager.plot_plugin_fallbacks = value

    @property
    def _explanation_plugin_overrides(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        return self.plugin_manager.explanation_plugin_overrides

    @_explanation_plugin_overrides.setter
    def _explanation_plugin_overrides(self, value: Dict[str, Any]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.explanation_plugin_overrides = value

    @property
    def _interval_plugin_override(self) -> Any:
        """Delegate to PluginManager."""
        return self.plugin_manager.interval_plugin_override

    @_interval_plugin_override.setter
    def _interval_plugin_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.interval_plugin_override = value

    @property
    def _fast_interval_plugin_override(self) -> Any:
        """Delegate to PluginManager."""
        return self.plugin_manager.fast_interval_plugin_override

    @_fast_interval_plugin_override.setter
    def _fast_interval_plugin_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.fast_interval_plugin_override = value

    @property
    def _plot_style_override(self) -> Any:
        """Delegate to PluginManager."""
        return self.plugin_manager.plot_style_override

    @_plot_style_override.setter
    def _plot_style_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.plot_style_override = value

    @property
    def _explanation_plugin_instances(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        return self.plugin_manager.explanation_plugin_instances

    @_explanation_plugin_instances.setter
    def _explanation_plugin_instances(self, value: Dict[str, Any]) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.explanation_plugin_instances = value

    # Public aliases to replace test access of private members (safe one-line delegations)
    @property
    def plugin_manager(self) -> PluginManager:
        """Public accessor for the active PluginManager."""
        return self.require_plugin_manager()

    @plugin_manager.setter
    def plugin_manager(self, value: Any) -> None:
        """Set the plugin manager for this explainer."""
        self._plugin_manager = value

    @plugin_manager.deleter
    def plugin_manager(self) -> None:
        """Delete the plugin manager."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager

    @property
    def interval_plugin_hints(self) -> Dict[str, Tuple[str, ...]]:
        """Public alias for `_interval_plugin_hints`.

        Tests should use this instead of accessing the private attribute.
        """
        return self._interval_plugin_hints

    @interval_plugin_hints.setter
    def interval_plugin_hints(self, value: Dict[str, Tuple[str, ...]]) -> None:
        self._interval_plugin_hints = value

    @interval_plugin_hints.deleter
    def interval_plugin_hints(self) -> None:
        if hasattr(self, "plugin_manager"):
            del self.plugin_manager.interval_plugin_hints

    @property
    def interval_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Public alias for `_interval_plugin_fallbacks`."""
        return self._interval_plugin_fallbacks

    @interval_plugin_fallbacks.setter
    def interval_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        self._interval_plugin_fallbacks = value

    @interval_plugin_fallbacks.deleter
    def interval_plugin_fallbacks(self) -> None:
        if hasattr(self, "plugin_manager"):
            del self.plugin_manager.interval_plugin_fallbacks

    @property
    def explanation_plugin_overrides(self) -> Dict[str, Any]:
        """Public alias for `_explanation_plugin_overrides`."""
        if hasattr(self, "plugin_manager"):
            return self._explanation_plugin_overrides
        return {}

    @explanation_plugin_overrides.setter
    def explanation_plugin_overrides(self, value: Dict[str, Any]) -> None:
        self._explanation_plugin_overrides = value

    @property
    def interval_plugin_override(self) -> Any:
        """Public alias for `_interval_plugin_override`."""
        if hasattr(self, "plugin_manager"):
            return self._interval_plugin_override
        return None

    @interval_plugin_override.setter
    def interval_plugin_override(self, value: Any) -> None:
        if hasattr(self, "plugin_manager"):
            self._interval_plugin_override = value
        # else do nothing

    @property
    def fast_interval_plugin_override(self) -> Any:
        """Public alias for `_fast_interval_plugin_override`."""
        return self._fast_interval_plugin_override

    @fast_interval_plugin_override.setter
    def fast_interval_plugin_override(self, value: Any) -> None:
        self._fast_interval_plugin_override = value

    @property
    def plot_style_override(self) -> Any:
        """Public alias for `_plot_style_override`."""
        return self._plot_style_override

    @plot_style_override.setter
    def plot_style_override(self, value: Any) -> None:
        self._plot_style_override = value

    @property
    def interval_preferred_identifier(self) -> Dict[str, str | None]:
        """Public alias for `_interval_preferred_identifier`."""
        return self._interval_preferred_identifier

    @interval_preferred_identifier.setter
    def interval_preferred_identifier(self, value: Dict[str, str | None]) -> None:
        self._interval_preferred_identifier = value

    @interval_preferred_identifier.deleter
    def interval_preferred_identifier(self) -> None:
        """Delete the interval preferred identifier."""
        del self._interval_preferred_identifier

    @property
    def telemetry_interval_sources(self) -> Dict[str, str | None]:
        """Public alias for `_telemetry_interval_sources`."""
        return self._telemetry_interval_sources

    @telemetry_interval_sources.setter
    def telemetry_interval_sources(self, value: Dict[str, str | None]) -> None:
        self._telemetry_interval_sources = value

    @telemetry_interval_sources.deleter
    def telemetry_interval_sources(self) -> None:
        """Delete the telemetry interval sources."""
        del self._telemetry_interval_sources

    @property
    def interval_plugin_identifiers(self) -> Dict[str, str | None]:
        """Public alias for `_interval_plugin_identifiers`."""
        return self._interval_plugin_identifiers

    @interval_plugin_identifiers.setter
    def interval_plugin_identifiers(self, value: Dict[str, str | None]) -> None:
        self._interval_plugin_identifiers = value

    @property
    def preprocessor_metadata(self) -> Any:
        """Public alias for `_preprocessor_metadata`."""
        return self._preprocessor_metadata

    @preprocessor_metadata.setter
    def preprocessor_metadata(self, value: Any) -> None:
        self._preprocessor_metadata = value

    @property
    def feature_names_internal(self) -> Any:
        """Public alias for `_feature_names`."""
        return self._feature_names

    @feature_names_internal.setter
    def feature_names_internal(self, value: Any) -> None:
        self._feature_names = value

    @property
    def perf_parallel(self) -> bool:
        """Public alias for `_perf_parallel`."""
        return self._perf_parallel

    @perf_parallel.setter
    def perf_parallel(self, value: bool) -> None:
        self._perf_parallel = value

    @property
    def get_sigma_test(self) -> bool:
        """Public alias for `_get_sigma_test`."""
        return self._get_sigma_test

    @get_sigma_test.setter
    def get_sigma_test(self, value: bool) -> None:
        self._get_sigma_test = value

    def initialize_interval_learner_for_fast_explainer(self, *args, **kwargs) -> Any:
        """Public alias for internal interval learner initialization."""
        return self._CalibratedExplainer__initialize_interval_learner_for_fast_explainer(
            *args, **kwargs
        )

    @interval_plugin_identifiers.deleter
    def interval_plugin_identifiers(self) -> None:
        """Delete the interval plugin identifiers."""
        del self._interval_plugin_identifiers

    @property
    def interval_context_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Public alias for `_interval_context_metadata`."""
        return self._interval_context_metadata

    @interval_context_metadata.setter
    def interval_context_metadata(self, value: Dict[str, Dict[str, Any]]) -> None:
        self._interval_context_metadata = value

    @interval_context_metadata.deleter
    def interval_context_metadata(self) -> None:
        """Delete the interval context metadata."""
        del self._interval_context_metadata

    @property
    def bridge_monitors(self) -> Dict[str, Any]:
        """Public alias for `_bridge_monitors`."""
        return self._bridge_monitors

    @bridge_monitors.setter
    def bridge_monitors(self, value: Dict[str, Any]) -> None:
        """Set the bridge monitors."""
        self.require_plugin_manager().bridge_monitors = value

    @property
    def explanation_plugin_instances(self) -> Dict[str, Any]:
        """Public alias for `_explanation_plugin_instances`."""
        return self._explanation_plugin_instances

    @explanation_plugin_instances.setter
    def explanation_plugin_instances(self, value: Dict[str, Any]) -> None:
        """Set the explanation plugin instances."""
        self.require_plugin_manager().explanation_plugin_instances = value

    @property
    def pyproject_explanations(self) -> Dict[str, Any] | None:
        """Public alias for `_pyproject_explanations`."""
        return self._pyproject_explanations

    @pyproject_explanations.setter
    def pyproject_explanations(self, value: Dict[str, Any] | None) -> None:
        self._pyproject_explanations = value

    @property
    def pyproject_intervals(self) -> Dict[str, Any] | None:
        """Public alias for `_pyproject_intervals`."""
        return self._pyproject_intervals

    @pyproject_intervals.setter
    def pyproject_intervals(self, value: Dict[str, Any] | None) -> None:
        self._pyproject_intervals = value

    @property
    def pyproject_plots(self) -> Dict[str, Any] | None:
        """Public alias for `_pyproject_plots`."""
        return self._pyproject_plots

    @pyproject_plots.setter
    def pyproject_plots(self, value: Dict[str, Any] | None) -> None:
        self._pyproject_plots = value

    @property
    def lime_helper(self) -> Any:
        """Public alias for `_lime_helper`."""
        return self._lime_helper

    @lime_helper.setter
    def lime_helper(self, value: Any) -> None:
        """Set the LIME helper."""
        self._lime_helper = value

    @lime_helper.deleter
    def lime_helper(self) -> None:
        """Delete the LIME helper."""
        if hasattr(self, "_lime_helper"):
            del self._lime_helper

    @property
    def shap_helper(self) -> Any:
        """Public alias for `_shap_helper`."""
        return self._shap_helper

    @shap_helper.setter
    def shap_helper(self, value: Any) -> None:
        """Set the SHAP helper."""
        self._shap_helper = value

    @shap_helper.deleter
    def shap_helper(self) -> None:
        """Delete the SHAP helper."""
        if hasattr(self, "_shap_helper"):
            del self._shap_helper

    @property
    def initialized(self) -> bool:
        """Return True if the explainer is initialized."""
        return getattr(self, "_CalibratedExplainer__initialized", False)

    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set the initialization state of the explainer."""
        self.__initialized = value

    @property
    def is_initialized(self) -> bool:
        """Public check for whether the explainer has been initialized.

        .. deprecated:: 0.10.1
            Use :attr:`initialized` instead.
        """
        return self.initialized

    @property
    def last_explanation_mode(self) -> str | None:
        """Return the mode of the last generated explanation."""
        return self._last_explanation_mode

    @last_explanation_mode.setter
    def last_explanation_mode(self, value: str | None) -> None:
        """Set the mode of the last generated explanation."""
        self._last_explanation_mode = value

    @property
    def feature_filter_per_instance_ignore(self) -> Any:
        """Return the per-instance feature filter ignore list."""
        return getattr(self, "_feature_filter_per_instance_ignore", None)

    @feature_filter_per_instance_ignore.setter
    def feature_filter_per_instance_ignore(self, value: Any) -> None:
        """Set the per-instance feature filter ignore list."""
        self._feature_filter_per_instance_ignore = value

    @feature_filter_per_instance_ignore.deleter
    def feature_filter_per_instance_ignore(self) -> None:
        """Delete the per-instance feature filter ignore list."""
        if hasattr(self, "_feature_filter_per_instance_ignore"):
            delattr(self, "_feature_filter_per_instance_ignore")

    @property
    def parallel_executor(self) -> Any:
        """Return the active parallel executor."""
        return getattr(self, "_perf_parallel", None)

    @parallel_executor.setter
    def parallel_executor(self, value: Any) -> None:
        """Set the active parallel executor."""
        self._perf_parallel = value

    @property
    def feature_filter_config(self) -> Any:
        """Return the feature filter configuration."""
        return getattr(self, "_feature_filter_config", None)

    @feature_filter_config.setter
    def feature_filter_config(self, value: Any) -> None:
        """Set the feature filter configuration."""
        self._feature_filter_config = value

    @property
    def predict_bridge(self) -> Any:
        """Return the prediction bridge."""
        return getattr(self, "_predict_bridge", None)

    @predict_bridge.setter
    def predict_bridge(self, value: Any) -> None:
        """Set the prediction bridge."""
        self._predict_bridge = value

    @property
    def categorical_value_counts_cache(self) -> Any:
        """Return the categorical value counts cache."""
        return getattr(self, "_categorical_value_counts_cache", None)

    @categorical_value_counts_cache.setter
    def categorical_value_counts_cache(self, value: Any) -> None:
        """Set the categorical value counts cache."""
        self._categorical_value_counts_cache = value

    @property
    def numeric_sorted_cache(self) -> Any:
        """Return the numeric sorted cache."""
        return getattr(self, "_numeric_sorted_cache", None)

    @numeric_sorted_cache.setter
    def numeric_sorted_cache(self, value: Any) -> None:
        """Set the numeric sorted cache."""
        self._numeric_sorted_cache = value

    @property
    def calibration_summary_shape(self) -> Any:
        """Return the calibration summary shape."""
        return getattr(self, "_calibration_summary_shape", None)

    @calibration_summary_shape.setter
    def calibration_summary_shape(self, value: Any) -> None:
        """Set the calibration summary shape."""
        self._calibration_summary_shape = value

    def enable_fast_mode(self) -> None:
        """Enable fast explanation mode.

        This initializes the interval learner for fast explanations if not already done.
        """
        if not self.is_fast():
            try:
                self._CalibratedExplainer__fast = True
                # Prefer calling the public method name so unit tests that patch
                # `initialize_interval_learner_for_fast_explainer` observe the
                # raised exception. Fall back to the name-mangled implementation
                # if the public alias is absent.
                init_fn = getattr(
                    self, "initialize_interval_learner_for_fast_explainer", None
                )
                if callable(init_fn):
                    init_fn()
                else:
                    self._CalibratedExplainer__initialize_interval_learner_for_fast_explainer()
            except Exception:  # adr002_allow
                self._CalibratedExplainer__fast = False
                raise

    @property
    def _bridge_monitors(self) -> Dict[str, Any]:
        """Expose bridge monitor registry managed by PluginManager."""
        return self.require_plugin_manager().bridge_monitors

    @property
    def _pyproject_explanations(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        return self.plugin_manager.pyproject_explanations

    @_pyproject_explanations.setter
    def _pyproject_explanations(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.pyproject_explanations = value

    @property
    def _pyproject_intervals(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        return self.plugin_manager.pyproject_intervals

    @_pyproject_intervals.setter
    def _pyproject_intervals(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.pyproject_intervals = value

    @property
    def _pyproject_plots(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        return self.plugin_manager.pyproject_plots

    @_pyproject_plots.setter
    def _pyproject_plots(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        self.plugin_manager.pyproject_plots = value

    @property
    def runtime_telemetry(self) -> Mapping[str, Any]:
        """Return the most recent telemetry payload reported by the explainer."""
        return dict(self.plugin_manager.last_telemetry)

    @property
    def preprocessor_metadata(self) -> Dict[str, Any] | None:
        """Return the telemetry-safe preprocessing snapshot if available."""
        if self._preprocessor_metadata is None:
            return None
        return dict(self._preprocessor_metadata)

    def set_preprocessor_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        """Update the stored preprocessing metadata snapshot."""
        if metadata is None:
            self._preprocessor_metadata = None
        else:
            self._preprocessor_metadata = dict(metadata)

    @property
    def x_cal(self):
        """Get the calibration input data.

        Returns
        -------
        array-like
            The calibration input data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        return CalibrationState.get_x_cal(self)

    @x_cal.setter
    def x_cal(self, value):
        """Set the calibration input data.

        Parameters
        ----------
        value : array-like of shape (n_samples, n_features)
            The new calibration input data.

        Raises
        ------
        ValueError
            If the number of features in value does not match the existing calibration data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.set_x_cal(self, value)

    @property
    def y_cal(self):
        """Get the calibration target data.

        Returns
        -------
        array-like
            The calibration target data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        return CalibrationState.get_y_cal(self)

    @y_cal.setter
    def y_cal(self, value):
        """Set the calibration target data.

        Parameters
        ----------
        value : array-like of shape (n_samples,)
            The new calibration target data.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.set_y_cal(self, value)

    def append_cal(self, x, y):
        """Append new calibration data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The new calibration input data to append.
        y : array-like of shape (n_samples,)
            The new calibration target data to append.
        """
        from ..calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.append_calibration(self, x, y)

    def _invalidate_calibration_summaries(self) -> None:
        """Drop cached calibration summaries used during explanation.

        Delegates to the calibration.summaries module which manages the cache.
        """
        from ..calibration.summaries import (  # pylint: disable=import-outside-toplevel
            invalidate_calibration_summaries as _invalidate,
        )

        _invalidate(self)

    def get_calibration_summaries(
        self, x_cal_np: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
        """Return cached categorical counts and sorted numeric calibration values.

        Delegates to the calibration.summaries module which manages caching of
        statistical summaries used during explanation generation.
        """
        from ..calibration.summaries import (  # pylint: disable=import-outside-toplevel
            get_calibration_summaries as _get,
        )

        return _get(self, x_cal_np)

    @property
    def num_features(self):
        """Get the number of features in the calibration data.

        Returns
        -------
        int
            The number of features in the calibration data. For dictionary input,
            returns the number of keys. For array input, returns the number of columns.
        """
        return (
            len(self._X_cal[0].keys())
            if isinstance(self._X_cal[0], dict)
            else len(self._X_cal[0, :])
        )

    @property
    def feature_names(self):
        """Get the feature names.

        Returns
        -------
        list
            The list of feature names. If no feature names were provided during initialization,
            returns None.
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        """Set the feature names.

        Parameters
        ----------
        value : list
            The list of feature names.
        """
        self._feature_names = list(value) if value is not None else None

    @property
    def interval_learner(self) -> Any:
        """Access the interval learner managed by the prediction orchestrator.

        Returns
        -------
        Any
            The interval calibrator (e.g., VennAbers, IntervalRegressor, or list for fast mode).

        Notes
        -----
        This is a backward-compatible property that delegates to the interval registry
        managed by the PredictionOrchestrator. See ADR-001.
        """
        return self.prediction_orchestrator.interval_registry.interval_learner

    @interval_learner.setter
    def interval_learner(self, value: Any) -> None:
        """Set the interval learner through the prediction orchestrator's registry.

        Parameters
        ----------
        value : Any
            The interval calibrator to set (e.g., VennAbers, IntervalRegressor).

        Notes
        -----
        This is a backward-compatible setter that delegates to the interval registry
        managed by the PredictionOrchestrator.
        """
        self.prediction_orchestrator.interval_registry.interval_learner = value

    def _get_sigma_test(self, x: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances.

        Parameters
        ----------
        x : np.ndarray
            Test instances for which to estimate difficulty.

        Returns
        -------
        np.ndarray
            Difficulty estimates (sigma values) for each test instance.

        Notes
        -----
        This is a backward-compatible method that delegates to the interval registry
        managed by the PredictionOrchestrator. See ADR-001.
        """
        return self.prediction_orchestrator.interval_registry.get_sigma_test(x)

    def get_sigma_test(self, x: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances.

        Parameters
        ----------
        x : np.ndarray
            Test instances for which to estimate difficulty.

        Returns
        -------
        np.ndarray
            Difficulty estimates (sigma values) for each test instance.
        """
        return self._get_sigma_test(x)

    def _CalibratedExplainer__initialize_interval_learner_for_fast_explainer(self) -> None:  # noqa: N802
        """Backward-compatible wrapper for fast-mode interval learner initialization.

        Notes
        -----
        This method delegates to the interval registry. It is kept for backward
        compatibility with the external fast_explanations plugin and other
        production code that calls this private method.

        See ADR-001.
        """
        self.prediction_orchestrator.interval_registry.initialize_for_fast_explainer()

    def reinitialize(self, learner, xs=None, ys=None, bins=None):
        """Reinitialize the explainer with a new learner.

        This is useful when the learner is updated or retrained and the explainer needs to be reinitialized.

        Parameters
        ----------
        learner : predictive learner
            A predictive learner that can be used to predict the target variable. The learner must be fitted and have a predict_proba method (for classification) or a predict method (for regression).
        xs : array-like, optional
            New calibration input data to append
        ys : array-like, optional
            New calibration target data to append

        Returns
        -------
        :class:`.CalibratedExplainer`
            A :class:`.CalibratedExplainer` object that can be used to explain predictions from a predictive learner.
        """
        self.__initialized = False
        check_is_fitted(learner)
        self.learner = learner
        if xs is not None and ys is not None:
            self.append_cal(xs, ys)
            if bins is not None:
                if self.bins is None:
                    raise ValidationError("Cannot mix calibration instances with and without bins.")
                if len(bins) != len(ys):
                    raise DataShapeError(
                        "The length of bins must match the number of added instances."
                    )
                self.bins = np.concatenate((self.bins, bins)) if self.bins is not None else bins
            # update interval learner via helper
            from ..calibration.interval_learner import update_interval_learner as _upd_il

            _upd_il(self, xs, ys, bins=bins)
        else:
            from ..calibration.interval_learner import initialize_interval_learner as _init_il

            _init_il(self)
        self.__initialized = True

    def __repr__(self):
        """Return the string representation of the CalibratedExplainer."""
        # pylint: disable=line-too-long
        disp_str = f"CalibratedExplainer(mode={self.mode}{', conditional=True' if self.bins is not None else ''}{f', discretizer={self.discretizer}' if self.discretizer is not None else ''}, learner={self.learner}{f', difficulty_estimator={self.difficulty_estimator})' if self.mode == 'regression' else ')'}"
        if self.verbose:
            disp_str += f"\n\tinit_time={self.init_time}"
            if self.latest_explanation is not None:
                disp_str += f"\n\ttotal_explain_time={self.latest_explanation.total_explain_time}"
            disp_str += f"\n\tsample_percentiles={self.sample_percentiles}\
                        \n\tseed={self.seed}\
                        \n\tverbose={self.verbose}"
            if self.feature_names is not None:
                disp_str += f"\n\tfeature_names={self.feature_names}"
            if self.categorical_features is not None:
                disp_str += f"\n\tcategorical_features={self.categorical_features}"
            if self.categorical_labels is not None:
                disp_str += f"\n\tcategorical_labels={self.categorical_labels}"
            if self.class_labels is not None:
                disp_str += f"\n\tclass_labels={self.class_labels}"
        return disp_str

    def obtain_interval_calibrator(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> Tuple[Any, str | None]:
        """Return the interval calibrator from the prediction orchestrator."""
        return self.prediction_orchestrator.obtain_interval_calibrator(fast=fast, metadata=metadata)

    def predict_calibrated(
        self,
        x: Any,
        threshold: float | None = None,
        low_high_percentiles: tuple[float, float] = (5, 95),
        classes: Any = None,
        bins: Any = None,
        feature: int | None = None,
        interval_summary: Any | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Predict calibrated values and intervals."""
        return self._predict(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            feature=feature,
            interval_summary=interval_summary,
            **kwargs,
        )

    def preload_lime(self, x_cal=None):
        """Materialize LIME explainer artifacts.

        Parameters
        ----------
        x_cal : array-like, optional
            Calibration data to use for preloading.

        Returns
        -------
        LimePipeline
            The LIME pipeline instance.
        """
        return self._lime_helper.preload(x_cal=x_cal)

    def preload_shap(self, num_test: int | None = None):
        """Materialize SHAP explainer artifacts.

        Parameters
        ----------
        num_test : int, optional
            Number of test samples to use for preloading.

        Returns
        -------
        tuple
            The SHAP explainer and reference explanation.
        """
        return self._shap_helper.preload(num_test=num_test)

    def _predict(self, *args, **kwargs) -> Any:
        """Delegate to predict_internal."""
        return self.predict_internal(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Any:
        """Public alias for _predict for testing."""
        return self._predict(*args, **kwargs)

    def predict_internal(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        classes=None,
        bins=None,
        feature=None,
        interval_summary=None,
        **kwargs,
    ):
        """Cache-aware prediction wrapper. Delegated to PredictionOrchestrator."""
        # Internal skip flag: when True, bypass reject orchestration. This is
        # used by internal callers (e.g., RejectOrchestrator) to obtain a raw
        # prediction without re-entering the reject flow.
        if "_ce_skip_reject" in kwargs:
            # consume internal-only flag and proceed without reject handling
            kwargs.pop("_ce_skip_reject")
            orchestrator = self.prediction_orchestrator
            if hasattr(orchestrator, "predict_internal"):
                return orchestrator.predict_internal(
                    x,
                    threshold=threshold,
                    low_high_percentiles=low_high_percentiles,
                    classes=classes,
                    bins=bins,
                    feature=feature,
                    interval_summary=interval_summary,
                    **kwargs,
                )
            return orchestrator.predict(
                x,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                classes=classes,
                bins=bins,
                feature=feature,
                interval_summary=interval_summary,
                **kwargs,
            )

        # Support per-call reject policy selection. When a non-NONE policy is
        # selected, delegate to the RejectOrchestrator and return a RejectResult
        # envelope. Per-call policy overrides the explainer-level default.
        from .reject.policy import RejectPolicy

        # Pop per-call policy if provided so downstream orchestrators don't
        # receive unexpected kwargs. Only an explicit per-call policy will
        # trigger reject orchestration here. Explainer-level defaults are
        # handled at the top-level explanation APIs (e.g., `explain_factual`).
        per_call_policy = None
        if "reject_policy" in kwargs:
            per_call_policy = kwargs.pop("reject_policy")

        if per_call_policy is None:
            effective_policy = getattr(self, "default_reject_policy", RejectPolicy.NONE)
        else:
            try:
                effective_policy = RejectPolicy(per_call_policy)
            except Exception:
                effective_policy = RejectPolicy.NONE

        if effective_policy is not None and effective_policy is not RejectPolicy.NONE:
            # Ensure reject orchestrator is available (implicit enable)
            try:
                _ = self.reject_orchestrator
            except Exception:
                try:
                    self.plugin_manager.initialize_orchestrators()
                except Exception:
                    pass

            orchestrator = self.prediction_orchestrator

            def _predict_fn(x_subset, **kw):
                # Call the lower-level orchestrator implementation to avoid
                # recursing back into CalibratedExplainer.predict.
                if hasattr(orchestrator, "predict_internal"):
                    return orchestrator.predict_internal(x_subset, **kw)
                return orchestrator.predict(x_subset, **kw)

            return self.reject_orchestrator.apply_policy(
                effective_policy,
                x,
                explain_fn=_predict_fn,
                bins=bins,
                interval_summary=interval_summary,
                **kwargs,
            )
        # Delegate directly to the orchestrator implementation method so
        # tests that inject a minimal/mock PluginManager (with a
        # `_prediction_orchestrator` stub) can set `_predict_impl.return_value`.
        # The public `.predict` may be a MagicMock in tests; calling the
        # implementation ensures the intended behavior is exercised.
        orchestrator = self.prediction_orchestrator
        if hasattr(orchestrator, "predict_internal"):
            return orchestrator.predict_internal(
                x,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                classes=classes,
                bins=bins,
                feature=feature,
                interval_summary=interval_summary,
                **kwargs,
            )
        return orchestrator.predict(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            feature=feature,
            interval_summary=interval_summary,
            **kwargs,
        )

    def explain_factual(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        _use_plugin: bool = True,
        **kwargs,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data with the discretizer automatically assigned for factual explanations.

        This is a thin delegator that sets up the appropriate discretizer and delegates to the orchestrator.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like, default=None
            Values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        **kwargs : dict
            Additional arguments passed to the explanation orchestrator.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FactualExplanation` for each instance.
        """
        if bins is None and self.is_mondrian():
            bins = self.bins
        # Thin delegator that sets discretizer and delegates to orchestrator
        discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        ctx = self._perf_parallel if self._perf_parallel is not None else contextlib.nullcontext()
        with ctx:
            reject_policy = kwargs.pop("reject_policy", None)
            invoke_kwargs = {
                "x": x,
                "threshold": threshold,
                "low_high_percentiles": low_high_percentiles,
                "bins": bins,
                "features_to_ignore": features_to_ignore,
                "discretizer": discretizer,
                "_use_plugin": _use_plugin,
                **kwargs,
            }
            if reject_policy is not None:
                invoke_kwargs["reject_policy"] = reject_policy
            return self.explanation_orchestrator.invoke_factual(**invoke_kwargs)

    def explore_alternatives(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        _use_plugin: bool = True,
        **kwargs,
    ) -> AlternativeExplanations:
        """Create a :class:`.AlternativeExplanations` object for the test data with the discretizer automatically assigned for alternative explanations.

        This is a thin delegator that sets up the appropriate discretizer and delegates to the orchestrator.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict.
        threshold : float, int or array-like, default=None
            Values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        **kwargs : dict
            Additional arguments passed to the explanation orchestrator.

        Returns
        -------
        AlternativeExplanations : :class:`.AlternativeExplanations`
        Notes
        -----
        The `explore_alternatives` will eventually be used instead of the `explain_counterfactual` method.
        """
        if bins is None and self.is_mondrian():
            bins = self.bins
        # Thin delegator that sets discretizer and delegates to orchestrator
        discretizer = "regressor" if "regression" in self.mode else "entropy"
        ctx = self._perf_parallel if self._perf_parallel is not None else contextlib.nullcontext()
        with ctx:
            reject_policy = kwargs.pop("reject_policy", None)
            invoke_kwargs = {
                "x": x,
                "threshold": threshold,
                "low_high_percentiles": low_high_percentiles,
                "bins": bins,
                "features_to_ignore": features_to_ignore,
                "discretizer": discretizer,
                "_use_plugin": _use_plugin,
                **kwargs,
            }
            if reject_policy is not None:
                invoke_kwargs["reject_policy"] = reject_policy
            return self.explanation_orchestrator.invoke_alternative(**invoke_kwargs)  # type: ignore[return-value]

    def __call__(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        reject_policy: Any | None = None,
        _use_plugin: bool = True,
        _skip_instance_parallel: bool = False,
    ) -> CalibratedExplanations:
        """Call self as a function to create a :class:`.CalibratedExplanations` object for the test data with the already assigned discretizer.

        Since v0.4.0, this method is equivalent to the `_explain` method.
        """
        call_kwargs: dict[str, Any] = {
            "_use_plugin": _use_plugin,
            "_skip_instance_parallel": _skip_instance_parallel,
        }
        if reject_policy is not None:
            call_kwargs["reject_policy"] = reject_policy

        return self._explain(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
            **call_kwargs,
        )

    def _explain(self, *args, **kwargs) -> CalibratedExplanations:
        """Generate explanations for test instances by analyzing feature effects.

        This is an internal orchestration primitive that delegates to the explanation orchestrator.
        It is NOT part of the public API and should not be called directly.

        This method:
        1. Makes predictions on original test instances
        2. Creates perturbed versions by varying feature values
        3. Analyzes how predictions change with feature perturbations
        4. Generates feature importance weights and prediction intervals

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A :class:`.CalibratedExplanations` containing one :class:`.CalibratedExplanation` for each instance.

        See Also
        --------
        :meth:`.CalibratedExplainer.explain_factual` : Refer to the documentation for `explain_factual` for more details.
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the documentation for `explore_alternatives` for more details.
        """
        # Delegate the args to the actual implementation
        return self._explain_impl(*args, **kwargs)

    def _explain_impl(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        reject_policy: Any | None = None,
        _use_plugin: bool = True,
        _skip_instance_parallel: bool = False,
    ) -> CalibratedExplanations:
        if bins is None and self.is_mondrian():
            bins = self.bins
        # Thin delegator to orchestrator
        if _use_plugin:
            mode = self.infer_explanation_mode()
            invoke_kwargs: dict[str, Any] = {
                "extras": {"mode": mode, "_skip_instance_parallel": _skip_instance_parallel}
            }
            if reject_policy is not None:
                invoke_kwargs["reject_policy"] = reject_policy
            return self.explanation_orchestrator.invoke(
                mode,
                x,
                threshold,
                low_high_percentiles,
                bins,
                features_to_ignore,
                **invoke_kwargs,
            )

        # Legacy path for backward compatibility and testing
        from .explain import legacy_explain  # pylint: disable=import-outside-toplevel

        return legacy_explain(
            self,
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            bins=bins,
            features_to_ignore=features_to_ignore,
        )

    # NOTE: Instance- and feature-parallel helpers have been moved into the
    # plugin-based implementation under `core.explain.*`. The legacy helper
    # methods were intentionally removed to centralize parallel execution in
    # the plugin modules. Tests should exercise the plugin classes
    # (e.g. InstanceParallelExplainExecutor, FeatureParallelExplainExecutor,
    # SequentialExplainExecutor) rather than calling these private helpers.

    # NOTE: merge_feature_result functionality has been moved to
    # `calibrated_explanations.core.explain._helpers.merge_feature_result`.
    # Plugins and explain code should call that free-function directly.

    # NOTE: Thin wrapper methods (_slice_threshold, _slice_bins, _validate_and_prepare_input,
    # _initialize_explanation, _compute_weight_delta, _discretize) have been removed.
    # Callers should import these directly from core.explain submodules:
    # - core.explain._helpers: slice_threshold, slice_bins, validate_and_prepare_input
    # - core.explain._computation: initialize_explanation, discretize
    # - core.explain._helpers: compute_weight_delta

    def explain_fast(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        *,
        reject_policy: Any | None = None,
        _use_plugin: bool = True,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in x.
        RuntimeError: Fast explanations are only possible if the explainer is a Fast Calibrated Explainer.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FastExplanation` for each instance.
        """
        if bins is None and self.is_mondrian():
            bins = self.bins
        if _use_plugin:
            return self._invoke_explanation_plugin(
                "fast",
                x,
                threshold,
                low_high_percentiles,
                bins,
                tuple(self.features_to_ignore),
                extras={"mode": "fast"},
                reject_policy=reject_policy,
            )

        # Delegate to external plugin pipeline for non-plugin path
        # pylint: disable-next=import-outside-toplevel
        import sys
        from pathlib import Path

        # Ensure the repository root is in the path
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from external_plugins.fast_explanations.pipeline import FastExplanationPipeline

        pipeline = FastExplanationPipeline(self)
        return pipeline.explain(x, threshold, low_high_percentiles, bins)

    # feature-merge and feature-parallel logic moved to plugin helpers

    def explain_lime(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
    ) -> CalibratedExplanations:
        """Create a :class:`.CalibratedExplanations` object for the test data.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in x.
        RuntimeError: Fast explanations are only possible if the explainer is a Fast Calibrated Explainer.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FastExplanation` for each instance.
        """
        if bins is None and self.is_mondrian():
            bins = self.bins
        # Delegate to external plugin pipeline
        # pylint: disable-next=import-outside-toplevel
        from pathlib import Path

        # Ensure the repository root is in the path
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from external_plugins.integrations.lime_pipeline import LimePipeline

        pipeline = LimePipeline(self)
        return pipeline.explain(x, threshold, low_high_percentiles, bins)

    def explain_shap(self, x, **kwargs):
        """Create SHAP-based explanations for the test data.

        Delegates to the external SHAP integration pipeline.

        Parameters
        ----------
        x : array-like
            A set with n_samples of test objects to explain.
        **kwargs
            Additional keyword arguments passed through to SHAP.

        Returns
        -------
        Any
            SHAP explanation object containing feature importance values.

        Raises
        ------
        ConfigurationError
            If SHAP is not properly installed or configured.
        """
        # Delegate to external plugin pipeline
        # pylint: disable-next=import-outside-toplevel
        from pathlib import Path

        # Ensure the repository root is in the path
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from external_plugins.integrations.shap_pipeline import ShapPipeline

        pipeline = ShapPipeline(self)
        return pipeline.explain(x, **kwargs)

    def is_lime_enabled(self, is_enabled: bool | None = None) -> bool:
        """Return or set the LIME helper enabled state."""
        if is_enabled is None:
            return self._lime_helper.is_enabled()
        self._lime_helper.set_enabled(bool(is_enabled))
        return self._lime_helper.is_enabled()

    def is_shap_enabled(self, is_enabled: bool | None = None) -> bool:
        """Return or set the SHAP helper enabled state."""
        if is_enabled is None:
            return self._shap_helper.is_enabled()
        self._shap_helper.set_enabled(bool(is_enabled))
        return self._shap_helper.is_enabled()

    def is_multiclass(self) -> bool:
        """Test if it is a multiclass problem.

        Returns
        -------
        bool
            True if multiclass (num_classes > 2).
        """
        return self.num_classes > 2

    def is_fast(self) -> bool:
        """Test if the explainer uses fast mode.

        Returns
        -------
        bool
            True if fast mode is enabled.
        """
        return self.__fast

    def is_mondrian(self) -> bool:
        """Test if Mondrian (per-bin) calibration is enabled.

        Returns
        -------
        bool
            True if bins are configured, indicating Mondrian calibration.
        """
        return self.bins is not None

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """Apply the discretizer to input data.

        Parameters
        ----------
        data : np.ndarray
            The data to discretize.

        Returns
        -------
        np.ndarray
            The discretized data.
        """
        from .explain import discretize as _discretize_func  # pylint: disable=import-outside-toplevel

        return _discretize_func(self, data)

    def rule_boundaries(self, instances, perturbed_instances=None):
        """Extract the rule boundaries for a set of instances.

        Parameters
        ----------
        instances : array-like
            The instances to extract boundaries for.
        perturbed_instances : array-like, optional
            Discretized versions of instances. Defaults to None.

        Returns
        -------
        array-like
            Min and max values for each feature for each instance.
        """
        from .explain import rule_boundaries as _rule_boundaries  # pylint: disable=import-outside-toplevel

        return _rule_boundaries(self, instances, perturbed_instances)

    def set_difficulty_estimator(self, difficulty_estimator, initialize=True) -> None:
        """Assign or update the difficulty estimator.

        If initialized to a difficulty estimator, the explainer can be used to reject explanations that are deemed too difficult.

        Parameters
        ----------
        difficulty_estimator : :class:`crepes.extras.DifficultyEstimator` or None):
            A :class:`crepes.extras.DifficultyEstimator` object from the crepes package. To remove the :class:`crepes.extras.DifficultyEstimator`, set to None.
        initialize (bool, optional):
            If true, then the interval learner is initialized once done. Defaults to True.
        """
        from .difficulty_estimator_helpers import (  # pylint: disable=import-outside-toplevel
            validate_difficulty_estimator,
        )

        validate_difficulty_estimator(difficulty_estimator)
        self.__initialized = False
        self.difficulty_estimator = difficulty_estimator
        if initialize:
            self.prediction_orchestrator.interval_registry.initialize()  # type: ignore[attr-defined]

    def set_mode(self, mode, initialize=True) -> None:
        """Assign the mode of the explainer. The mode can be either 'classification' or 'regression'.

        Parameters
        ----------
            mode (str): The mode can be either 'classification' or 'regression'.
            initialize (bool, optional): If true, then the interval learner is initialized once done. Defaults to True.

        Raises
        ------
            ValueError: The mode can be either 'classification' or 'regression'.
        """
        self.__initialized = False
        if mode == "classification":
            # assert 'predict_proba' in dir(self.learner), "The learner must have a predict_proba method."
            self.num_classes = len(np.unique(self.y_cal))
        elif mode == "regression":
            # assert 'predict' in dir(self.learner), "The learner must have a predict method."
            self.num_classes = 0
        else:
            raise ValidationError("The mode must be either 'classification' or 'regression'.")
        self.mode = mode
        if initialize:
            self.prediction_orchestrator.interval_registry.initialize()  # type: ignore[attr-defined]

    def initialize_reject_learner(self, calibration_set=None, threshold=None):
        """Initialize the reject learner with a threshold value.

        The reject learner is a :class:`crepes.base.ConformalClassifier`
        that is trained on the calibration data. The reject learner is used to determine whether a test
        instance is within the calibration data distribution. The reject learner is only available for
        classification, unless a threshold is assigned.

        Parameters
        ----------
        calibration_set : array-like, optional
            The calibration set to use. Defaults to None.
        threshold : float, optional
            The threshold value. Defaults to None.
        """
        return self.reject_orchestrator.initialize_reject_learner(
            calibration_set=calibration_set, threshold=threshold
        )

    def predict_reject(self, x, bins=None, confidence=0.95):
        """Predict whether to reject the explanations for the test data.

        Use conformal classifier to identify test instances that may be too different from calibration data.

        Parameters
        ----------
        x : array-like
            The test data.
        bins : array-like, optional
            Mondrian categories. Defaults to None.
        confidence : float, default=0.95
            The confidence level.

        Returns
        -------
        array-like
            Returns rejection decisions and error/rejection rates.
        """
        return self.reject_orchestrator.predict_reject(x, bins=bins, confidence=confidence)

    # pylint: disable=too-many-branches
    def set_discretizer(
        self,
        discretizer,
        x_cal=None,
        y_cal=None,
        features_to_ignore=None,
        *,
        condition_source: Optional[str] = None,
    ) -> None:
        """Assign the discretizer to be used.

        Parameters
        ----------
        discretizer : str or discretizer object
            The discretizer to be used.
        X_cal : array-like, optional
            The calibration data for the discretizer.
        y_cal : array-like, optional
            The calibration target data for the discretizer.
        """
        self.explanation_orchestrator.set_discretizer(
            discretizer,
            x_cal=x_cal,
            y_cal=y_cal,
            features_to_ignore=features_to_ignore,
            condition_source=condition_source,
        )

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def predict(self, x, uq_interval=False, calibrated=True, **kwargs):
        """Generate predictions for the test data.

        Parameters
        ----------
        x : array-like
            The test data.
        uq_interval : bool, default=False
            Whether to return uncertainty intervals.
        calibrated : bool, default=True
            If True, the calibrator is used for prediction. If False, the underlying learner is used for prediction.
        **kwargs : Various types, optional
            Additional parameters to customize the explanation process. Supported parameters include:

            - threshold : float, int, or array-like of shape (n_samples,), optional, default=None
                Specifies the threshold for probabilistic regression. Returns calibrated probabilities
                P(y <= threshold) for regression tasks. This parameter is ignored for classification tasks.

            - low_high_percentiles : tuple of two floats, optional, default=(5, 95)
                The lower and upper percentiles used to calculate the prediction interval for regression tasks.
                Determines the breadth of the interval based on the distribution of the predictions.
                This parameter is ignored for classification tasks and when threshold is provided.

        Raises
        ------
        RuntimeError
            If the learner has not been fitted prior to making predictions.

        Warning
            If the learner is not calibrated.

        Returns
        -------
        calibrated_prediction : float or array-like, or str
            The calibrated prediction. For regression tasks without threshold, this is the median of the
            conformal predictive system. For probabilistic regression (with threshold), this is a probability
            P(y <= threshold). For classification tasks, it is the class label with the highest calibrated probability.
        interval : tuple of floats, optional
            A tuple (low, high) representing the lower and upper bounds of the uncertainty interval. This is returned only if ``uq_interval=True``.

        Examples
        --------
        For a prediction without prediction intervals:

        .. code-block:: python

            w.predict(x)

        For a prediction with uncertainty quantification intervals:

        .. code-block:: python

            w.predict(x, uq_interval=True)

        Notes
        -----
        The `threshold` and `low_high_percentiles` parameters are only used for regression tasks.
        """
        from .prediction_helpers import (  # pylint: disable=import-outside-toplevel
            handle_uncalibrated_regression_prediction,
            handle_uncalibrated_classification_prediction,
            format_regression_prediction,
            format_classification_prediction,
        )

        # Lazy import API params functions (deferred from module level)
        from ..api.params import (
            canonicalize_kwargs,
            validate_param_combination,
            warn_on_aliases,
        )

        # emit deprecation warnings for aliases and normalize kwargs
        warn_on_aliases(kwargs)
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)
        if "interval_summary" not in kwargs or kwargs["interval_summary"] is None:
            kwargs["interval_summary"] = self.interval_summary
        else:
            kwargs["interval_summary"] = coerce_interval_summary(kwargs["interval_summary"])

        if not calibrated:
            if self.mode == "regression":
                return handle_uncalibrated_regression_prediction(
                    self.learner, x, threshold=kwargs.get("threshold"), uq_interval=uq_interval
                )
            return handle_uncalibrated_classification_prediction(
                self.learner, x, threshold=kwargs.get("threshold"), uq_interval=uq_interval
            )

        # Calibrated predictions
        if self.mode == "regression":
            predict, low, high, _ = self._predict(x, **kwargs)
            return format_regression_prediction(
                predict, low, high, threshold=kwargs.get("threshold"), uq_interval=uq_interval
            )

        # Classification
        predict, low, high, new_classes = self._predict(x, **kwargs)
        return format_classification_prediction(
            predict,
            low,
            high,
            new_classes,
            self.is_multiclass(),
            label_map=self.label_map,
            class_labels=self.class_labels,
            uq_interval=uq_interval,
        )

    def predict_proba(self, x, uq_interval=False, calibrated=True, threshold=None, **kwargs):
        """Generate probability predictions for the test data.

        This is a wrapper around the predict_proba method which is more similar to the scikit-learn predict_proba method for classification.
        As opposed to predict_proba, this method may output uncertainty intervals.

        Parameters
        ----------
        x : array-like
            The test data for which predictions are to be made. This should be in a format compatible with sklearn (e.g., numpy arrays, pandas DataFrames).
        uq_interval : bool, default=False
            If true, then the prediction interval is returned as well.
        calibrated : bool, default=True
            If True, the calibrator is used for prediction. If False, the underlying learner is used for prediction.
        threshold : float, int or array-like of shape (n_samples,), optional, default=None
            Threshold values used with regression to get probability of being below the threshold. Only applicable to regression.

        Raises
        ------
        RuntimeError
            If the learner is not fitted before predicting.

        ValueError
            If the `threshold` parameter's length does not match the number of instances in `x`, or if it is not a single constant value applicable to all instances.

        RuntimeError
            If the learner is not fitted before predicting.

        Warning
            If the learner is not calibrated.

        Returns
        -------
        calibrated probability :
            The calibrated probability of the positive class (or the predicted class for multiclass).
        (low, high) : tuple of float lists, corresponding to the lower and upper bound of each prediction interval.

        Examples
        --------
        For a prediction without uncertainty quantification intervals:

        .. code-block:: python

            w.predict_proba(x)

        For a prediction with uncertainty quantification intervals:

        .. code-block:: python

            w.predict_proba(x, uq_interval=True)

        Notes
        -----
        The `threshold` parameter is only used for regression tasks.
        """
        # strip plotting-only keys that callers may pass
        kwargs.pop("show", None)
        kwargs.pop("style_override", None)
        # Lazy import API params functions (deferred from module level)
        from ..api.params import (
            canonicalize_kwargs,
            validate_param_combination,
            warn_on_aliases,
        )

        # emit deprecation warnings for aliases and normalize kwargs
        warn_on_aliases(kwargs)
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)

        # Inject default interval_summary if not provided
        kwargs.setdefault("interval_summary", self.interval_summary)

        if not calibrated:
            if threshold is not None:
                raise ValidationError(
                    "A thresholded prediction is not possible for uncalibrated learners."
                )
            if uq_interval:
                proba = self.learner.predict_proba(x)
                if proba.shape[1] > 2:
                    return proba, (proba, proba)
                return proba, (proba[:, 1], proba[:, 1])
            return self.learner.predict_proba(x)

        # Calibrated predictions
        if self.mode == "regression":
            if is_fast_interval_collection(self.interval_learner):
                proba_1, low, high, _ = self.interval_learner[-1].predict_probability(
                    x, y_threshold=threshold, **kwargs
                )
            else:
                proba_1, low, high, _ = self.interval_learner.predict_probability(
                    x, y_threshold=threshold, **kwargs
                )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            return (proba, (low, high)) if uq_interval else proba

        # Classification - multiclass
        if self.is_multiclass():
            if is_fast_interval_collection(self.interval_learner):
                proba, low, high, _ = self.interval_learner[-1].predict_proba(
                    x, output_interval=True, **kwargs
                )
            else:
                proba, low, high, _ = self.interval_learner.predict_proba(
                    x, output_interval=True, **kwargs
                )
            return (proba, (low, high)) if uq_interval else proba

        # Classification - binary
        if is_fast_interval_collection(self.interval_learner):
            proba, low, high = self.interval_learner[-1].predict_proba(
                x, output_interval=True, **kwargs
            )
        else:
            proba, low, high = self.interval_learner.predict_proba(
                x, output_interval=True, **kwargs
            )
        return (proba, (low, high)) if uq_interval else proba

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(self, x, y=None, threshold=None, **kwargs):
        """Generate plots for the test data."""
        # Pass any style overrides along to the plotting function
        style_override = kwargs.pop("style_override", None)
        kwargs["style_override"] = style_override
        # Lazy import plotting function (deferred from module level)
        from ..plotting import plot_global

        plot_global(self, x, y=y, threshold=threshold, **kwargs)

    def calibrated_confusion_matrix(self):
        """Generate a calibrated confusion matrix.

        Generates a confusion matrix for the calibration set to provide insights about model behavior.
        The confusion matrix is only available for classification tasks. Stratified cross-validation is
        used on the calibration set to generate the confusion matrix while avoiding quadratic
        recalibration overhead.

        Returns
        -------
        array-like
            The calibrated confusion matrix.
        """
        if self.mode != "classification":
            raise ValidationError(
                "The confusion matrix is only available for classification tasks."
            )
        from .calibration_metrics import (  # pylint: disable=import-outside-toplevel
            compute_calibrated_confusion_matrix,
        )

        return compute_calibrated_confusion_matrix(
            self.x_cal, self.y_cal, self.learner, bins=self.bins
        )

    def predict_calibration(self):
        """Predict the target values for the calibration data.

        Returns
        -------
        array-like
            Predicted values for the calibration data. For models that expose a hat matrix,
            this returns updated predictions using that matrix; otherwise it uses the
            predict_function on the calibration data.
        """
        return self.predict_function(self.x_cal)

    # Public alias for testing purposes (to avoid private member access in tests)
    @property
    def fast(self) -> bool:
        return self.__fast

    @fast.setter
    def fast(self, value: bool) -> None:
        self.__fast = value

    @property
    def _fast(self) -> bool:
        return self.fast

    @_fast.setter
    def _fast(self, value: bool) -> None:
        self.fast = value

    @property
    def noise_type(self) -> str:
        return self.__noise_type

    @noise_type.setter
    def noise_type(self, value: str) -> None:
        self.__noise_type = value

    @property
    def _noise_type(self) -> str:
        return self.noise_type

    @_noise_type.setter
    def _noise_type(self, value: str) -> None:
        self.noise_type = value

    @property
    def scale_factor(self) -> float | None:
        return self.__scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float | None) -> None:
        self.__scale_factor = value

    @property
    def _scale_factor(self) -> float | None:
        return self.scale_factor

    @_scale_factor.setter
    def _scale_factor(self, value: float | None) -> None:
        self.scale_factor = value

    @property
    def severity(self) -> float | None:
        return self.__severity

    @severity.setter
    def severity(self, value: float | None) -> None:
        self.__severity = value

    @property
    def _severity(self) -> float | None:
        return self.severity

    @_severity.setter
    def _severity(self, value: float | None) -> None:
        self.severity = value


__all__ = ["CalibratedExplainer"]
