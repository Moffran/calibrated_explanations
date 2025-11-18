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

import warnings as _warnings
import logging as _logging
from collections import Counter
from time import time

import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _tomllib = None  # type: ignore[assignment]
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

from ..perf import CalibratorCache, ParallelExecutor
from ..plotting import _plot_global
from .calibration.venn_abers import VennAbers
from ..explanations import AlternativeExplanations, CalibratedExplanations
from ..integrations import LimeHelper, ShapHelper
from ..utils.discretizers import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)
from ..utils.helper import (
    check_is_fitted,
    convert_targets_to_numeric,
    immutable_array,
    safe_isinstance,
)
from ..api.params import canonicalize_kwargs, validate_param_combination, warn_on_aliases
from ..plugins import (
    ExplanationContext,
    IntervalCalibratorContext,
)
from ..plugins.builtins import LegacyPredictBridge
from ..plugins.registry import (
    ensure_builtin_plugins,
)

from .exceptions import (
    ValidationError,
    DataShapeError,
    ConfigurationError,
    NotFittedError,
)
from .explain._helpers import compute_weight_delta
from .explain.orchestrator import ExplanationOrchestrator
from .config_helpers import read_pyproject_section
from ..plugins.manager import PluginManager


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
            try:
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
            except Exception as exc:
                raise exc
            if len(x_cal) != len(y_oob):
                raise DataShapeError(
                    "The length of the out-of-bag predictions does not match the length of X_cal."
                )
            y_cal = y_oob
        self.x_cal = x_cal
        self.y_cal = y_cal

        self.set_seed(kwargs.get("seed", 42))
        self.sample_percentiles = kwargs.get("sample_percentiles", [25, 50, 75])
        self.verbose = kwargs.get("verbose", False)
        self.bins = bins

        self.__fast = kwargs.get("fast", False)
        self.__noise_type = kwargs.get("noise_type", "uniform")
        self.__scale_factor = kwargs.get("scale_factor", 5)
        self.__severity = kwargs.get("severity", 1)

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
        self._preprocess()

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
        self.latest_explanation: Optional[CalibratedExplanations] = None
        self._lime_helper = LimeHelper(self)
        self._shap_helper = ShapHelper(self)
        self.reject = kwargs.get("reject", False)

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.__set_mode(str.lower(mode), initialize=False)

        # Initialize plugin manager (SINGLE SOURCE OF TRUTH for plugin management)
        # PluginManager handles ALL plugin initialization including:
        # - Reading pyproject.toml configurations
        # - Setting up plugin overrides from kwargs
        # - Creating and initializing orchestrators
        # - Building plugin fallback chains
        self._plugin_manager = PluginManager(self)
        self._plugin_manager.initialize_from_kwargs(kwargs)
        self._plugin_manager.initialize_orchestrators()

        self._perf_cache: CalibratorCache[Any] | None = perf_cache
        self._perf_parallel: ParallelExecutor | None = perf_parallel

        # Orchestrator references are now accessed via properties that delegate to PluginManager
        # No direct assignment needed - properties handle the delegation

        # Reject learner initialization
        self.reject_learner = (
            self.initialize_reject_learner() if kwargs.get("reject", False) else None
        )

        self._predict_bridge = LegacyPredictBridge(self)

        self.init_time = time() - init_time

    def _coerce_plugin_override(self, override: Any) -> Any:
        """Normalise a plugin override into an instance when possible.

        Delegates to PluginManager.
        """
        # Handle case where PluginManager is not initialized (e.g., in tests)
        if hasattr(self, "_plugin_manager") and self._plugin_manager is not None:
            return self._plugin_manager.coerce_plugin_override(override)

        # Fallback for direct coercion (for backward compatibility in tests)
        if override is None:
            return None
        if isinstance(override, str):
            return override
        if callable(override) and not hasattr(override, "plugin_meta"):
            try:
                candidate = override()
            except Exception as exc:  # pragma: no cover - defensive
                raise ConfigurationError(
                    "Callable explanation plugin override raised an exception"
                ) from exc
            return candidate
        return override

    def _infer_explanation_mode(self) -> str:
        """Infer the explanation mode from runtime state."""
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
    def _prediction_orchestrator(self) -> Any:
        """Get PredictionOrchestrator from PluginManager or test stub storage."""
        # First check if it's stored directly in __dict__ (for test stubs)
        if "__dict__" in dir(self) and "_prediction_orchestrator_stub" in self.__dict__:
            return self.__dict__["_prediction_orchestrator_stub"]
        # Then check PluginManager
        if hasattr(self, "_plugin_manager") and hasattr(self._plugin_manager, "_prediction_orchestrator"):
            return self._plugin_manager._prediction_orchestrator
        # Raise AttributeError if not found
        raise AttributeError("'CalibratedExplainer' object has no attribute '_prediction_orchestrator'")

    @_prediction_orchestrator.setter
    def _prediction_orchestrator(self, value: Any) -> None:
        """Set PredictionOrchestrator for test stubs."""
        # Store directly with a different name to avoid recursion
        self.__dict__["_prediction_orchestrator_stub"] = value

    @property
    def _explanation_orchestrator(self) -> Any:
        """Get ExplanationOrchestrator from PluginManager or test stub storage."""
        # First check if it's stored directly in __dict__ (for test stubs)
        if "__dict__" in dir(self) and "_explanation_orchestrator_stub" in self.__dict__:
            return self.__dict__["_explanation_orchestrator_stub"]
        # Then check PluginManager
        if hasattr(self, "_plugin_manager") and hasattr(self._plugin_manager, "_explanation_orchestrator"):
            return self._plugin_manager._explanation_orchestrator
        # Raise AttributeError if not found
        raise AttributeError("'CalibratedExplainer' object has no attribute '_explanation_orchestrator'")

    @_explanation_orchestrator.setter
    def _explanation_orchestrator(self, value: Any) -> None:
        """Set ExplanationOrchestrator for test stubs."""
        # Store directly with a different name to avoid recursion
        self.__dict__["_explanation_orchestrator_stub"] = value

    @property
    def _reject_orchestrator(self) -> Any:
        """Get RejectOrchestrator from PluginManager or test stub storage."""
        if "__dict__" in dir(self) and "_reject_orchestrator_stub" in self.__dict__:
            return self.__dict__["_reject_orchestrator_stub"]

        if hasattr(self, "_plugin_manager") and hasattr(self._plugin_manager, "_reject_orchestrator"):
            return self._plugin_manager._reject_orchestrator

        raise AttributeError("'CalibratedExplainer' object has no attribute '_reject_orchestrator'")

    @_reject_orchestrator.setter
    def _reject_orchestrator(self, value: Any) -> None:
        """Set RejectOrchestrator for test stubs."""
        self.__dict__["_reject_orchestrator_stub"] = value

    def _build_explanation_chain(self, mode: str) -> Tuple[str, ...]:
        """Delegate to PluginManager for explanation chain building."""
        if not hasattr(self, "_plugin_manager"):
            # Fallback for tests that create minimal stubs
            return tuple()
        default_id = self._plugin_manager._default_explanation_identifiers.get(mode, "")
        return self._plugin_manager._build_explanation_chain(mode, default_id)

    def _build_interval_chain(self, *, fast: bool) -> Tuple[str, ...]:
        """Delegate to PluginManager for interval chain building."""
        if not hasattr(self, "_plugin_manager"):
            return tuple()
        return self._plugin_manager._build_interval_chain(fast=fast)

    def _build_plot_style_chain(self) -> Tuple[str, ...]:
        """Delegate to PluginManager for plot style chain building."""
        if not hasattr(self, "_plugin_manager"):
            return tuple()
        return self._plugin_manager._build_plot_chain()

    def _check_explanation_runtime_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        identifier: str | None,
        mode: str,
    ) -> str | None:
        """Delegate to ExplanationOrchestrator."""
        return self._explanation_orchestrator._check_metadata(
            metadata, identifier=identifier, mode=mode
        )

    def _instantiate_plugin(self, prototype: Any) -> Any:
        """Delegate to ExplanationOrchestrator."""
        return self._explanation_orchestrator._instantiate_plugin(prototype)

    def _build_instance_telemetry_payload(self, explanations: Any) -> Dict[str, Any]:
        """Delegate to ExplanationOrchestrator."""
        return self._explanation_orchestrator._build_instance_telemetry_payload(explanations)

    def _invoke_explanation_plugin(
        self,
        mode: str,
        x: Any,
        threshold: Any,
        low_high_percentiles: Any,
        bins: Any,
        features_to_ignore: Any,
        *,
        extras: Mapping[str, Any] | None = None,
    ) -> Any:
        """Delegate to ExplanationOrchestrator."""
        return self._explanation_orchestrator.invoke(
            mode,
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
            extras=extras,
        )

    def _ensure_interval_runtime_state(self) -> None:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._ensure_interval_runtime_state()

    def _gather_interval_hints(self, *, fast: bool) -> Tuple[str, ...]:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._gather_interval_hints(fast=fast)

    def _check_interval_runtime_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        identifier: str | None,
        fast: bool,
    ) -> str | None:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._check_interval_runtime_metadata(
            metadata, identifier=identifier, fast=fast
        )

    def _resolve_interval_plugin(
        self,
        *,
        fast: bool,
        hints: Sequence[str] = (),
    ) -> Tuple[Any, str | None]:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._resolve_interval_plugin(fast=fast, hints=hints)

    def _build_interval_context(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> IntervalCalibratorContext:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._build_interval_context(fast=fast, metadata=metadata)

    def _obtain_interval_calibrator(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> Tuple[Any, str | None]:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._obtain_interval_calibrator(
            fast=fast, metadata=metadata
        )

    def _capture_interval_calibrators(
        self,
        *,
        context: IntervalCalibratorContext,
        calibrator: Any,
        fast: bool,
    ) -> None:
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._capture_interval_calibrators(
            context=context, calibrator=calibrator, fast=fast
        )

    def _predict_impl(
        self, x, threshold=None, low_high_percentiles=(5, 95), classes=None, bins=None, **kwargs
    ):
        """Delegate to PredictionOrchestrator."""
        return self._prediction_orchestrator._predict_impl(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            **kwargs,
        )

    # ===================================================================
    # Backward-compatibility properties for plugin state (via PluginManager)
    # ===================================================================
    # These properties delegate to PluginManager for backward compatibility
    # with code that accesses plugin state directly from explainer.

    @property
    def _explanation_plugin_overrides(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return {}
        return self._plugin_manager._explanation_plugin_overrides

    @_explanation_plugin_overrides.setter
    def _explanation_plugin_overrides(self, value: Dict[str, Any]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_explanation_overrides = value
            return
        self._plugin_manager._explanation_plugin_overrides = value

    @property
    def _interval_plugin_override(self) -> Any:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._interval_plugin_override

    @_interval_plugin_override.setter
    def _interval_plugin_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_interval_override = value
            return
        self._plugin_manager._interval_plugin_override = value

    @property
    def _fast_interval_plugin_override(self) -> Any:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._fast_interval_plugin_override

    @_fast_interval_plugin_override.setter
    def _fast_interval_plugin_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_fast_interval_override = value
            return
        self._plugin_manager._fast_interval_plugin_override = value

    @property
    def _plot_style_override(self) -> Any:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._plot_style_override

    @_plot_style_override.setter
    def _plot_style_override(self, value: Any) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_plot_style_override = value
            return
        self._plugin_manager._plot_style_override = value

    @property
    def _bridge_monitors(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return {}
        return self._plugin_manager._bridge_monitors

    @_bridge_monitors.setter
    def _bridge_monitors(self, value: Dict[str, Any]) -> None:
        """Set bridge monitors for test stubs."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_bridge_monitors_stub"] = value
            return
        self._plugin_manager._bridge_monitors = value

    @property
    def _explanation_plugin_instances(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return {}
        return self._plugin_manager._explanation_plugin_instances

    @property
    def _explanation_plugin_identifiers(self) -> Dict[str, str]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return {}
        return self._plugin_manager._explanation_plugin_identifiers

    @property
    def _explanation_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_explanation_plugin_fallbacks_stub", {})
        return getattr(self._plugin_manager, "_explanation_plugin_fallbacks", {})

    @_explanation_plugin_fallbacks.setter
    def _explanation_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_explanation_plugin_fallbacks_stub"] = value
            return
        self._plugin_manager._explanation_plugin_fallbacks = value

    @property
    def _plot_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_plot_plugin_fallbacks_stub", {})
        return self._plugin_manager._plot_plugin_fallbacks

    @_plot_plugin_fallbacks.setter
    def _plot_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_plot_plugin_fallbacks_stub"] = value
            return
        self._plugin_manager._plot_plugin_fallbacks = value

    @property
    def _interval_plugin_hints(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_interval_plugin_hints_stub", {})
        return getattr(self._plugin_manager, "_interval_plugin_hints", {})

    @_interval_plugin_hints.setter
    def _interval_plugin_hints(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_interval_plugin_hints_stub"] = value
            return
        self._plugin_manager._interval_plugin_hints = value

    @_interval_plugin_hints.deleter
    def _interval_plugin_hints(self) -> None:
        """Deleter for backward compatibility."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager._interval_plugin_hints

    @property
    def _interval_plugin_fallbacks(self) -> Dict[str, Tuple[str, ...]]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_interval_plugin_fallbacks_stub", {})
        return getattr(self._plugin_manager, "_interval_plugin_fallbacks", {})

    @_interval_plugin_fallbacks.setter
    def _interval_plugin_fallbacks(self, value: Dict[str, Tuple[str, ...]]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_interval_plugin_fallbacks_stub"] = value
            return
        self._plugin_manager._interval_plugin_fallbacks = value

    @_interval_plugin_fallbacks.deleter
    def _interval_plugin_fallbacks(self) -> None:
        """Deleter for backward compatibility."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager._interval_plugin_fallbacks

    @property
    def _interval_plugin_identifiers(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_interval_plugin_identifiers_stub", {})
        return getattr(self._plugin_manager, "_interval_plugin_identifiers", {})

    @_interval_plugin_identifiers.setter
    def _interval_plugin_identifiers(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_interval_plugin_identifiers_stub"] = value
            return
        self._plugin_manager._interval_plugin_identifiers = value

    @_interval_plugin_identifiers.deleter
    def _interval_plugin_identifiers(self) -> None:
        """Deleter for backward compatibility."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager._interval_plugin_identifiers

    @property
    def _telemetry_interval_sources(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_telemetry_interval_sources_stub", {})
        return getattr(self._plugin_manager, "_telemetry_interval_sources", {})

    @_telemetry_interval_sources.setter
    def _telemetry_interval_sources(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_telemetry_interval_sources_stub"] = value
            return
        self._plugin_manager._telemetry_interval_sources = value

    @_telemetry_interval_sources.deleter
    def _telemetry_interval_sources(self) -> None:
        """Deleter for backward compatibility."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager._telemetry_interval_sources

    @property
    def _interval_preferred_identifier(self) -> Dict[str, str | None]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_interval_preferred_identifier_stub", {})
        return getattr(self._plugin_manager, "_interval_preferred_identifier", {})

    @_interval_preferred_identifier.setter
    def _interval_preferred_identifier(self, value: Dict[str, str | None]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_interval_preferred_identifier_stub"] = value
            return
        self._plugin_manager._interval_preferred_identifier = value

    @_interval_preferred_identifier.deleter
    def _interval_preferred_identifier(self) -> None:
        """Deleter for backward compatibility."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager._interval_preferred_identifier

    @property
    def _interval_context_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_interval_context_metadata_stub", {})
        return getattr(self._plugin_manager, "_interval_context_metadata", {})

    @_interval_context_metadata.setter
    def _interval_context_metadata(self, value: Dict[str, Dict[str, Any]]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_interval_context_metadata_stub"] = value
            return
        self._plugin_manager._interval_context_metadata = value

    @_interval_context_metadata.deleter
    def _interval_context_metadata(self) -> None:
        """Deleter for backward compatibility."""
        if hasattr(self, "_plugin_manager"):
            del self._plugin_manager._interval_context_metadata

    @property
    def _plot_style_chain(self) -> Tuple[str, ...] | None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return self.__dict__.get("_plot_style_chain_stub", None)
        return getattr(self._plugin_manager, "_plot_style_chain", None)

    @_plot_style_chain.setter
    def _plot_style_chain(self, value: Tuple[str, ...] | None) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self.__dict__["_plot_style_chain_stub"] = value
            return
        self._plugin_manager._plot_style_chain = value

    @property
    def _explanation_contexts(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return {}
        return self._plugin_manager._explanation_contexts

    @property
    def _last_explanation_mode(self) -> str | None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._last_explanation_mode

    @_last_explanation_mode.setter
    def _last_explanation_mode(self, value: str | None) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_last_explanation_mode = value
            return
        self._plugin_manager._last_explanation_mode = value

    @property
    def _last_telemetry(self) -> Dict[str, Any]:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return {}
        return self._plugin_manager._last_telemetry

    @_last_telemetry.setter
    def _last_telemetry(self, value: Dict[str, Any]) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_last_telemetry = value
            return
        self._plugin_manager._last_telemetry = value

    @property
    def _pyproject_explanations(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._pyproject_explanations

    @_pyproject_explanations.setter
    def _pyproject_explanations(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_pyproject_explanations = value
            return
        self._plugin_manager._pyproject_explanations = value

    @property
    def _pyproject_intervals(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._pyproject_intervals

    @_pyproject_intervals.setter
    def _pyproject_intervals(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_pyproject_intervals = value
            return
        self._plugin_manager._pyproject_intervals = value

    @property
    def _pyproject_plots(self) -> Dict[str, Any] | None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            return None
        return self._plugin_manager._pyproject_plots

    @_pyproject_plots.setter
    def _pyproject_plots(self, value: Dict[str, Any] | None) -> None:
        """Delegate to PluginManager."""
        if not hasattr(self, "_plugin_manager"):
            self._plugin_manager_cache_pyproject_plots = value
            return
        self._plugin_manager._pyproject_plots = value

    @property
    def runtime_telemetry(self) -> Mapping[str, Any]:
        """Return the most recent telemetry payload reported by the explainer."""
        return dict(self._last_telemetry)

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
        from .calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

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
        from .calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.set_x_cal(self, value)

    @property
    def y_cal(self):
        """Get the calibration target data.

        Returns
        -------
        array-like
            The calibration target data.
        """
        from .calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        return CalibrationState.get_y_cal(self)

    @y_cal.setter
    def y_cal(self, value):
        """Set the calibration target data.

        Parameters
        ----------
        value : array-like of shape (n_samples,)
            The new calibration target data.
        """
        from .calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

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
        from .calibration.state import CalibrationState  # pylint: disable=import-outside-toplevel

        CalibrationState.append_calibration(self, x, y)

    def _invalidate_calibration_summaries(self) -> None:
        """Drop cached calibration summaries used during explanation.

        Delegates to the calibration.summaries module which manages the cache.
        """
        from .calibration.summaries import (  # pylint: disable=import-outside-toplevel
            invalidate_calibration_summaries as _invalidate,
        )

        _invalidate(self)

    def _get_calibration_summaries(
        self, x_cal_np: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
        """Return cached categorical counts and sorted numeric calibration values.

        Delegates to the calibration.summaries module which manages caching of
        statistical summaries used during explanation generation.
        """
        from .calibration.summaries import (  # pylint: disable=import-outside-toplevel
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
        return self._prediction_orchestrator._interval_registry.interval_learner

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
        self._prediction_orchestrator._interval_registry.interval_learner = value

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
        return self._prediction_orchestrator._interval_registry.get_sigma_test(x)

    def _CalibratedExplainer__initialize_interval_learner_for_fast_explainer(self) -> None:  # noqa: N802
        """Backward-compatible wrapper for fast-mode interval learner initialization.

        Notes
        -----
        This method delegates to the interval registry. It is kept for backward
        compatibility with the external fast_explanations plugin and other
        production code that calls this private method.

        See ADR-001.
        """
        self._prediction_orchestrator._interval_registry.initialize_for_fast_explainer()

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
            from .calibration.interval_learner import update_interval_learner as _upd_il

            _upd_il(self, xs, ys, bins=bins)
        else:
            from .calibration.interval_learner import initialize_interval_learner as _init_il

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

    def _predict(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        classes=None,
        bins=None,
        feature=None,
        **kwargs,
    ):
        """Cache-aware prediction wrapper. Delegated to PredictionOrchestrator."""
        return self._prediction_orchestrator.predict(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            feature=feature,
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

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FactualExplanation` for each instance.
        """
        # Thin delegator that sets discretizer and delegates to orchestrator
        discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        return self._explanation_orchestrator.invoke_factual(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
            discretizer=discretizer,
            _use_plugin=_use_plugin,
        )

    def explain_counterfactual(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
    ) -> AlternativeExplanations:
        """See documentation for the `explore_alternatives` method.

        See Also
        --------
        :meth:`.CalibratedExplainer.explore_alternatives` : Refer to the documentation for `explore_alternatives` for more details.

        Warnings
        --------
        Deprecated: This method is deprecated and may be removed in future versions. Use `explore_alternatives` instead.
        """
        _warnings.warn(
            "The `explain_counterfactual` method is deprecated and may be removed in future versions. Use `explore_alternatives` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.explore_alternatives(
            x, threshold, low_high_percentiles, bins, features_to_ignore
        )

    def explore_alternatives(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        _use_plugin: bool = True,
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

        Returns
        -------
        AlternativeExplanations : :class:`.AlternativeExplanations`
            An `AlternativeExplanations` containing one :class:`.AlternativeExplanation` for each instance.

        Notes
        -----
        The `explore_alternatives` will eventually be used instead of the `explain_counterfactual` method.
        """
        # Thin delegator that sets discretizer and delegates to orchestrator
        discretizer = "regressor" if "regression" in self.mode else "entropy"
        return self._explanation_orchestrator.invoke_alternative(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
            discretizer=discretizer,
            _use_plugin=_use_plugin,
        )  # type: ignore[return-value]

    def __call__(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        _use_plugin: bool = True,
        _skip_instance_parallel: bool = False,
    ) -> CalibratedExplanations:
        """Call self as a function to create a :class:`.CalibratedExplanations` object for the test data with the already assigned discretizer.

        Since v0.4.0, this method is equivalent to the `explain` method.
        """
        return self.explain(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
            _use_plugin=_use_plugin,
        )

    def explain(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        features_to_ignore=None,
        *,
        _use_plugin: bool = True,
        _skip_instance_parallel: bool = False,
    ) -> CalibratedExplanations:
        """Generate explanations for test instances by analyzing feature effects.

        This is a thin delegator that delegates to the explanation orchestrator.

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
        # Thin delegator to orchestrator
        if _use_plugin:
            mode = self._infer_explanation_mode()
            return self._explanation_orchestrator.invoke(
                mode,
                x,
                threshold,
                low_high_percentiles,
                bins,
                features_to_ignore,
                extras={"mode": mode, "_skip_instance_parallel": _skip_instance_parallel},
            )

        # Legacy path for backward compatibility and testing
        from .explain._legacy_explain import explain as legacy_explain  # pylint: disable=import-outside-toplevel

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

    @staticmethod
    def _slice_threshold(threshold, start: int, stop: int, total_len: int):
        """Delegate to explain._helpers.

        Return the portion of *threshold* covering ``[start, stop)``.
        Moved to explain._helpers for consolidation.
        """
        from .explain._helpers import slice_threshold

        return slice_threshold(threshold, start, stop, total_len)

    @staticmethod
    def _compute_weight_delta(baseline, perturbed):
        """Delegate to explain._helpers.

        Return the contribution weight delta between baseline and perturbed.
        Compatibility wrapper for compute_weight_delta moved to explain._helpers.
        """

        return compute_weight_delta(baseline, perturbed)

    @staticmethod
    def _slice_bins(bins, start: int, stop: int):
        """Delegate to explain._helpers.

        Return the subset of *bins* covering ``[start, stop)``.
        Moved to explain._helpers for consolidation.
        """
        from .explain._helpers import slice_bins

        return slice_bins(bins, start, stop)

    def _validate_and_prepare_input(self, x):
        """Delegate to explain helpers.

        Validates and prepares input data for explanation generation.
        Moved to explain._helpers to consolidate all explanation logic.
        """
        from .explain._helpers import validate_and_prepare_input as _vh

        return _vh(self, x)

    def _initialize_explanation(self, x, low_high_percentiles, threshold, bins, features_to_ignore):
        """Delegate to explain computation.

        Initializes a CalibratedExplanations object with all metadata.
        Moved to explain._computation to consolidate all explanation logic.
        """
        from .explain._computation import initialize_explanation as _ih

        return _ih(self, x, low_high_percentiles, threshold, bins, features_to_ignore)

    def explain_fast(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        bins=None,
        *,
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
        if _use_plugin:
            return self._invoke_explanation_plugin(
                "fast",
                x,
                threshold,
                low_high_percentiles,
                bins,
                tuple(self.features_to_ignore),
                extras={"mode": "fast"},
            )

        # Delegate to external plugin pipeline for non-plugin path
        # pylint: disable-next=import-outside-toplevel
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
        # Delegate to external plugin pipeline
        # pylint: disable-next=import-outside-toplevel
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
        from external_plugins.integrations.shap_pipeline import ShapPipeline

        pipeline = ShapPipeline(self)
        return pipeline.explain(x, **kwargs)

    def assign_threshold(self, threshold):
        """Assign the threshold for the explainer.

        The threshold is used to calculate the p-values for the predictions.
        """
        if threshold is None:
            return None
        if isinstance(threshold, (list, np.ndarray)):
            return (
                np.empty((0,), dtype=tuple) if isinstance(threshold[0], tuple) else np.empty((0,))
            )
        return threshold

    def _assign_weight(self, instance_predict, prediction):
        """Compute contribution weight as the delta from the global prediction.

        This method computes per-instance or per-class weight deltas for
        probabilistic regression feature attribution. For scalar weight
        computation, use calibrated_explanations.core.explain.feature_task.assign_weight_scalar
        which is optimized for single-value inputs.

        Parameters
        ----------
        instance_predict : scalar or array-like
            Baseline prediction(s).
        prediction : scalar or array-like
            Perturbed prediction(s).

        Returns
        -------
        scalar or list
            Weight delta(s). Returns same type as inputs.
        """
        return (
            prediction - instance_predict
            if np.isscalar(prediction)
            else [prediction[i] - ip for i, ip in enumerate(instance_predict)]
        )  # used for probabilistic regression feature attribution

    def is_multiclass(self):
        """Test if it is a multiclass problem.

        Returns
        -------
        bool
            True if multiclass.
        """
        return self.num_classes > 2

    def is_fast(self):
        """Test if the explainer is fast.

        Returns
        -------
        bool
            True if fast.
        """
        return self.__fast

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
        from .explain._computation import rule_boundaries as _rule_boundaries  # pylint: disable=import-outside-toplevel

        return _rule_boundaries(self, instances, perturbed_instances)

    def __get_greater_values(self, f: int, greater: float):
        """Get sampled values above ``greater`` for numerical features.

        Uses percentile sampling from calibration data.
        """
        if not np.any(self.x_cal[:, f] > greater):
            return np.array([])
        candidates = np.percentile(
            self.x_cal[self.x_cal[:, f] > greater, f], self.sample_percentiles
        )
        return candidates

    def __get_lesser_values(self, f: int, lesser: float):
        """Get sampled values below ``lesser`` for numerical features.

        Uses percentile sampling from calibration data.
        """
        if not np.any(self.x_cal[:, f] < lesser):
            return np.array([])
        candidates = np.percentile(
            self.x_cal[self.x_cal[:, f] < lesser, f], self.sample_percentiles
        )
        return candidates

    def __get_covered_values(self, f: int, lesser: float, greater: float):
        """Get sampled values within the ``[lesser, greater]`` interval.

        Uses percentile sampling from calibration data.
        """
        covered = np.where((self.x_cal[:, f] >= lesser) & (self.x_cal[:, f] <= greater))[0]
        if len(covered) == 0:
            return np.array([])
        candidates = np.percentile(self.x_cal[covered, f], self.sample_percentiles)
        return candidates

    def set_seed(self, seed: int) -> None:
        """Change the seed used in the random number generator.

        Parameters
        ----------
        seed : int
            The seed to be used in the random number generator.
        """
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

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
        if difficulty_estimator is not None:
            try:
                if not difficulty_estimator.fitted:
                    raise NotFittedError(
                        "The difficulty estimator is not fitted. Please fit the estimator first."
                    )
            except AttributeError as e:
                raise NotFittedError(
                    "The difficulty estimator is not fitted. Please fit the estimator first."
                ) from e
        self.__initialized = False
        self.difficulty_estimator = difficulty_estimator
        if initialize:
            self._prediction_orchestrator._interval_registry.initialize()  # type: ignore[attr-defined]

    def __set_mode(self, mode, initialize=True) -> None:
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
            self._prediction_orchestrator._interval_registry.initialize()  # type: ignore[attr-defined]

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
        return self._reject_orchestrator.initialize_reject_learner(
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
        return self._reject_orchestrator.predict_reject(x, bins=bins, confidence=confidence)

    def _preprocess(self):
        """Identify constant calibration features that can be ignored downstream."""
        constant_columns = [
            f for f in range(self.num_features) if np.all(self.x_cal[:, f] == self.x_cal[0, f])
        ]
        self.features_to_ignore = constant_columns

    def _discretize(self, x):
        """Apply the discretizer to the data sample x.

        For new data samples and missing values, the nearest bin is used.

        Parameters
        ----------
        x : array-like
            The data sample to discretize.

        Returns
        -------
        array-like
            The discretized data sample.
        """
        from .explain._computation import discretize  # pylint: disable=import-outside-toplevel

        return discretize(self, x)

    # pylint: disable=too-many-branches
    def set_discretizer(self, discretizer, x_cal=None, y_cal=None, features_to_ignore=None) -> None:
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
        if x_cal is None:
            x_cal = self.x_cal
        if y_cal is None:
            y_cal = self.y_cal

        if discretizer is None:
            discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        elif "regression" in self.mode:
            if not (
                discretizer is None
                or discretizer
                in {
                    "regressor",
                    "binaryRegressor",
                }
            ):
                raise ValidationError(
                    "The discretizer must be 'binaryRegressor' (default for factuals) or 'regressor' (default for alternatives) for regression."
                )
        else:
            if not (
                discretizer is None
                or discretizer
                in {
                    "entropy",
                    "binaryEntropy",
                }
            ):
                raise ValidationError(
                    "The discretizer must be 'binaryEntropy' (default for factuals) or 'entropy' (default for alternatives) for classification."
                )

        if features_to_ignore is None:
            features_to_ignore = []
        not_to_discretize = np.union1d(
            np.union1d(self.categorical_features, self.features_to_ignore), features_to_ignore
        )
        if discretizer == "binaryEntropy":
            if isinstance(self.discretizer, BinaryEntropyDiscretizer):
                return
            self.discretizer = BinaryEntropyDiscretizer(
                x_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )
        elif discretizer == "binaryRegressor":
            if isinstance(self.discretizer, BinaryRegressorDiscretizer):
                return
            self.discretizer = BinaryRegressorDiscretizer(
                x_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )

        elif discretizer == "entropy":
            if isinstance(self.discretizer, EntropyDiscretizer):
                return
            self.discretizer = EntropyDiscretizer(
                x_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )
        elif discretizer == "regressor":
            if isinstance(self.discretizer, RegressorDiscretizer):
                return
            self.discretizer = RegressorDiscretizer(
                x_cal, not_to_discretize, self.feature_names, labels=y_cal, random_state=self.seed
            )
        self.discretized_X_cal = self._discretize(immutable_array(self.x_cal))

        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in range(self.num_features):
            assert self.discretized_X_cal is not None
            column = self.discretized_X_cal[:, feature]
            feature_count: Dict[Any, int] = {}
            for item in column:
                feature_count[item] = feature_count.get(item, 0) + 1
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = np.array(frequencies) / float(sum(frequencies))

    def _is_mondrian(self):
        """Return whether the explainer is a Mondrian explainer.

        Returns
        -------
            bool: True if Mondrian
        """
        return self.bins is not None

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
        # emit deprecation warnings for aliases and normalize kwargs
        warn_on_aliases(kwargs)
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)

        if not calibrated:
            if "threshold" in kwargs:
                raise ValidationError(
                    "A thresholded prediction is not possible for uncalibrated predictions."
                )
            if uq_interval:
                predict = self.learner.predict(x)
                return predict, (predict, predict)
            return self.learner.predict(x)

        if self.mode in "regression":
            predict, low, high, _ = self._predict(x, **kwargs)
            if "threshold" in kwargs:

                def get_label(predict, threshold):
                    if np.isscalar(threshold):
                        return f"y_hat <= {threshold}" if predict >= 0.5 else f"y_hat > {threshold}"
                    if isinstance(threshold, tuple):
                        return (
                            f"{threshold[0]} < y_hat <= {threshold[1]}"
                            if predict >= 0.5
                            else f"y_hat <= {threshold[0]} || y_hat > {threshold[1]}"
                        )
                    return (
                        "Error in CalibratedExplainer.predict.get_label()"  # should not reach here
                    )

                threshold = kwargs["threshold"]
                if np.isscalar(threshold) or isinstance(threshold, tuple):
                    new_classes = [get_label(predict[i], threshold) for i in range(len(predict))]
                else:
                    new_classes = [get_label(predict[i], threshold[i]) for i in range(len(predict))]
                return (new_classes, (low, high)) if uq_interval else new_classes
            return (predict, (low, high)) if uq_interval else predict

        predict, low, high, new_classes = self._predict(x, **kwargs)
        if new_classes is None:
            new_classes = (predict >= 0.5).astype(int)
        if self.label_map is not None or self.class_labels is not None:
            new_classes = np.array([self.class_labels[c] for c in new_classes])
        return (new_classes, (low, high)) if uq_interval else new_classes

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
        # emit deprecation warnings for aliases and normalize kwargs
        warn_on_aliases(kwargs)
        kwargs = canonicalize_kwargs(kwargs)
        validate_param_combination(kwargs)
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
        if self.mode in "regression":
            if isinstance(self.interval_learner, list):
                proba_1, low, high, _ = self.interval_learner[-1].predict_probability(
                    x, y_threshold=threshold, **kwargs
                )
            else:
                proba_1, low, high, _ = self.interval_learner.predict_probability(
                    x, y_threshold=threshold, **kwargs
                )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            return (proba, (low, high)) if uq_interval else proba
        if self.is_multiclass():  # pylint: disable=protected-access
            if isinstance(self.interval_learner, list):
                proba, low, high, _ = self.interval_learner[-1].predict_proba(
                    x, output_interval=True, **kwargs
                )
            else:
                proba, low, high, _ = self.interval_learner.predict_proba(
                    x, output_interval=True, **kwargs
                )
            return (proba, (low, high)) if uq_interval else proba
        if isinstance(self.interval_learner, list):
            proba, low, high = self.interval_learner[-1].predict_proba(
                x, output_interval=True, **kwargs
            )
        else:
            proba, low, high = self.interval_learner.predict_proba(
                x, output_interval=True, **kwargs
            )
        return (proba, (low, high)) if uq_interval else proba

    def _is_lime_enabled(self, is_enabled=None) -> bool:
        """Return whether LIME export is enabled.

        .. deprecated:: 0.10.0
            Access LIME state through LimePipeline instead.
        """
        # Delegate to LimePipeline for lazy initialization and caching
        if not hasattr(self, "_lime_pipeline"):
            # pylint: disable-next=import-outside-toplevel
            from external_plugins.integrations.lime_pipeline import LimePipeline

            self._lime_pipeline = LimePipeline(self)
        return self._lime_pipeline._is_lime_enabled(is_enabled)

    def _is_shap_enabled(self, is_enabled=None) -> bool:
        """Return whether SHAP export is enabled.

        .. deprecated:: 0.10.0
            Access SHAP state through ShapPipeline instead.
        """
        # Delegate to ShapPipeline for lazy initialization and caching
        if not hasattr(self, "_shap_pipeline"):
            # pylint: disable-next=import-outside-toplevel
            from external_plugins.integrations.shap_pipeline import ShapPipeline

            self._shap_pipeline = ShapPipeline(self)
        return self._shap_pipeline._is_shap_enabled(is_enabled)

    def _preload_lime(self, x_cal=None):
        """Materialize LIME explainer artifacts when the dependency is available.

        .. deprecated:: 0.10.0
            Use LimePipeline._preload_lime instead.
        """
        # Delegate to LimePipeline for lazy initialization and caching
        if not hasattr(self, "_lime_pipeline"):
            # pylint: disable-next=import-outside-toplevel
            from external_plugins.integrations.lime_pipeline import LimePipeline

            self._lime_pipeline = LimePipeline(self)
        return self._lime_pipeline._preload_lime(x_cal=x_cal)

    def _preload_shap(self, num_test=None):
        """Eagerly compute SHAP explanations to amortize repeated requests.

        .. deprecated:: 0.10.0
            Use ShapPipeline._preload_shap instead.
        """
        # Delegate to ShapPipeline for lazy initialization and caching
        if not hasattr(self, "_shap_pipeline"):
            # pylint: disable-next=import-outside-toplevel
            from external_plugins.integrations.shap_pipeline import ShapPipeline

            self._shap_pipeline = ShapPipeline(self)
        return self._shap_pipeline._preload_shap(num_test=num_test)

    # pylint: disable=duplicate-code, too-many-branches, too-many-statements, too-many-locals
    def plot(self, x, y=None, threshold=None, **kwargs):
        """Generate plots for the test data."""
        # Pass any style overrides along to the plotting function
        style_override = kwargs.pop("style_override", None)
        kwargs["style_override"] = style_override
        _plot_global(self, x, y=y, threshold=threshold, **kwargs)

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
        if not (self.mode == "classification"):
            raise ValidationError(
                "The confusion matrix is only available for classification tasks."
            )
        y_cal = np.asarray(self.y_cal)
        bins = None if self.bins is None else np.asarray(self.bins)
        n_samples = len(y_cal)

        if n_samples == 0:
            raise ValidationError(
                "At least one calibration sample is required to build a confusion matrix."
            )

        cal_predicted_classes = np.empty_like(y_cal)

        # Determine the maximum feasible number of stratified folds.
        n_splits = min(10, n_samples)
        class_counts = Counter(y_cal)
        while n_splits > 1 and any(count < n_splits for count in class_counts.values()):
            n_splits -= 1

        if n_splits <= 1:
            va = VennAbers(self.x_cal, self.y_cal, self.learner, bins=self.bins)
            _, _, _, predict = va.predict_proba(
                self.x_cal,
                output_interval=True,
                bins=self.bins,
            )
            cal_predicted_classes[:] = predict
            return confusion_matrix(self.y_cal, cal_predicted_classes)

        if len(class_counts) > 1:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            split_iter = splitter.split(self.x_cal, y_cal)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            split_iter = splitter.split(self.x_cal)

        for train_idx, test_idx in split_iter:
            va = VennAbers(
                self.x_cal[train_idx],
                y_cal[train_idx],
                self.learner,
                bins=bins[train_idx] if bins is not None else None,
            )
            _, _, _, predict = va.predict_proba(
                self.x_cal[test_idx],
                output_interval=True,
                bins=bins[test_idx] if bins is not None else None,
            )
            cal_predicted_classes[test_idx] = predict
        return confusion_matrix(self.y_cal, cal_predicted_classes)

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


__all__ = ["CalibratedExplainer"]
