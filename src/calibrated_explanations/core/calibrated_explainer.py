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
import os
from pathlib import Path
from collections import defaultdict
from time import time

import numpy as np
import contextlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:  # pragma: no cover - optional dependency path
        import tomli as _tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - tomllib unavailable
        _tomllib = None  # type: ignore[assignment]
from crepes import ConformalClassifier
from crepes.extras import hinge
from sklearn.metrics import confusion_matrix

from ..perf import CalibratorCache, ParallelExecutor
from ..plotting import _plot_global
from .venn_abers import VennAbers
from ..explanations import AlternativeExplanations, CalibratedExplanations
from ..utils.discretizers import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)
from ..utils.helper import (
    assert_threshold,
    check_is_fitted,
    concatenate_thresholds,
    convert_targets_to_numeric,
    immutable_array,
    safe_import,
    safe_mean,
    safe_isinstance,
)
from ..api.params import canonicalize_kwargs, validate_param_combination, warn_on_aliases
from ..plugins import (
    ExplanationContext,
    ExplanationRequest,
    IntervalCalibratorContext,
    validate_explanation_batch,
)
from ..plugins.builtins import LegacyPredictBridge
from ..plugins.registry import (
    EXPLANATION_PROTOCOL_VERSION,
    ensure_builtin_plugins,
    find_explanation_descriptor,
    find_explanation_plugin,
    find_interval_descriptor,
    find_interval_plugin,
    find_interval_plugin_trusted,
    is_identifier_denied,
)
from ..plugins.predict import PredictBridge

from .exceptions import (
    ValidationError,
    DataShapeError,
    ConfigurationError,
    NotFittedError,
)


def _read_pyproject_section(path: Sequence[str]) -> Dict[str, Any]:
    """Return a mapping from the requested ``pyproject.toml`` section."""
    if _tomllib is None:
        return {}

    candidate = Path.cwd() / "pyproject.toml"
    if not candidate.exists():
        return {}
    try:
        with candidate.open("rb") as fh:  # type: ignore[arg-type]
            data = _tomllib.load(fh)
    except Exception:  # pragma: no cover - permissive fallback
        return {}

    cursor: Any = data
    for key in path:
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        else:
            return {}
    if isinstance(cursor, dict):
        return dict(cursor)
    return {}


def _split_csv(value: str | None) -> Tuple[str, ...]:
    """Split a comma separated environment variable into a tuple."""
    if not value:
        return ()
    entries = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(entries)


def _coerce_string_tuple(value: Any) -> Tuple[str, ...]:
    """Coerce a configuration value into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Iterable):
        result: List[str] = []
        for item in value:
            if isinstance(item, str) and item:
                result.append(item)
        return tuple(result)
    return ()


_EXPLANATION_MODES: Tuple[str, ...] = ("factual", "alternative", "fast")

_DEFAULT_EXPLANATION_IDENTIFIERS: Dict[str, str] = {
    "factual": "core.explanation.factual",
    "alternative": "core.explanation.alternative",
    "fast": "core.explanation.fast",
}


class _PredictBridgeMonitor(PredictBridge):
    """Runtime guard ensuring plugins use the calibrated predict bridge."""

    def __init__(self, bridge: PredictBridge) -> None:
        self._bridge = bridge
        self._calls: List[str] = []

    def reset_usage(self) -> None:
        self._calls.clear()

    def predict(
        self,
        x: Any,
        *,
        mode: str,
        task: str,
        bins: Any | None = None,
    ) -> Mapping[str, Any]:
        self._calls.append("predict")
        return self._bridge.predict(x, mode=mode, task=task, bins=bins)

    def predict_interval(
        self,
        x: Any,
        *,
        task: str,
        bins: Any | None = None,
    ) -> Sequence[Any]:
        self._calls.append("predict_interval")
        return self._bridge.predict_interval(x, task=task, bins=bins)

    def predict_proba(self, x: Any, bins: Any | None = None) -> Sequence[Any]:
        self._calls.append("predict_proba")
        return self._bridge.predict_proba(x, bins=bins)

    @property
    def calls(self) -> Tuple[str, ...]:
        return tuple(self._calls)

    @property
    def used(self) -> bool:
        return bool(self._calls)


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
        self.__shap_enabled = False
        self.__lime_enabled = False
        self.lime: Any = None
        self.lime_exp: Any = None
        self.shap: Any = None
        self.shap_exp: Any = None
        self.reject = kwargs.get("reject", False)

        self.set_difficulty_estimator(difficulty_estimator, initialize=False)
        self.__set_mode(str.lower(mode), initialize=False)

        self.interval_learner: Any = None
        self._perf_cache: CalibratorCache[Any] | None = perf_cache
        self._perf_parallel: ParallelExecutor | None = perf_parallel
        self._pyproject_explanations = _read_pyproject_section(
            ("tool", "calibrated_explanations", "explanations")
        )
        self._pyproject_intervals = _read_pyproject_section(
            ("tool", "calibrated_explanations", "intervals")
        )
        self._pyproject_plots = _read_pyproject_section(
            ("tool", "calibrated_explanations", "plots")
        )
        self._explanation_plugin_overrides: Dict[str, Any] = {
            mode: kwargs.get(f"{mode}_plugin") for mode in _EXPLANATION_MODES
        }
        self._interval_plugin_override = kwargs.get("interval_plugin")
        self._fast_interval_plugin_override = kwargs.get("fast_interval_plugin")
        self._plot_style_override = kwargs.get("plot_style")
        self._bridge_monitors: Dict[str, _PredictBridgeMonitor] = {}
        self._explanation_plugin_instances: Dict[str, Any] = {}
        self._explanation_plugin_identifiers: Dict[str, str] = {}
        self._explanation_plugin_fallbacks: Dict[str, Tuple[str, ...]] = {}
        self._plot_plugin_fallbacks: Dict[str, Tuple[str, ...]] = {}
        self._interval_plugin_hints: Dict[str, Tuple[str, ...]] = {}
        self._interval_plugin_fallbacks: Dict[str, Tuple[str, ...]] = {}
        self._interval_plugin_identifiers: Dict[str, str | None] = {
            "default": None,
            "fast": None,
        }
        self._telemetry_interval_sources: Dict[str, str | None] = {
            "default": None,
            "fast": None,
        }
        self._interval_preferred_identifier: Dict[str, str | None] = {
            "default": None,
            "fast": None,
        }
        self._interval_context_metadata: Dict[str, Dict[str, Any]] = {
            "default": {},
            "fast": {},
        }
        self._plot_style_chain: Tuple[str, ...] | None = None
        self._explanation_contexts: Dict[str, ExplanationContext] = {}
        self._last_explanation_mode: str | None = None
        self._last_telemetry: Dict[str, Any] = {}
        self._ensure_interval_runtime_state()
        for mode in _EXPLANATION_MODES:
            self._explanation_plugin_fallbacks[mode] = self._build_explanation_chain(mode)
        self._interval_plugin_fallbacks["default"] = self._build_interval_chain(fast=False)
        self._interval_plugin_fallbacks["fast"] = self._build_interval_chain(fast=True)
        self._plot_style_chain = self._build_plot_style_chain()

        # Phase 1A delegation: interval learner initialization via helper
        from .calibration_helpers import initialize_interval_learner as _init_il

        _init_il(self)
        self.reject_learner = (
            self.initialize_reject_learner() if kwargs.get("reject", False) else None
        )

        self._predict_bridge = LegacyPredictBridge(self)

        self.init_time = time() - init_time

    # ------------------------------------------------------------------
    # Plugin resolution helpers (ADR-015)
    # ------------------------------------------------------------------

    def _build_explanation_chain(self, mode: str) -> Tuple[str, ...]:
        """Return the ordered identifier fallback chain for *mode*."""
        entries: List[str] = []

        override = self._explanation_plugin_overrides.get(mode)
        if isinstance(override, str) and override:
            entries.append(override)

        env_key = f"CE_EXPLANATION_PLUGIN_{mode.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            entries.append(env_value.strip())
        entries.extend(_split_csv(os.environ.get(f"{env_key}_FALLBACKS")))

        py_settings = self._pyproject_explanations or {}
        py_value = py_settings.get(mode)
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(_coerce_string_tuple(py_settings.get(f"{mode}_fallbacks")))

        # Deduplicate while maintaining order and extend using metadata fallbacks
        seen: set[str] = set()
        expanded: List[str] = []
        for identifier in entries:
            if not identifier or identifier in seen:
                continue
            expanded.append(identifier)
            seen.add(identifier)
            descriptor = find_explanation_descriptor(identifier)
            if descriptor:
                for fallback in _coerce_string_tuple(descriptor.metadata.get("fallbacks")):
                    if fallback and fallback not in seen:
                        expanded.append(fallback)
                        seen.add(fallback)

        default_identifier = _DEFAULT_EXPLANATION_IDENTIFIERS.get(mode)
        if default_identifier and default_identifier not in seen:
            if mode == "fast" and find_explanation_descriptor(default_identifier) is None:
                pass
            else:
                expanded.append(default_identifier)
        return tuple(expanded)

    def _build_interval_chain(self, *, fast: bool) -> Tuple[str, ...]:
        """Return the ordered interval plugin chain for the requested mode."""
        entries: List[str] = []
        override = self._fast_interval_plugin_override if fast else self._interval_plugin_override
        preferred_identifier: str | None = None
        if isinstance(override, str) and override:
            entries.append(override)
            preferred_identifier = override

        env_key = "CE_INTERVAL_PLUGIN_FAST" if fast else "CE_INTERVAL_PLUGIN"
        env_value = os.environ.get(env_key)
        if env_value:
            entries.append(env_value.strip())
            if preferred_identifier is None:
                preferred_identifier = env_value.strip()
        entries.extend(_split_csv(os.environ.get(f"{env_key}_FALLBACKS")))

        py_settings = self._pyproject_intervals or {}
        py_key = "fast" if fast else "default"
        py_value = py_settings.get(py_key)
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(_coerce_string_tuple(py_settings.get(f"{py_key}_fallbacks")))

        default_identifier = "core.interval.fast" if fast else "core.interval.legacy"
        seen: set[str] = set()
        ordered: List[str] = []
        for identifier in entries:
            if identifier and identifier not in seen:
                ordered.append(identifier)
                seen.add(identifier)
                descriptor = find_interval_descriptor(identifier)
                if descriptor:
                    for fallback in _coerce_string_tuple(descriptor.metadata.get("fallbacks")):
                        if fallback and fallback not in seen:
                            ordered.append(fallback)
                            seen.add(fallback)
        if default_identifier not in seen:
            if fast and find_interval_descriptor(default_identifier) is None:
                pass
            else:
                ordered.append(default_identifier)
        key = "fast" if fast else "default"
        self._interval_preferred_identifier[key] = preferred_identifier
        return tuple(ordered)

    def _build_plot_style_chain(self) -> Tuple[str, ...]:
        """Return the ordered plot style fallback chain."""
        entries: List[str] = []
        if isinstance(self._plot_style_override, str) and self._plot_style_override:
            entries.append(self._plot_style_override)

        env_value = os.environ.get("CE_PLOT_STYLE")
        if env_value:
            entries.append(env_value.strip())
        entries.extend(_split_csv(os.environ.get("CE_PLOT_STYLE_FALLBACKS")))

        py_settings = self._pyproject_plots or {}
        py_value = py_settings.get("style")
        if isinstance(py_value, str) and py_value:
            entries.append(py_value)
        entries.extend(_coerce_string_tuple(py_settings.get("style_fallbacks")))
        entries.append("legacy")
        seen: set[str] = set()
        ordered: List[str] = []
        for identifier in entries:
            if identifier and identifier not in seen:
                ordered.append(identifier)
                seen.add(identifier)
        if "plot_spec.default" not in seen:
            if "legacy" in ordered:
                legacy_index = ordered.index("legacy")
                ordered.insert(legacy_index, "plot_spec.default")
            else:
                ordered.append("plot_spec.default")
        if "legacy" not in ordered:
            ordered.append("legacy")
        return tuple(ordered)

    def _ensure_interval_runtime_state(self) -> None:
        """Ensure interval tracking members exist for legacy instances."""
        storage = self.__dict__
        if "_interval_plugin_hints" not in storage:
            storage["_interval_plugin_hints"] = {}
        if "_interval_plugin_fallbacks" not in storage:
            storage["_interval_plugin_fallbacks"] = {}
        if "_interval_plugin_identifiers" not in storage:
            storage["_interval_plugin_identifiers"] = {"default": None, "fast": None}
        if "_telemetry_interval_sources" not in storage:
            storage["_telemetry_interval_sources"] = {"default": None, "fast": None}
        if "_interval_preferred_identifier" not in storage:
            storage["_interval_preferred_identifier"] = {"default": None, "fast": None}
        if "_interval_context_metadata" not in storage:
            storage["_interval_context_metadata"] = {"default": {}, "fast": {}}

    def _coerce_plugin_override(self, override: Any) -> Any:
        """Normalise a plugin override into an instance when possible."""
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

    def _check_explanation_runtime_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        identifier: str | None,
        mode: str,
    ) -> str | None:
        """Return an error message if *metadata* is incompatible at runtime."""
        prefix = identifier or str((metadata or {}).get("name") or "<anonymous>")
        if metadata is None:
            return f"{prefix}: plugin metadata unavailable"

        schema_version = metadata.get("schema_version")
        if schema_version != EXPLANATION_PROTOCOL_VERSION:
            return (
                f"{prefix}: explanation schema_version {schema_version} unsupported; "
                f"expected {EXPLANATION_PROTOCOL_VERSION}"
            )

        tasks = _coerce_string_tuple(metadata.get("tasks"))
        if not tasks:
            return f"{prefix}: plugin metadata missing tasks declaration"
        if "both" not in tasks and self.mode not in tasks:
            declared = ", ".join(tasks)
            return f"{prefix}: does not support task '{self.mode}' " f"(declared: {declared})"

        modes = _coerce_string_tuple(metadata.get("modes"))
        if not modes:
            return f"{prefix}: plugin metadata missing modes declaration"
        if mode not in modes:
            declared = ", ".join(modes)
            return f"{prefix}: does not declare mode '{mode}' (modes: {declared})"

        capabilities = metadata.get("capabilities")
        cap_set: set[str] = set()
        if isinstance(capabilities, Iterable):
            for capability in capabilities:
                cap_set.add(str(capability))

        missing: List[str] = []
        if "explain" not in cap_set:
            missing.append("explain")
        mode_cap = f"explanation:{mode}"
        if mode_cap not in cap_set:
            alt_mode_cap = f"mode:{mode}"
            if alt_mode_cap not in cap_set:
                missing.append(mode_cap)
        task_cap = f"task:{self.mode}"
        if task_cap not in cap_set and "task:both" not in cap_set:
            missing.append(task_cap)

        if missing:
            return f"{prefix}: missing required capabilities {', '.join(sorted(missing))}"

        return None

    def _instantiate_plugin(self, prototype: Any) -> Any:
        """Best-effort instantiation that avoids sharing state across explainers."""
        if prototype is None:
            return None
        if callable(prototype) and hasattr(prototype, "plugin_meta"):
            return prototype
        plugin_cls = type(prototype)
        try:
            return plugin_cls()
        except Exception:
            try:
                import copy

                return copy.deepcopy(prototype)
            except Exception:  # pragma: no cover - defensive
                return prototype

    def _gather_interval_hints(self, *, fast: bool) -> Tuple[str, ...]:
        """Return interval dependency hints collected from explanation plugins."""
        if fast:
            return self._interval_plugin_hints.get("fast", ())
        ordered: List[str] = []
        seen: set[str] = set()
        for mode in ("factual", "alternative"):
            for identifier in self._interval_plugin_hints.get(mode, ()):  # noqa: B020
                if identifier not in seen:
                    ordered.append(identifier)
                    seen.add(identifier)
        return tuple(ordered)

    def _check_interval_runtime_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        identifier: str | None,
        fast: bool,
    ) -> str | None:
        """Validate interval plugin metadata for the current execution."""
        prefix = identifier or str((metadata or {}).get("name") or "<anonymous>")
        if metadata is None:
            return f"{prefix}: interval metadata unavailable"

        schema_version = metadata.get("schema_version")
        if schema_version not in (None, 1):
            return f"{prefix}: unsupported interval schema_version {schema_version}"

        modes = _coerce_string_tuple(metadata.get("modes"))
        if not modes:
            return f"{prefix}: plugin metadata missing modes declaration"
        required_mode = "regression" if "regression" in self.mode else "classification"
        if required_mode not in modes:
            declared = ", ".join(modes)
            return f"{prefix}: does not support mode '{required_mode}' (modes: {declared})"

        capabilities = set(_coerce_string_tuple(metadata.get("capabilities")))
        required_cap = (
            "interval:regression" if "regression" in self.mode else "interval:classification"
        )
        if required_cap not in capabilities:
            declared = ", ".join(sorted(capabilities)) or "<none>"
            return f"{prefix}: missing capability '{required_cap}' (capabilities: {declared})"

        if fast and not bool(metadata.get("fast_compatible")):
            return f"{prefix}: not marked fast_compatible"
        if metadata.get("requires_bins") and self.bins is None:
            return f"{prefix}: requires bins but explainer has none configured"
        return None

    def _resolve_interval_plugin(
        self,
        *,
        fast: bool,
        hints: Sequence[str] = (),
    ) -> Tuple[Any, str | None]:
        """Resolve the interval plugin for the requested execution path."""
        ensure_builtin_plugins()

        raw_override = (
            self._fast_interval_plugin_override if fast else self._interval_plugin_override
        )
        override = self._coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            identifier = getattr(override, "plugin_meta", {}).get("name")
            return override, identifier

        if isinstance(raw_override, str):
            preferred_identifier = raw_override
        else:
            key = "fast" if fast else "default"
            preferred_identifier = self._interval_preferred_identifier.get(key)
        chain = list(self._interval_plugin_fallbacks.get("fast" if fast else "default", ()))
        if hints:
            ordered = []
            seen: set[str] = set()
            for identifier in tuple(hints) + tuple(chain):
                if identifier and identifier not in seen:
                    ordered.append(identifier)
                    seen.add(identifier)
            chain = ordered

        errors: List[str] = []
        for identifier in chain:
            if is_identifier_denied(identifier):
                message = f"{identifier}: denied via CE_DENY_PLUGIN"
                if preferred_identifier == identifier:
                    raise ConfigurationError("Interval plugin override failed: " + message)
                errors.append(message)
                continue
            descriptor = find_interval_descriptor(identifier)
            plugin = None
            metadata: Mapping[str, Any] | None = None
            preferred = preferred_identifier == identifier
            if descriptor is not None:
                metadata = descriptor.metadata
                if descriptor.trusted or preferred:
                    plugin = descriptor.plugin
            if plugin is None:
                if preferred:
                    plugin = find_interval_plugin(identifier)
                else:
                    plugin = find_interval_plugin_trusted(identifier)
            if plugin is None:
                message = f"{identifier}: not registered"
                if preferred_identifier == identifier:
                    raise ConfigurationError("Interval plugin override failed: " + message)
                errors.append(message)
                continue

            meta_source = metadata or getattr(plugin, "plugin_meta", None)
            error = self._check_interval_runtime_metadata(
                meta_source,
                identifier=identifier,
                fast=fast,
            )
            if error:
                if preferred_identifier == identifier:
                    raise ConfigurationError(error)
                errors.append(error)
                continue

            plugin = self._instantiate_plugin(plugin)
            return plugin, identifier

        raise ConfigurationError(
            "Unable to resolve interval plugin for "
            + ("fast" if fast else "default")
            + " mode. Tried: "
            + ", ".join(chain or ("<none>",))
            + ("; errors: " + "; ".join(errors) if errors else "")
        )

    def _build_interval_context(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> IntervalCalibratorContext:
        """Construct the frozen interval calibrator context."""
        calibration_splits: Tuple[Any, ...] = ((self.x_cal, self.y_cal),)
        bins = {"calibration": self.bins}
        difficulty = {"estimator": self.difficulty_estimator}
        fast_flags = {"fast": fast}
        residuals: Mapping[str, Any] = {}
        key = "fast" if fast else "default"
        stored_metadata = dict(self._interval_context_metadata.get(key, {}))
        enriched_metadata = stored_metadata
        enriched_metadata.update(metadata)
        enriched_metadata.setdefault("task", self.mode)
        enriched_metadata.setdefault("mode", self.mode)
        enriched_metadata.setdefault("predict_function", getattr(self, "predict_function", None))
        enriched_metadata.setdefault("difficulty_estimator", self.difficulty_estimator)
        enriched_metadata.setdefault("explainer", self)
        enriched_metadata.setdefault("categorical_features", tuple(self.categorical_features))
        enriched_metadata.setdefault("num_features", self.num_features)
        enriched_metadata.setdefault(
            "noise_config",
            {
                "noise_type": getattr(self, "_CalibratedExplainer__noise_type", None),
                "scale_factor": getattr(self, "_CalibratedExplainer__scale_factor", None),
                "severity": getattr(self, "_CalibratedExplainer__severity", None),
                "seed": getattr(self, "seed", None),
                "rng": getattr(self, "rng", None),
            },
        )
        if fast:
            existing_fast = enriched_metadata.get("existing_fast_calibrators")
            if not existing_fast:
                stored_fast = stored_metadata.get("fast_calibrators") or stored_metadata.get(
                    "existing_fast_calibrators"
                )
                if stored_fast:
                    existing_fast = stored_fast
            if not existing_fast and isinstance(self.interval_learner, list):
                existing_fast = tuple(self.interval_learner)
            if existing_fast:
                enriched_metadata["existing_fast_calibrators"] = tuple(existing_fast)
        return IntervalCalibratorContext(
            learner=self.learner,
            calibration_splits=calibration_splits,
            bins=bins,
            residuals=residuals,
            difficulty=difficulty,
            metadata=enriched_metadata,
            fast_flags=fast_flags,
        )

    def _obtain_interval_calibrator(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> Tuple[Any, str | None]:
        """Resolve and instantiate the interval calibrator for the active mode."""
        self._ensure_interval_runtime_state()
        hints = self._gather_interval_hints(fast=fast)
        plugin, identifier = self._resolve_interval_plugin(fast=fast, hints=hints)
        context = self._build_interval_context(fast=fast, metadata=metadata)
        try:
            calibrator = plugin.create(context, fast=fast)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ConfigurationError(
                f"Interval plugin execution failed for {'fast' if fast else 'default'} mode: {exc}"
            ) from exc
        self._capture_interval_calibrators(
            context=context,
            calibrator=calibrator,
            fast=fast,
        )
        key = "fast" if fast else "default"
        self._interval_plugin_identifiers[key] = identifier
        self._telemetry_interval_sources[key] = identifier
        metadata_dict: Dict[str, Any]
        if isinstance(context.metadata, dict):
            metadata_dict = context.metadata
        else:
            metadata_dict = dict(context.metadata)
        if fast:
            if isinstance(calibrator, Sequence) and not isinstance(calibrator, (str, bytes)):
                calibrators_tuple = tuple(calibrator)
            else:
                calibrators_tuple = (calibrator,)
            metadata_dict["fast_calibrators"] = calibrators_tuple
            metadata_dict["existing_fast_calibrators"] = calibrators_tuple
        else:
            metadata_dict["calibrator"] = calibrator
        # persist captured metadata for future invocations without sharing references
        self._interval_context_metadata[key] = dict(metadata_dict)
        if metadata_dict is not context.metadata and isinstance(context.metadata, dict):
            context.metadata.update(metadata_dict)
        return calibrator, identifier

    def _capture_interval_calibrators(
        self,
        *,
        context: IntervalCalibratorContext,
        calibrator: Any,
        fast: bool,
    ) -> None:
        """Record the returned calibrator inside the interval context metadata."""
        metadata = context.metadata
        if not isinstance(metadata, dict):
            return

        if fast:
            if isinstance(calibrator, Sequence) and not isinstance(
                calibrator, (str, bytes, bytearray)
            ):
                metadata.setdefault("fast_calibrators", tuple(calibrator))
            elif calibrator is not None:
                metadata.setdefault("fast_calibrators", (calibrator,))
        else:
            metadata.setdefault("calibrator", calibrator)

    def _resolve_explanation_plugin(self, mode: str) -> Tuple[Any, str | None]:
        """Resolve or instantiate the plugin handling *mode*."""
        ensure_builtin_plugins()

        raw_override = self._explanation_plugin_overrides.get(mode)
        override = self._coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            plugin = override
            identifier = getattr(plugin, "plugin_meta", {}).get("name")
            return plugin, identifier

        preferred_identifier = raw_override if isinstance(raw_override, str) else None
        chain = self._explanation_plugin_fallbacks.get(mode, ())
        if not chain and mode == "fast":
            raise ConfigurationError(
                "Fast explanation plugin 'core.explanation.fast' is not registered. "
                "Install the external plugins extra with ``pip install \"calibrated-explanations[external-plugins]\"`` "
                "and call ``external_plugins.fast_explanations.register()`` or rerun "
                "``explain_fast(..., _use_plugin=False)`` to fall back to the legacy path."
            )
        errors: List[str] = []
        for identifier in chain:
            is_preferred = preferred_identifier is not None and identifier == preferred_identifier
            if is_identifier_denied(identifier):
                message = f"{identifier}: denied via CE_DENY_PLUGIN"
                if is_preferred:
                    raise ConfigurationError(
                        "Explanation plugin override failed: " + message
                    )
                errors.append(message)
                continue
            descriptor = find_explanation_descriptor(identifier)
            metadata: Mapping[str, Any] | None = None
            plugin = None
            if descriptor is not None:
                metadata = descriptor.metadata
                if descriptor.trusted:
                    plugin = descriptor.plugin
            if plugin is None:
                plugin = find_explanation_plugin(identifier)
            if plugin is None:
                message = f"{identifier}: not registered"
                if is_preferred:
                    raise ConfigurationError("Explanation plugin override failed: " + message)
                errors.append(message)
                continue

            meta_source = metadata or getattr(plugin, "plugin_meta", None)
            error = self._check_explanation_runtime_metadata(
                meta_source,
                identifier=identifier,
                mode=mode,
            )
            if error:
                if is_preferred:
                    raise ConfigurationError(error)
                errors.append(error)
                continue

            plugin = self._instantiate_plugin(plugin)
            try:
                supports = plugin.supports_mode
            except AttributeError as exc:
                errors.append(f"{identifier}: missing supports_mode ({exc})")
                continue
            try:
                if not supports(mode, task=self.mode):
                    errors.append(f"{identifier}: mode '{mode}' unsupported for task {self.mode}")
                    continue
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{identifier}: error during supports_mode ({exc})")
                continue
            return plugin, identifier

        raise ConfigurationError(
            "Unable to resolve explanation plugin for mode '"
            + mode
            + "'. Tried: "
            + ", ".join(chain or ("<none>",))
            + ("; errors: " + "; ".join(errors) if errors else "")
        )

    def _ensure_explanation_plugin(self, mode: str) -> Tuple[Any, str | None]:
        """Return the plugin instance for *mode*, initialising on demand."""
        if mode in self._explanation_plugin_instances:
            return self._explanation_plugin_instances[
                mode
            ], self._explanation_plugin_identifiers.get(mode)

        plugin, identifier = self._resolve_explanation_plugin(mode)
        metadata: Mapping[str, Any] | None = None
        if identifier:
            descriptor = find_explanation_descriptor(identifier)
            if descriptor:
                metadata = descriptor.metadata
                interval_dependency = metadata.get("interval_dependency")
                hints = _coerce_string_tuple(interval_dependency)
                if hints:
                    self._interval_plugin_hints[mode] = hints
            else:
                metadata = getattr(plugin, "plugin_meta", None)
        else:
            metadata = getattr(plugin, "plugin_meta", None)

        error = self._check_explanation_runtime_metadata(
            metadata,
            identifier=identifier,
            mode=mode,
        )
        if error:
            raise ConfigurationError(error)

        if metadata is not None and not identifier:
            hints = _coerce_string_tuple(metadata.get("interval_dependency"))
            if hints:
                self._interval_plugin_hints[mode] = hints
        context = self._build_explanation_context(mode, plugin, identifier)
        try:
            plugin.initialize(context)
        except Exception as exc:
            raise ConfigurationError(
                f"Explanation plugin initialisation failed for mode '{mode}': {exc}"
            ) from exc
        self._explanation_plugin_instances[mode] = plugin
        if identifier:
            self._explanation_plugin_identifiers[mode] = identifier
        self._explanation_contexts[mode] = context
        return plugin, identifier

    def _build_explanation_context(
        self, mode: str, plugin: Any, identifier: str | None
    ) -> ExplanationContext:
        """Construct the immutable context passed to explanation plugins."""
        helper_handles = {"explainer": self}
        interval_settings = {
            "dependencies": self._interval_plugin_hints.get(mode, ()),
        }
        plot_chain = self._derive_plot_chain(mode, identifier)
        self._plot_plugin_fallbacks[mode] = plot_chain
        plot_settings = {"fallbacks": plot_chain}

        monitor = self._bridge_monitors.get(mode)
        if monitor is None:
            monitor = _PredictBridgeMonitor(self._predict_bridge)
            self._bridge_monitors[mode] = monitor

        context = ExplanationContext(
            task=self.mode,
            mode=mode,
            feature_names=tuple(self.feature_names),
            categorical_features=tuple(self.categorical_features),
            categorical_labels=(
                {k: dict(v) for k, v in (self.categorical_labels or {}).items()}
                if self.categorical_labels
                else {}
            ),
            discretizer=self.discretizer,
            helper_handles=helper_handles,
            predict_bridge=monitor,
            interval_settings=interval_settings,
            plot_settings=plot_settings,
        )
        return context

    def _derive_plot_chain(self, mode: str, identifier: str | None) -> Tuple[str, ...]:
        """Return plot fallback chain seeded by plugin metadata."""
        preferred: List[str] = []
        if identifier:
            descriptor = find_explanation_descriptor(identifier)
            if descriptor:
                plot_dependency = descriptor.metadata.get("plot_dependency")
                for hint in _coerce_string_tuple(plot_dependency):
                    if hint:
                        preferred.append(hint)
        base_chain = self._plot_style_chain or ("legacy",)
        seen: set[str] = set()
        ordered: List[str] = []
        for item in tuple(preferred) + base_chain:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return tuple(ordered)

    def _build_instance_telemetry_payload(self, explanations: Any) -> Dict[str, Any]:
        """Extract telemetry details from the first explanation instance, if present."""
        try:
            first_explanation = explanations[0]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive: empty or non-indexable containers
            return {}
        builder = getattr(first_explanation, "to_telemetry", None)
        if callable(builder):
            payload = builder()
            if isinstance(payload, dict):
                return payload
        return {}

    def _infer_explanation_mode(self) -> str:
        """Infer the explanation mode based on the active discretizer."""
        if isinstance(self.discretizer, (EntropyDiscretizer, RegressorDiscretizer)):
            return "alternative"
        return "factual"

    def _invoke_explanation_plugin(
        self,
        mode: str,
        x,
        threshold,
        low_high_percentiles,
        bins,
        features_to_ignore,
        extras: Mapping[str, Any] | None = None,
    ) -> CalibratedExplanations:
        """Invoke the configured plugin for *mode* and materialise the batch."""
        plugin, _identifier = self._ensure_explanation_plugin(mode)
        request = ExplanationRequest(
            threshold=threshold,
            low_high_percentiles=tuple(low_high_percentiles)
            if low_high_percentiles is not None
            else None,
            bins=bins,
            features_to_ignore=tuple(features_to_ignore or []),
            extras=dict(extras or {}),
        )
        monitor = self._bridge_monitors.get(mode)
        if monitor is not None:
            monitor.reset_usage()
        try:
            batch = plugin.explain_batch(x, request)
        except Exception as exc:
            raise ConfigurationError(
                f"Explanation plugin execution failed for mode '{mode}': {exc}"
            ) from exc
        try:
            validate_explanation_batch(
                batch,
                expected_mode=mode,
                expected_task=self.mode,
            )
        except Exception as exc:
            raise ConfigurationError(
                f"Explanation plugin for mode '{mode}' returned an invalid batch: {exc}"
            ) from exc
        metadata = batch.collection_metadata
        metadata.setdefault("task", self.mode)
        interval_key = "fast" if mode == "fast" else "default"
        interval_source = self._telemetry_interval_sources.get(interval_key)
        if interval_source:
            metadata["interval_source"] = interval_source
            metadata.setdefault("proba_source", interval_source)
        metadata.setdefault(
            "interval_dependencies",
            tuple(self._interval_plugin_hints.get(mode, ())),
        )
        preprocessor_meta = self.preprocessor_metadata
        if preprocessor_meta:
            metadata.setdefault("preprocessor", preprocessor_meta)
        plot_chain = self._plot_plugin_fallbacks.get(mode)
        if plot_chain:
            metadata.setdefault("plot_fallbacks", tuple(plot_chain))
            metadata.setdefault("plot_source", plot_chain[0])
        telemetry_payload = {
            "mode": mode,
            "task": self.mode,
            "interval_source": interval_source,
            "proba_source": metadata.get("proba_source"),
            "plot_source": metadata.get("plot_source"),
            "plot_fallbacks": tuple(plot_chain or ()),
        }
        if preprocessor_meta:
            telemetry_payload["preprocessor"] = preprocessor_meta
        self._last_telemetry = dict(telemetry_payload)
        if monitor is not None and not monitor.used:
            raise ConfigurationError(
                "Explanation plugin for mode '"
                + mode
                + "' did not use the calibrated predict bridge"
            )
        container_cls = batch.container_cls
        if hasattr(container_cls, "from_batch"):
            result = container_cls.from_batch(batch)
            instance_payload = self._build_instance_telemetry_payload(result)
            if instance_payload:
                telemetry_payload.update(instance_payload)
                self._last_telemetry.update(instance_payload)
            with contextlib.suppress(Exception):
                result.telemetry = dict(telemetry_payload)
            self.latest_explanation = result
            self._last_explanation_mode = mode
            return result
        raise ConfigurationError("Explanation plugin returned a batch that cannot be materialised")

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
        return self.__X_cal if isinstance(self._X_cal[0], dict) else self._X_cal

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
        if safe_isinstance(value, "pandas.core.frame.DataFrame"):
            value = value.values

        if len(value.shape) == 1:
            value = value.reshape(1, -1)

        self._X_cal = value

        if isinstance(self._X_cal[0], dict):
            self.__X_cal = np.array([[x[f] for f in x] for x in self._X_cal])
        self._invalidate_calibration_summaries()

    @property
    def y_cal(self):
        """Get the calibration target data.

        Returns
        -------
        array-like
            The calibration target data.
        """
        return self._y_cal

    @y_cal.setter
    def y_cal(self, value):
        """Set the calibration target data.

        Parameters
        ----------
        value : array-like of shape (n_samples,)
            The new calibration target data.
        """
        if safe_isinstance(value, "pandas.core.frame.DataFrame"):
            self._y_cal = np.asarray(value.values)
        else:
            if len(value.shape) == 2 and value.shape[1] == 1:
                value = value.ravel()
            self._y_cal = np.asarray(value)

    def append_cal(self, x, y):
        """Append new calibration data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The new calibration input data to append.
        y : array-like of shape (n_samples,)
            The new calibration target data to append.
        """
        if x.shape[1] != self.num_features:
            raise DataShapeError("Number of features must match existing calibration data")
        self.x_cal = np.vstack((self.x_cal, x))
        self.y_cal = np.concatenate((self.y_cal, y))

    def _invalidate_calibration_summaries(self) -> None:
        """Drop cached calibration summaries used during explanation."""

        self._categorical_value_counts_cache = None
        self._numeric_sorted_cache = None
        self._calibration_summary_shape = None

    def _get_calibration_summaries(
        self, x_cal_np: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
        """Return cached categorical counts and sorted numeric calibration values."""

        if x_cal_np is None:
            x_cal_np = np.asarray(self.x_cal)
        shape = getattr(x_cal_np, "shape", None)
        if (
            self._categorical_value_counts_cache is None
            or self._numeric_sorted_cache is None
            or self._calibration_summary_shape != shape
        ):
            categorical_value_counts: Dict[int, Dict[Any, int]] = {}
            numeric_sorted_cache: Dict[int, np.ndarray] = {}
            if x_cal_np.size:
                categorical_features = tuple(int(f) for f in self.categorical_features)
                for f_cat in categorical_features:
                    unique_vals, unique_counts = np.unique(
                        x_cal_np[:, f_cat], return_counts=True
                    )
                    categorical_value_counts[int(f_cat)] = {
                        val: int(cnt)
                        for val, cnt in zip(
                            unique_vals.tolist(), unique_counts.tolist()
                        )
                    }
                numeric_features = [
                    f for f in range(self.num_features) if f not in categorical_features
                ]
                for f_num in numeric_features:
                    numeric_sorted_cache[f_num] = np.sort(
                        np.asarray(x_cal_np[:, f_num])
                    )
            self._categorical_value_counts_cache = categorical_value_counts
            self._numeric_sorted_cache = numeric_sorted_cache
            self._calibration_summary_shape = shape
        assert self._categorical_value_counts_cache is not None
        assert self._numeric_sorted_cache is not None
        return self._categorical_value_counts_cache, self._numeric_sorted_cache

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
            # Phase 1A delegation: update interval learner via helper
            from .calibration_helpers import update_interval_learner as _upd_il

            _upd_il(self, xs, ys, bins=bins)
        else:
            from .calibration_helpers import initialize_interval_learner as _init_il

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

    # pylint: disable=invalid-name, too-many-return-statements
    def _predict_impl(
        self,
        x,
        threshold=None,  # The same meaning as threshold has for cps in crepes.
        low_high_percentiles=(5, 95),
        classes=None,
        bins=None,
        feature=None,
        **kwargs,
    ):
        """Execute the internal prediction method for classification and regression cases.

        For classification:
        - Returns probabilities and intervals for binary/multiclass
        - Handles Mondrian categories via bins parameter

        For regression:
        - Returns predictions and uncertainty intervals
        - Can return probability predictions when threshold is provided

        Parameters
        ----------
        x : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic explanations for regression.
        low_high_percentiles : a tuple of floats, default=(5, 95)
            The low and high percentile used to calculate the interval. Applicable to regression.
        classes : None or array-like of shape (n_samples,), default=None
            The classes predicted for the original instance. None if not multiclass or regression.

        Raises
        ------
        ValueError: The length of the threshold-parameter must be either a constant or the same as the number of
            instances in x.

        Returns
        -------
        predict : ndarray of shape (n_samples,)
            The prediction for the test data. For classification, this is the regularized probability
            of the positive class, derived using the intervals from VennAbers. For regression, this is the
            median prediction from the ConformalPredictiveSystem.
        low : ndarray of shape (n_samples,)
            The lower bound of the prediction interval. For classification, this is derived using
            VennAbers. For regression, this is the lower percentile given as parameter, derived from the
            ConformalPredictiveSystem.
        high : ndarray of shape (n_samples,)
            The upper bound of the prediction interval. For classification, this is derived using
            VennAbers. For regression, this is the upper percentile given as parameter, derived from the
            ConformalPredictiveSystem.
        classes : ndarray of shape (n_samples,)
            The classes predicted for the original instance. None if not multiclass or regression.
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        """
        # strip plotting-only keys that callers may pass
        kwargs.pop("show", None)
        kwargs.pop("style_override", None)
        if not self.__initialized:
            raise NotFittedError("The learner must be initialized before calling predict.")
        if feature is None and self.is_fast():
            feature = self.num_features  # Use the calibrator defined using X_cal
        if self.mode == "classification":
            if self.is_multiclass():
                if self.is_fast():
                    predict, low, high, new_classes = self.interval_learner[feature].predict_proba(
                        x, output_interval=True, classes=classes, bins=bins
                    )
                else:
                    predict, low, high, new_classes = self.interval_learner.predict_proba(
                        x, output_interval=True, classes=classes, bins=bins
                    )
                if classes is None:
                    return (
                        [predict[i, c] for i, c in enumerate(new_classes)],
                        [low[i, c] for i, c in enumerate(new_classes)],
                        [high[i, c] for i, c in enumerate(new_classes)],
                        new_classes,
                    )
                if type(classes) not in (list, np.ndarray):
                    classes = [classes]
                return [predict[i, c] for i, c in enumerate(classes)], low, high, None

            if self.is_fast():
                predict, low, high = self.interval_learner[feature].predict_proba(
                    x, output_interval=True, bins=bins
                )
            else:
                predict, low, high = self.interval_learner.predict_proba(
                    x, output_interval=True, bins=bins
                )
            return predict[:, 1], low, high, None
        if "regression" in self.mode:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            if threshold is None:  # normal regression
                if not (low_high_percentiles[0] <= low_high_percentiles[1]):
                    raise ValidationError(
                        "The low percentile must be smaller than (or equal to) the high percentile."
                    )
                if not (
                    (
                        (low_high_percentiles[0] > 0 and low_high_percentiles[0] <= 50)
                        and (low_high_percentiles[1] >= 50 and low_high_percentiles[1] < 100)
                    )
                    or low_high_percentiles[0] == -np.inf
                    or low_high_percentiles[1] == np.inf
                    and not (
                        low_high_percentiles[0] == -np.inf and low_high_percentiles[1] == np.inf
                    )
                ):
                    raise ValidationError(
                        "The percentiles must be between 0 and 100 (exclusive). \
                            The lower percentile can be -np.inf and the higher percentile can \
                            be np.inf (but not at the same time) to allow one-sided intervals."
                    )
                low = (
                    [low_high_percentiles[0], 50]
                    if low_high_percentiles[0] != -np.inf
                    else [50, 50]
                )
                high = (
                    [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]
                )

                try:
                    if self.is_fast():
                        return self.interval_learner[feature].predict_uncertainty(
                            x, low_high_percentiles, bins=bins
                        )
                    return self.interval_learner.predict_uncertainty(
                        x, low_high_percentiles, bins=bins
                    )
                except Exception:  # typically crepes broadcasting/shape errors
                    if self.suppress_crepes_errors:
                        # Log and return placeholder arrays (caller should handle downstream)
                        _warnings.warn(
                            "crepes produced an unexpected result (likely too-small calibration set); returning zeros as a degraded fallback.",
                            UserWarning,
                            stacklevel=2,
                        )
                        n = x.shape[0]
                        # produce zero-length or zero arrays consistent with expected shape
                        return np.zeros(n), np.zeros(n), np.zeros(n), None
                    # Preserve prior behavior: re-raise the original exception so callers/tests
                    # see the original error (instead of converting it to DataShapeError).
                    raise

            # regression with threshold condition
            assert_threshold(threshold, x)
            try:
                if self.is_fast():
                    return self.interval_learner[feature].predict_probability(
                        x, threshold, bins=bins
                    )
                # pylint: disable=unexpected-keyword-arg
                return self.interval_learner.predict_probability(x, threshold, bins=bins)
            except Exception as exc:
                if self.suppress_crepes_errors:
                    _warnings.warn(
                        "crepes produced an unexpected result while computing probabilities; returning zeros as a degraded fallback.",
                        UserWarning,
                        stacklevel=2,
                    )
                    n = x.shape[0]
                    return np.zeros(n), np.zeros(n), np.zeros(n), None
                    # Re-raise as a clearer DataShapeError with guidance
                    raise DataShapeError(
                        "Error while computing prediction intervals from the underlying crepes library. "
                        "This commonly occurs when the calibration set is too small for the requested percentiles. "
                        "Consider using a larger calibration set, or instantiate the explainer with suppress_crepes_errors=True to return a degraded fallback. "
                        f"(original error: {exc})"
                    ) from exc

        return None, None, None, None  # Should never happen

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
        """Cache-aware wrapper around :meth:`_predict_impl`."""

        cache = getattr(self, "_perf_cache", None)
        cache_enabled = getattr(cache, "enabled", False)
        key_parts = None
        if cache_enabled:
            x_arr = np.asarray(x)
            key_parts = (
                ("mode", self.mode),
                ("feature", feature),
                ("shape", x_arr.shape),
                ("x", x_arr),
                ("threshold", np.asarray(threshold) if threshold is not None else None),
                ("percentiles", tuple(low_high_percentiles)),
                ("classes", np.asarray(classes) if classes is not None else None),
                ("bins", np.asarray(bins) if bins is not None else None),
                ("kwargs", dict(kwargs) if kwargs else {}),
            )
            cached = cache.get(stage="predict", parts=key_parts)
            if cached is not None:
                return cached

        result = self._predict_impl(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            feature=feature,
            **kwargs,
        )

        if cache_enabled and key_parts is not None:
            cache.set(stage="predict", parts=key_parts, value=result)
        return result

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

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in x.

        Returns
        -------
        CalibratedExplanations : :class:`.CalibratedExplanations`
            A `CalibratedExplanations` containing one :class:`.FactualExplanation` for each instance.
        """
        discretizer = "binaryRegressor" if "regression" in self.mode else "binaryEntropy"
        self.set_discretizer(discretizer, features_to_ignore=features_to_ignore)
        return self.explain(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
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

        Raises
        ------
        ValueError: The number of features in the test data must be the same as in the calibration data.
        Warning: The threshold-parameter is only supported for mode='regression'.
        ValueError: The length of the threshold parameter must be either a constant or the same as the number of
            instances in x.

        Returns
        -------
        AlternativeExplanations : :class:`.AlternativeExplanations`
            An `AlternativeExplanations` containing one :class:`.AlternativeExplanation` for each instance.

        Notes
        -----
        The `explore_alternatives` will eventually be used instead of the `explain_counterfactual` method.
        """
        discretizer = "regressor" if "regression" in self.mode else "entropy"
        self.set_discretizer(discretizer, features_to_ignore=features_to_ignore)
        # At runtime, explain() will return an AlternativeExplanations when an alternative discretizer is set.
        # Help mypy with a narrow cast here without changing behavior.
        return self.explain(
            x,
            threshold,
            low_high_percentiles,
            bins,
            features_to_ignore,
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
    ) -> CalibratedExplanations:
        """Generate explanations for test instances by analyzing feature effects.

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
        if _use_plugin:
            mode = self._infer_explanation_mode()
            return self._invoke_explanation_plugin(
                mode,
                x,
                threshold,
                low_high_percentiles,
                bins,
                features_to_ignore,
                extras={"mode": mode},
            )

        # Track total explanation time
        total_time = time()

        if features_to_ignore is None:
            features_to_ignore_array = np.asarray(self.features_to_ignore, dtype=int)
        else:
            features_to_ignore_array = np.asarray(
                np.union1d(self.features_to_ignore, features_to_ignore), dtype=int
            )
        features_to_ignore_set = set(features_to_ignore_array.tolist())

        # Validate inputs and initialize explanation object
        x = self._validate_and_prepare_input(x)
        explanation = self._initialize_explanation(
            x, low_high_percentiles, threshold, bins, features_to_ignore_array
        )

        instance_time = time()

        # Step 1: Get predictions for original test instances
        (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
        ) = self._explain_predict_step(
            x, threshold, low_high_percentiles, bins, features_to_ignore_array
        )

        # Step 2: Initialize data structures to store feature-level results
        n_instances = x.shape[0]
        num_features = self.num_features

        weights_predict = np.zeros((n_instances, num_features))
        weights_low = np.zeros((n_instances, num_features))
        weights_high = np.zeros((n_instances, num_features))
        predict_matrix = np.zeros((n_instances, num_features))
        low_matrix = np.zeros((n_instances, num_features))
        high_matrix = np.zeros((n_instances, num_features))

        rule_values: List[Dict[int, Any]] = [dict() for _ in range(n_instances)]
        instance_binned: List[Dict[str, Dict[int, Any]]] = [
            {
                "predict": {},
                "low": {},
                "high": {},
                "current_bin": {},
                "rule_values": {},
                "counts": {},
                "fractions": {},
            }
            for _ in range(n_instances)
        ]

        # Step 3: Process each feature to analyze its effects
        perturbed_feature = np.asarray(perturbed_feature, dtype=object)
        x_cal_np = np.asarray(x_cal)
        if perturbed_feature.size:
            feature_ids = perturbed_feature[:, 0].astype(int)
            feature_index_lists: Dict[int, List[int]] = defaultdict(list)
            for idx, fid in enumerate(feature_ids):
                feature_index_lists[int(fid)].append(idx)
            feature_index_map = {
                fid: np.asarray(indices, dtype=int) for fid, indices in feature_index_lists.items()
            }
        else:
            feature_index_map = {}

        categorical_value_counts, numeric_sorted_cache = self._get_calibration_summaries(
            x_cal_np
        )

        for f in range(self.num_features):
            if f in features_to_ignore_set:
                for i in range(n_instances):
                    rule_values[i][f] = (self.feature_values[f], x[i, f], x[i, f])
                    instance_binned[i]["predict"][f] = predict[i]
                    instance_binned[i]["low"][f] = low[i]
                    instance_binned[i]["high"][f] = high[i]
                    instance_binned[i]["current_bin"][f] = -1
                    instance_binned[i]["counts"][f] = np.array([], dtype=float)
                    instance_binned[i]["fractions"][f] = np.array([], dtype=float)
                continue

            feature_indices = feature_index_map.get(f)
            if feature_indices is None or feature_indices.size == 0:
                for i in range(n_instances):
                    rule_values[i][f] = (self.feature_values[f], x[i, f], x[i, f])
                    instance_binned[i]["predict"][f] = predict[i]
                    instance_binned[i]["low"][f] = low[i]
                    instance_binned[i]["high"][f] = high[i]
                    instance_binned[i]["current_bin"][f] = -1
                    instance_binned[i]["counts"][f] = np.array([], dtype=float)
                    instance_binned[i]["fractions"][f] = np.array([], dtype=float)
                continue

            feature_slice = perturbed_feature[feature_indices]
            feature_predict_local = predict[feature_indices]
            feature_low_local = low[feature_indices]
            feature_high_local = high[feature_indices]
            feature_instances = feature_slice[:, 1].astype(int)
            unique_instances = np.unique(feature_instances)

            slice_instances = feature_instances

            if f in self.categorical_features:
                grouped: Dict[int, Dict[Tuple[Any, Any], np.ndarray]] = {}
                for rel_idx, row in enumerate(feature_slice):
                    inst = int(row[1])
                    key = (row[2], row[3])
                    grouped.setdefault(inst, {}).setdefault(key, []).append(rel_idx)
                for inst, mapping in grouped.items():
                    for key, rel_list in mapping.items():
                        grouped[inst][key] = np.asarray(rel_list, dtype=int)
                feature_values = self.feature_values[f]
                value_counts_cache: Dict[Any, int] = categorical_value_counts.get(int(f), {})
                for inst in unique_instances:
                    i = int(inst)
                    current_bin = -1
                    average_predict = np.zeros(len(feature_values))
                    low_predict = np.zeros(len(feature_values))
                    high_predict = np.zeros(len(feature_values))
                    counts = np.zeros(len(feature_values))
                    for bin_index, value in enumerate(feature_values):
                        rel_indices = grouped.get(i, {}).get((value, None))
                        rel_indices = rel_indices if rel_indices is not None else np.empty((0,), dtype=int)
                        if x[i, f] == value:
                            current_bin = bin_index
                        counts[bin_index] = value_counts_cache.get(value, 0)
                        if rel_indices.size:
                            average_predict[bin_index] = safe_mean(feature_predict_local[rel_indices])
                            low_predict[bin_index] = safe_mean(feature_low_local[rel_indices])
                            high_predict[bin_index] = safe_mean(feature_high_local[rel_indices])
                        else:
                            average_predict[bin_index] = 0
                            low_predict[bin_index] = 0
                            high_predict[bin_index] = 0
                    mask = np.ones_like(average_predict, dtype=bool)
                    if 0 <= current_bin < mask.size:
                        mask[current_bin] = False
                    uncovered = np.nonzero(mask)[0]
                    counts_uncovered = counts[mask]
                    total_counts = counts_uncovered.sum() if uncovered.size else 0
                    fractions = (
                        counts_uncovered / total_counts
                        if uncovered.size and total_counts
                        else np.zeros(uncovered.size, dtype=float)
                    )
                    instance_binned[i]["predict"][f] = average_predict
                    instance_binned[i]["low"][f] = low_predict
                    instance_binned[i]["high"][f] = high_predict
                    instance_binned[i]["current_bin"][f] = current_bin
                    instance_binned[i]["counts"][f] = counts
                    instance_binned[i]["fractions"][f] = fractions
                    rule_values[i][f] = (feature_values, x[i, f], x[i, f])
                    if uncovered.size == 0:
                        predict_matrix[i, f] = 0
                        low_matrix[i, f] = 0
                        high_matrix[i, f] = 0
                        weights_predict[i, f] = 0
                        weights_low[i, f] = 0
                        weights_high[i, f] = 0
                    else:
                        predict_matrix[i, f] = safe_mean(average_predict[mask])
                        low_matrix[i, f] = safe_mean(low_predict[mask])
                        high_matrix[i, f] = safe_mean(high_predict[mask])
                        weights_predict[i, f] = self._assign_weight(
                            predict_matrix[i, f], prediction["predict"][i]
                        )
                        tmp_low = self._assign_weight(low_matrix[i, f], prediction["predict"][i])
                        tmp_high = self._assign_weight(high_matrix[i, f], prediction["predict"][i])
                        weights_low[i, f] = np.min([tmp_low, tmp_high])
                        weights_high[i, f] = np.max([tmp_low, tmp_high])
            else:
                slice_bins = np.array(feature_slice[:, 2], dtype=int)
                slice_flags = np.asarray(feature_slice[:, 3], dtype=object)
                numeric_grouped: Dict[Tuple[int, int, Any], np.ndarray] = {}
                for rel_idx, inst in enumerate(slice_instances):
                    key = (int(inst), int(slice_bins[rel_idx]), slice_flags[rel_idx])
                    numeric_grouped.setdefault(key, []).append(rel_idx)
                for key, rel_list in numeric_grouped.items():
                    numeric_grouped[key] = np.asarray(rel_list, dtype=int)

                sorted_cal = numeric_sorted_cache.get(f)
                if sorted_cal is None:
                    feature_values = np.unique(np.asarray(x_cal_np[:, f]))
                    sorted_cal = np.sort(feature_values)
                else:
                    feature_values = np.unique(sorted_cal)
                lower_boundary = np.array(rule_boundaries[:, f, 0], copy=True)
                upper_boundary = np.array(rule_boundaries[:, f, 1], copy=True)
                if feature_values.size:
                    min_val = np.min(feature_values)
                    max_val = np.max(feature_values)
                    lower_boundary = np.where(min_val < lower_boundary, lower_boundary, -np.inf)
                    upper_boundary = np.where(max_val > upper_boundary, upper_boundary, np.inf)
                rule_boundaries[:, f, 0] = lower_boundary
                rule_boundaries[:, f, 1] = upper_boundary

                avg_predict_map: Dict[int, np.ndarray] = {}
                low_predict_map: Dict[int, np.ndarray] = {}
                high_predict_map: Dict[int, np.ndarray] = {}
                counts_map: Dict[int, np.ndarray] = {}
                rule_value_map: Dict[int, List[np.ndarray]] = {}
                for i in range(len(x)):
                    num_bins = 1 + (1 if lower_boundary[i] != -np.inf else 0)
                    num_bins += 1 if upper_boundary[i] != np.inf else 0
                    avg_predict_map[i] = np.zeros(num_bins)
                    low_predict_map[i] = np.zeros(num_bins)
                    high_predict_map[i] = np.zeros(num_bins)
                    counts_map[i] = np.zeros(num_bins)
                    rule_value_map[i] = []

                bin_value = np.zeros(len(x), dtype=int)
                current_bin = -np.ones(len(x), dtype=int)

                unique_lower, lower_inverse = np.unique(lower_boundary, return_inverse=True)
                unique_upper, upper_inverse = np.unique(upper_boundary, return_inverse=True)
                lower_groups = {
                    idx: np.flatnonzero(lower_inverse == idx) for idx in range(unique_lower.size)
                }
                upper_groups = {
                    idx: np.flatnonzero(upper_inverse == idx) for idx in range(unique_upper.size)
                }
                lower_cache = {
                    val: 0 if val == -np.inf else int(np.searchsorted(sorted_cal, val, side="left"))
                    for val in unique_lower
                }
                upper_cache = {
                    val: 0 if val == np.inf else int(
                        sorted_cal.size - np.searchsorted(sorted_cal, val, side="right")
                    )
                    for val in unique_upper
                }
                bounds_matrix = np.column_stack((lower_boundary, upper_boundary))
                unique_bounds, bound_inverse = np.unique(bounds_matrix, axis=0, return_inverse=True)
                between_cache: Dict[int, int] = {}
                for idx_bound, (lb, ub) in enumerate(unique_bounds):
                    left = 0 if lb == -np.inf else int(np.searchsorted(sorted_cal, lb, side="left"))
                    right = (
                        sorted_cal.size
                        if ub == np.inf
                        else int(np.searchsorted(sorted_cal, ub, side="right"))
                    )
                    between_cache[idx_bound] = right - left

                for j, val in enumerate(unique_lower):
                    values_tuple = lesser_values[f].get(j)
                    if not values_tuple or values_tuple[0].size == 0:
                        continue
                    for idx in lower_groups.get(j, []):
                        inst = int(idx)
                        rel_indices = numeric_grouped.get((inst, j, True), np.empty((0,), dtype=int))
                        avg_predict_map[inst][bin_value[inst]] = (
                            safe_mean(feature_predict_local[rel_indices]) if rel_indices.size else 0
                        )
                        low_predict_map[inst][bin_value[inst]] = (
                            safe_mean(feature_low_local[rel_indices]) if rel_indices.size else 0
                        )
                        high_predict_map[inst][bin_value[inst]] = (
                            safe_mean(feature_high_local[rel_indices]) if rel_indices.size else 0
                        )
                        counts_map[inst][bin_value[inst]] = lower_cache.get(val, 0)
                        rule_value_map[inst].append(values_tuple[0])
                        bin_value[inst] += 1

                for j, val in enumerate(unique_upper):
                    values_tuple = greater_values[f].get(j)
                    if not values_tuple or values_tuple[0].size == 0:
                        continue
                    for idx in upper_groups.get(j, []):
                        inst = int(idx)
                        rel_indices = numeric_grouped.get((inst, j, False), np.empty((0,), dtype=int))
                        avg_predict_map[inst][bin_value[inst]] = (
                            safe_mean(feature_predict_local[rel_indices]) if rel_indices.size else 0
                        )
                        low_predict_map[inst][bin_value[inst]] = (
                            safe_mean(feature_low_local[rel_indices]) if rel_indices.size else 0
                        )
                        high_predict_map[inst][bin_value[inst]] = (
                            safe_mean(feature_high_local[rel_indices]) if rel_indices.size else 0
                        )
                        counts_map[inst][bin_value[inst]] = upper_cache.get(val, 0)
                        rule_value_map[inst].append(values_tuple[0])
                        bin_value[inst] += 1

                for inst in range(len(x)):
                    current_index = bin_value[inst]
                    for j in range(unique_bounds.shape[0]):
                        rel_indices = numeric_grouped.get(
                            (inst, j, None), np.empty((0,), dtype=int)
                        )
                        avg_predict_map[inst][current_index] = (
                            safe_mean(feature_predict_local[rel_indices]) if rel_indices.size else 0
                        )
                        low_predict_map[inst][current_index] = (
                            safe_mean(feature_low_local[rel_indices]) if rel_indices.size else 0
                        )
                        high_predict_map[inst][current_index] = (
                            safe_mean(feature_high_local[rel_indices]) if rel_indices.size else 0
                        )
                        counts_map[inst][current_index] = between_cache.get(j, 0)
                        rule_entry = covered_values[f].get(j)
                        if rule_entry is None:
                            rule_entry = covered_values[f].get(inst)
                        rule_value_map[inst].append(
                            rule_entry[0] if rule_entry is not None else np.array([])
                        )
                        current_bin[inst] = current_index

                for idx in range(len(x)):
                    inst = int(idx)
                    rule_values[inst][f] = (rule_value_map[inst], x[inst, f], x[inst, f])
                    mask = np.ones_like(avg_predict_map[inst], dtype=bool)
                    if 0 <= current_bin[inst] < mask.size:
                        mask[current_bin[inst]] = False
                    uncovered = np.nonzero(mask)[0]
                    counts_uncovered = counts_map[inst][mask]
                    total_counts = counts_uncovered.sum() if uncovered.size else 0
                    fractions = (
                        counts_uncovered / total_counts
                        if uncovered.size and total_counts
                        else np.zeros(uncovered.size, dtype=float)
                    )
                    instance_binned[inst]["predict"][f] = avg_predict_map[inst]
                    instance_binned[inst]["low"][f] = low_predict_map[inst]
                    instance_binned[inst]["high"][f] = high_predict_map[inst]
                    instance_binned[inst]["current_bin"][f] = current_bin[inst]
                    instance_binned[inst]["counts"][f] = counts_map[inst]
                    instance_binned[inst]["fractions"][f] = fractions
                    if uncovered.size == 0:
                        predict_matrix[inst, f] = 0
                        low_matrix[inst, f] = 0
                        high_matrix[inst, f] = 0
                        weights_predict[inst, f] = 0
                        weights_low[inst, f] = 0
                        weights_high[inst, f] = 0
                    else:
                        predict_matrix[inst, f] = safe_mean(avg_predict_map[inst][mask])
                        low_matrix[inst, f] = safe_mean(low_predict_map[inst][mask])
                        high_matrix[inst, f] = safe_mean(high_predict_map[inst][mask])
                        weights_predict[inst, f] = self._assign_weight(
                            predict_matrix[inst, f], prediction["predict"][inst]
                        )
                        tmp_low = self._assign_weight(low_matrix[inst, f], prediction["predict"][inst])
                        tmp_high = self._assign_weight(high_matrix[inst, f], prediction["predict"][inst])
                        weights_low[inst, f] = np.min([tmp_low, tmp_high])
                        weights_high[inst, f] = np.max([tmp_low, tmp_high])
        binned_predict: Dict[str, List[Any]] = {
            "predict": [],
            "low": [],
            "high": [],
            "current_bin": [],
            "rule_values": [],
            "counts": [],
            "fractions": [],
        }
        feature_weights: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
        feature_predict: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}

        for i in range(n_instances):
            binned_predict["predict"].append(instance_binned[i]["predict"])
            binned_predict["low"].append(instance_binned[i]["low"])
            binned_predict["high"].append(instance_binned[i]["high"])
            binned_predict["current_bin"].append(instance_binned[i]["current_bin"])
            binned_predict["rule_values"].append(rule_values[i])
            binned_predict["counts"].append(instance_binned[i]["counts"])
            binned_predict["fractions"].append(instance_binned[i]["fractions"])

            feature_weights["predict"].append(weights_predict[i].copy())
            feature_weights["low"].append(weights_low[i].copy())
            feature_weights["high"].append(weights_high[i].copy())

            feature_predict["predict"].append(predict_matrix[i].copy())
            feature_predict["low"].append(low_matrix[i].copy())
            feature_predict["high"].append(high_matrix[i].copy())
        elapsed_time = time() - instance_time
        list_instance_time = [elapsed_time / n_instances for _ in range(n_instances)]

        explanation = explanation.finalize(
            binned_predict,
            feature_weights,
            feature_predict,
            prediction,
            instance_time=list_instance_time,
            total_time=total_time,
        )
        self.latest_explanation = explanation
        self._last_explanation_mode = self._infer_explanation_mode()
        return explanation

    def _validate_and_prepare_input(self, x):
        """Delegate to extracted helper (Phase 1A)."""
        from .prediction_helpers import validate_and_prepare_input as _vh

        return _vh(self, x)

    def _initialize_explanation(self, x, low_high_percentiles, threshold, bins, features_to_ignore):
        """Delegate to extracted helper (Phase 1A)."""
        from .prediction_helpers import initialize_explanation as _ih

        return _ih(self, x, low_high_percentiles, threshold, bins, features_to_ignore)

    def _explain_predict_step(self, x, threshold, low_high_percentiles, bins, features_to_ignore):
        # Phase 1A: delegate initial setup to prediction_helpers to lock behavior
        from .prediction_helpers import explain_predict_step as _eps

        (
            _base_predict,
            _base_low,
            _base_high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
            perturbed_threshold,
            perturbed_bins,
            perturbed_x,
            perturbed_class,
        ) = _eps(self, x, threshold, low_high_percentiles, bins, features_to_ignore)

        perturbed_x_parts: List[np.ndarray] = []
        perturbed_feature_parts: List[np.ndarray] = []
        perturbed_bins_parts: List[np.ndarray] = []
        perturbed_class_parts: List[np.ndarray] = []
        threshold_items: List[Any] = []

        features_to_ignore_array = (
            np.asarray(features_to_ignore, dtype=int)
            if features_to_ignore is not None
            else np.empty((0,), dtype=int)
        )
        features_to_ignore_set = set(int(f) for f in features_to_ignore_array.tolist())

        # Sub-step 1.b: prepare and add the perturbed test instances (unchanged logic)
        # pylint: disable=too-many-nested-blocks
        for f in range(self.num_features):
            if f in features_to_ignore_set:
                continue
            if f in self.categorical_features:
                feature_values = np.asarray(self.feature_values[f])
                if feature_values.size == 0:
                    continue

                num_instances = x.shape[0]
                num_values = int(feature_values.size)

                # Assemble the perturbations for this categorical feature in a single
                # tiled matrix to avoid repeatedly copying the full feature matrix.
                tiled_x = np.tile(x, (num_values, 1))
                tiled_x[:, f] = np.repeat(feature_values, num_instances)
                perturbed_x_parts.append(tiled_x)

                feature_info = np.empty((num_instances * num_values, 4), dtype=object)
                feature_info[:, 0] = f
                feature_info[:, 1] = np.tile(np.arange(num_instances), num_values)
                feature_info[:, 2] = np.repeat(feature_values, num_instances)
                feature_info[:, 3] = None
                perturbed_feature_parts.append(feature_info)

                if bins is not None:
                    bins_array = np.array(bins, copy=True)
                    if bins_array.ndim == 0:
                        perturbed_bins_parts.append(np.repeat(bins_array, num_values))
                    else:
                        tile_shape = (num_values,) + (1,) * (bins_array.ndim - 1)
                        perturbed_bins_parts.append(np.tile(bins_array, tile_shape))

                predict_array = np.array(prediction["predict"], copy=True)
                if predict_array.ndim == 0:
                    perturbed_class_parts.append(np.repeat(predict_array, num_values))
                else:
                    tile_shape = (num_values,) + (1,) * (predict_array.ndim - 1)
                    perturbed_class_parts.append(np.tile(predict_array, tile_shape))

                if threshold is not None and isinstance(threshold, (list, np.ndarray)):
                    for _ in range(num_values):
                        threshold_items.extend(threshold[i] for i in range(num_instances))
            else:
                x_copy = x.copy()
                feature_values = np.unique(np.array(x_cal[:, f]))
                lower_boundary = np.array(rule_boundaries[:, f, 0], copy=True)
                upper_boundary = np.array(rule_boundaries[:, f, 1], copy=True)

                if feature_values.size:
                    has_lesser = (
                        feature_values[np.newaxis, :] < lower_boundary[:, np.newaxis]
                    ).any(axis=1)
                    has_greater = (
                        feature_values[np.newaxis, :] > upper_boundary[:, np.newaxis]
                    ).any(axis=1)
                else:
                    has_lesser = np.zeros(lower_boundary.shape[0], dtype=bool)
                    has_greater = np.zeros(upper_boundary.shape[0], dtype=bool)
                lower_boundary = np.where(has_lesser, lower_boundary, -np.inf)
                upper_boundary = np.where(has_greater, upper_boundary, np.inf)
                rule_boundaries[:, f, 0] = lower_boundary
                rule_boundaries[:, f, 1] = upper_boundary

                lesser_values[f] = {}
                greater_values[f] = {}
                covered_values[f] = {}
                bins_array = np.asarray(bins) if bins is not None else None
                classes_array = np.asarray(prediction["classes"])

                for j, val in enumerate(np.unique(lower_boundary)):
                    lesser_values[f][j] = (np.unique(self.__get_lesser_values(f, val)), val)
                    indices = np.where(lower_boundary == val)[0]
                    values = lesser_values[f][j][0]
                    if values.size == 0 or indices.size == 0:
                        continue

                    base_slice = np.array(x[indices, :], copy=True)
                    num_instances_subset = base_slice.shape[0]
                    num_values = values.size

                    tiled_x = np.tile(base_slice, (num_values, 1))
                    tiled_x[:, f] = np.repeat(values, num_instances_subset)
                    perturbed_x_parts.append(tiled_x)

                    feature_info = np.empty((num_instances_subset * num_values, 4), dtype=object)
                    feature_info[:, 0] = f
                    feature_info[:, 1] = np.tile(indices, num_values)
                    feature_info[:, 2] = j
                    feature_info[:, 3] = True
                    perturbed_feature_parts.append(feature_info)

                    if bins_array is not None:
                        bins_subset = np.array(bins_array[indices], copy=True)
                        tile_shape = (num_values,) + (1,) * (bins_subset.ndim - 1)
                        perturbed_bins_parts.append(np.tile(bins_subset, tile_shape))

                    class_subset = np.array(classes_array[indices], copy=True)
                    tile_shape = (num_values,) + (1,) * (class_subset.ndim - 1)
                    perturbed_class_parts.append(np.tile(class_subset, tile_shape))

                    if threshold is not None and isinstance(threshold, (list, np.ndarray)):
                        threshold_subset = [threshold[i] for i in indices]
                        threshold_items.extend(threshold_subset * num_values)

                for j, val in enumerate(np.unique(upper_boundary)):
                    greater_values[f][j] = (np.unique(self.__get_greater_values(f, val)), val)
                    indices = np.where(upper_boundary == val)[0]
                    values = greater_values[f][j][0]
                    if values.size == 0 or indices.size == 0:
                        continue

                    base_slice = np.array(x[indices, :], copy=True)
                    num_instances_subset = base_slice.shape[0]
                    num_values = values.size

                    tiled_x = np.tile(base_slice, (num_values, 1))
                    tiled_x[:, f] = np.repeat(values, num_instances_subset)
                    perturbed_x_parts.append(tiled_x)

                    feature_info = np.empty((num_instances_subset * num_values, 4), dtype=object)
                    feature_info[:, 0] = f
                    feature_info[:, 1] = np.tile(indices, num_values)
                    feature_info[:, 2] = j
                    feature_info[:, 3] = False
                    perturbed_feature_parts.append(feature_info)

                    if bins_array is not None:
                        bins_subset = np.array(bins_array[indices], copy=True)
                        tile_shape = (num_values,) + (1,) * (bins_subset.ndim - 1)
                        perturbed_bins_parts.append(np.tile(bins_subset, tile_shape))

                    class_subset = np.array(classes_array[indices], copy=True)
                    tile_shape = (num_values,) + (1,) * (class_subset.ndim - 1)
                    perturbed_class_parts.append(np.tile(class_subset, tile_shape))

                    if threshold is not None and isinstance(threshold, (list, np.ndarray)):
                        threshold_subset = [threshold[i] for i in indices]
                        threshold_items.extend(threshold_subset * num_values)
                for i in range(len(x)):
                    covered_values[f][i] = (
                        self.__get_covered_values(f, lower_boundary[i], upper_boundary[i]),
                        (lower_boundary[i], upper_boundary[i]),
                    )
                    if covered_values[f][i][0].size == 0:
                        continue
                    for value in covered_values[f][i][0]:
                        x_local = x_copy[i, :].copy()
                        x_local[f] = value
                        perturbed_x_parts.append(x_local[np.newaxis, :])
                        perturbed_feature_parts.append(
                            np.array([(f, i, i, None)], dtype=object)
                        )
                        if bins is not None:
                            perturbed_bins_parts.append(np.array([bins[i]]))
                        perturbed_class_parts.append(np.array([prediction["classes"][i]], copy=True))
                        if threshold is not None and isinstance(threshold, (list, np.ndarray)):
                            threshold_items.append(threshold[i])

        if perturbed_x_parts:
            base_parts = [perturbed_x] if perturbed_x.size else []
            perturbed_x = np.concatenate(base_parts + perturbed_x_parts, axis=0)
        if perturbed_feature_parts:
            base_parts = [perturbed_feature.astype(object)] if perturbed_feature.size else []
            perturbed_feature = np.concatenate(base_parts + perturbed_feature_parts, axis=0)
        if perturbed_bins is not None and perturbed_bins_parts:
            base_parts = [perturbed_bins] if perturbed_bins.size else []
            perturbed_bins = np.concatenate(base_parts + perturbed_bins_parts, axis=0)
        if perturbed_class_parts:
            base_parts = [perturbed_class] if perturbed_class.size else []
            perturbed_class = np.concatenate(base_parts + perturbed_class_parts, axis=0)

        if threshold is not None and isinstance(threshold, (list, np.ndarray)):
            if isinstance(threshold[0], tuple):
                base_items: List[Any] = (
                    list(perturbed_threshold) if len(perturbed_threshold) else []
                )
                perturbed_threshold = base_items + threshold_items
            else:
                base_array = perturbed_threshold if len(perturbed_threshold) else np.empty((0,))
                if threshold_items:
                    perturbed_threshold = np.concatenate(
                        [base_array, np.asarray(threshold_items)]
                    )
                else:
                    perturbed_threshold = base_array
        # Sub-step 1.c: Predict and convert to numpy arrays to allow boolean indexing
        if (
            threshold is not None
            and isinstance(threshold, (list, np.ndarray))
            and isinstance(threshold[0], tuple)
        ):
            perturbed_threshold = [tuple(pair) for pair in perturbed_threshold]
        predict, low, high, _ = self._predict(
            perturbed_x,
            threshold=perturbed_threshold,
            low_high_percentiles=low_high_percentiles,
            classes=perturbed_class,
            bins=perturbed_bins,
        )
        predict = np.array(predict)
        low = np.array(low)
        high = np.array(high)
        # predicted_class = np.array(perturbed_class)
        return (
            predict,
            low,
            high,
            prediction,
            perturbed_feature,
            rule_boundaries,
            lesser_values,
            greater_values,
            covered_values,
            x_cal,
        )

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

        if not self.is_fast():
            try:
                self.__fast = True
                self.__initialize_interval_learner_for_fast_explainer()
            except Exception as exc:
                self.__fast = False
                raise ConfigurationError(
                    "Fast explanations are only possible if the explainer is a Fast Calibrated Explainer."
                ) from exc
        total_time = time()
        instance_time = []
        if safe_isinstance(x, "pandas.core.frame.DataFrame"):
            x = x.values  # pylint: disable=invalid-name
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.num_features:
            raise DataShapeError(
                "The number of features in the test data must be the same as in the \
                            calibration data."
            )
        if self._is_mondrian():
            if bins is None:
                raise ValidationError(
                    "The bins parameter must be specified for Mondrian explanations."
                )
            if len(bins) != len(x):
                raise DataShapeError(
                    "The length of the bins parameter must be the same as the number of instances in x."
                )
        explanation = CalibratedExplanations(self, x, threshold, bins)

        if threshold is not None:
            if "regression" not in self.mode:
                raise ValidationError(
                    "The threshold parameter is only supported for mode='regression'."
                )
            assert_threshold(threshold, x)
        # explanation.low_high_percentiles = low_high_percentiles
        elif "regression" in self.mode:
            explanation.low_high_percentiles = low_high_percentiles

        feature_weights: Dict[str, List[np.ndarray]] = {
            "predict": [],
            "low": [],
            "high": [],
        }
        feature_predict: Dict[str, List[np.ndarray]] = {
            "predict": [],
            "low": [],
            "high": [],
        }
        instance_weights = [
            {
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(x))
        ]
        instance_predict = [
            {
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(x))
        ]

        feature_time = time()

        predict, low, high, predicted_class = self._predict(
            x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
        )
        prediction: Dict[str, Any] = {
            "predict": predict,
            "low": low,
            "high": high,
            "classes": (predicted_class if self.is_multiclass() else np.ones(x.shape[0])),
        }
        y_cal = self.y_cal
        self.y_cal = self.scaled_y_cal
        features_to_process = [
            f for f in range(self.num_features) if f not in self.features_to_ignore
        ]

        def _process_feature(f_idx: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            local_predict, local_low, local_high, _ = self._predict(
                x,
                threshold=threshold,
                low_high_percentiles=low_high_percentiles,
                bins=bins,
                feature=f_idx,
            )
            weights_predict = np.zeros(len(x))
            weights_low = np.zeros(len(x))
            weights_high = np.zeros(len(x))
            for idx in range(len(x)):
                weights_predict[idx] = self._assign_weight(
                    local_predict[idx], prediction["predict"][idx]
                )
                tmp_low = self._assign_weight(local_low[idx], prediction["predict"][idx])
                tmp_high = self._assign_weight(local_high[idx], prediction["predict"][idx])
                weights_low[idx] = np.min([tmp_low, tmp_high])
                weights_high[idx] = np.max([tmp_low, tmp_high])
            return (
                f_idx,
                np.asarray(weights_predict),
                np.asarray(weights_low),
                np.asarray(weights_high),
                np.asarray(local_predict),
                np.asarray(local_low),
                np.asarray(local_high),
            )

        executor = getattr(self, "_perf_parallel", None)
        if executor is not None and executor.config.enabled:
            feature_results = executor.map(_process_feature, features_to_process)
        else:
            feature_results = [_process_feature(f_idx) for f_idx in features_to_process]

        for (
            feature_index,
            weights_predict,
            weights_low,
            weights_high,
            local_predict,
            local_low,
            local_high,
        ) in feature_results:
            for i in range(len(x)):
                instance_weights[i]["predict"][feature_index] = weights_predict[i]
                instance_weights[i]["low"][feature_index] = weights_low[i]
                instance_weights[i]["high"][feature_index] = weights_high[i]
                instance_predict[i]["predict"][feature_index] = local_predict[i]
                instance_predict[i]["low"][feature_index] = local_low[i]
                instance_predict[i]["high"][feature_index] = local_high[i]
        self.y_cal = y_cal

        for i in range(len(x)):
            feature_weights["predict"].append(instance_weights[i]["predict"])
            feature_weights["low"].append(instance_weights[i]["low"])
            feature_weights["high"].append(instance_weights[i]["high"])

            feature_predict["predict"].append(instance_predict[i]["predict"])
            feature_predict["low"].append(instance_predict[i]["low"])
            feature_predict["high"].append(instance_predict[i]["high"])
        feature_time = time() - feature_time
        instance_time = [feature_time / x.shape[0]] * x.shape[0]

        explanation.finalize_fast(
            feature_weights,
            feature_predict,
            prediction,
            instance_time=instance_time,
            total_time=total_time,
        )
        self.latest_explanation = explanation
        self._last_explanation_mode = "fast"
        return explanation

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
        if not self.__lime_enabled:
            self._preload_lime()
        total_time = time()
        instance_time = []
        if safe_isinstance(x, "pandas.core.frame.DataFrame"):
            x = x.values  # pylint: disable=invalid-name
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.num_features:
            raise DataShapeError(
                "The number of features in the test data must be the same as in the \
                            calibration data."
            )
        if self._is_mondrian():
            if bins is None:
                raise ValidationError(
                    "The bins parameter must be specified for Mondrian explanations."
                )
            if len(bins) != len(x):
                raise DataShapeError(
                    "The length of the bins parameter must be the same as the number of instances in x."
                )
        explanation = CalibratedExplanations(self, x, threshold, bins)

        if threshold is not None:
            if "regression" not in self.mode:
                raise ValidationError(
                    "The threshold parameter is only supported for mode='regression'."
                )
            assert_threshold(threshold, x)
        # explanation.low_high_percentiles = low_high_percentiles
        elif "regression" in self.mode:
            explanation.low_high_percentiles = low_high_percentiles

        feature_weights: Dict[str, List[np.ndarray]] = {
            "predict": [],
            "low": [],
            "high": [],
        }
        feature_predict: Dict[str, List[np.ndarray]] = {
            "predict": [],
            "low": [],
            "high": [],
        }
        prediction: Dict[str, Any] = {"predict": [], "low": [], "high": [], "classes": []}

        instance_weights = [
            {
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(x))
        ]
        instance_predict = [
            {
                "predict": np.zeros(self.num_features),
                "low": np.zeros(self.num_features),
                "high": np.zeros(self.num_features),
            }
            for _ in range(len(x))
        ]

        predict, low, high, predicted_class = self._predict(
            x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
        )
        prediction["predict"] = predict
        prediction["low"] = low
        prediction["high"] = high
        if self.is_multiclass():
            prediction["classes"] = predicted_class
        else:
            prediction["classes"] = np.ones(x.shape[0])

            explainer = self.lime

        def low_proba(x):
            _, low, _, _ = self._predict(
                x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
            )
            return np.asarray([[1 - l, l] for l in low])  # noqa E741

        def high_proba(x):
            _, _, high, _ = self._predict(
                x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
            )
            return np.asarray([[1 - h, h] for h in high])  # noqa E741

        res_struct: Dict[str, Dict[str, Any]] = {}
        res_struct["low"] = {}
        res_struct["high"] = {}
        res_struct["low"]["explanation"], res_struct["high"]["explanation"] = [], []
        res_struct["low"]["abs_rank"], res_struct["high"]["abs_rank"] = [], []
        res_struct["low"]["values"], res_struct["high"]["values"] = [], []

        for i, instance in enumerate(x):
            instance_timer = time()

            assert explainer is not None
            low = explainer.explain_instance(
                instance, predict_fn=low_proba, num_features=len(instance)
            )
            high = explainer.explain_instance(
                instance, predict_fn=high_proba, num_features=len(instance)
            )

            res_struct["low"]["explanation"].append(low)
            res_struct["high"]["explanation"].append(high)
            res_struct["low"]["abs_rank"], res_struct["high"]["abs_rank"] = (
                np.zeros(len(instance)),
                np.zeros(len(instance)),
            )
            res_struct["low"]["values"], res_struct["high"]["values"] = (
                np.zeros(len(instance)),
                np.zeros(len(instance)),
            )

            for j, f in enumerate(low.local_exp[1]):
                res_struct["low"]["abs_rank"][f[0]] = low.local_exp[1][j][0]
                res_struct["low"]["values"][f[0]] = f[1]
            for j, f in enumerate(high.local_exp[1]):
                res_struct["high"]["abs_rank"][f[0]] = high.local_exp[1][j][0]
                res_struct["high"]["values"][f[0]] = f[1]

            for f in range(self.num_features):
                tmp_low = res_struct["low"]["values"][f]
                tmp_high = res_struct["high"]["values"][f]
                instance_weights[i]["low"][f] = np.min([tmp_low, tmp_high])
                instance_weights[i]["high"][f] = np.max([tmp_low, tmp_high])
                instance_weights[i]["predict"][f] = instance_weights[i]["high"][f] / (
                    1 - instance_weights[i]["low"][f] + instance_weights[i]["high"][f]
                )

                instance_predict[i]["low"][f] = (
                    low.predict_proba[-1] - instance_weights[i]["low"][f]
                )
                instance_predict[i]["high"][f] = (
                    high.predict_proba[-1] - instance_weights[i]["high"][f]
                )
                instance_predict[i]["predict"][f] = instance_predict[i]["high"][f] / (
                    1 - instance_predict[i]["low"][f] + instance_predict[i]["high"][f]
                )

            feature_weights["predict"].append(instance_weights[i]["predict"])
            feature_weights["low"].append(instance_weights[i]["low"])
            feature_weights["high"].append(instance_weights[i]["high"])

            feature_predict["predict"].append(instance_predict[i]["predict"])
            feature_predict["low"].append(instance_predict[i]["low"])
            feature_predict["high"].append(instance_predict[i]["high"])
            instance_time.append(time() - instance_timer)

        explanation.finalize_fast(
            feature_weights,
            feature_predict,
            prediction,
            instance_time=instance_time,
            total_time=total_time,
        )
        self.latest_explanation = explanation
        return explanation

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
        return (
            prediction - instance_predict
            if np.isscalar(prediction)
            else [prediction[i] - ip for i, ip in enumerate(instance_predict)]
        )  # probabilistic regression

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
        # backwards compatibility
        if len(instances.shape) == 1:
            min_max = []
            if perturbed_instances is None:
                perturbed_instances = self._discretize(instances.reshape(1, -1))
            for f in range(self.num_features):
                if f not in self.discretizer.to_discretize:
                    min_max.append([instances[f], instances[f]])
                else:
                    bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
                    min_max.append(
                        [
                            self.discretizer.mins[f][
                                np.digitize(perturbed_instances[0, f], bins, right=True) - 1
                            ],
                            self.discretizer.maxs[f][
                                np.digitize(perturbed_instances[0, f], bins, right=True) - 1
                            ],
                        ]
                    )
            return min_max
        instances = np.array(instances)  # Ensure instances is a numpy array
        if perturbed_instances is None:
            perturbed_instances = self._discretize(instances)
        else:
            perturbed_instances = np.array(
                perturbed_instances
            )  # Ensure perturbed_instances is a numpy array

        all_min_max = []
        for instance, perturbed_instance in zip(instances, perturbed_instances):
            min_max = []
            for f in range(self.num_features):
                if f not in self.discretizer.to_discretize:
                    min_max.append([instance[f], instance[f]])
                else:
                    bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
                    min_max.append(
                        [
                            self.discretizer.mins[f][
                                np.digitize(perturbed_instance[f], bins, right=True) - 1
                            ],
                            self.discretizer.maxs[f][
                                np.digitize(perturbed_instance[f], bins, right=True) - 1
                            ],
                        ]
                    )
            all_min_max.append(min_max)
        return np.array(all_min_max)

    def __get_greater_values(self, f: int, greater: float):
        """Get sampled values above ``greater`` for numerical features.

        Uses percentile sampling from calibration data.
        """
        if not np.any(self.x_cal[:, f] > greater):
            return np.array([])
        return np.percentile(self.x_cal[self.x_cal[:, f] > greater, f], self.sample_percentiles)

    def __get_lesser_values(self, f: int, lesser: float):
        """Get sampled values below ``lesser`` for numerical features.

        Uses percentile sampling from calibration data.
        """
        if not np.any(self.x_cal[:, f] < lesser):
            return np.array([])
        return np.percentile(self.x_cal[self.x_cal[:, f] < lesser, f], self.sample_percentiles)

    def __get_covered_values(self, f: int, lesser: float, greater: float):
        """Get sampled values within the ``[lesser, greater]`` interval.

        Uses percentile sampling from calibration data.
        """
        covered = np.where((self.x_cal[:, f] >= lesser) & (self.x_cal[:, f] <= greater))[0]
        return np.percentile(self.x_cal[covered, f], self.sample_percentiles)

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
            self.__initialize_interval_learner()

    def __constant_sigma(self, x: np.ndarray, learner=None, beta=None) -> np.ndarray:  # pylint: disable=unused-argument
        return np.ones(x.shape[0]) if isinstance(x, (np.ndarray, list, tuple)) else np.ones(1)

    def _get_sigma_test(self, x: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances."""
        if self.difficulty_estimator is None:
            return self.__constant_sigma(x)
        return self.difficulty_estimator.apply(x)

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
            self.__initialize_interval_learner()

    def __update_interval_learner(self, xs, ys, bins=None) -> None:  # pylint: disable=unused-argument
        if self.is_fast():
            raise ConfigurationError("Fast explanations are not supported in this update path.")
        if self.mode == "classification":
            # pylint: disable=fixme
            # TODO: change so that existing calibrators are extended with new calibration instances
            self.interval_learner = VennAbers(
                self.x_cal,
                self.y_cal,
                self.learner,
                self.bins,
                difficulty_estimator=self.difficulty_estimator,
                predict_function=self.predict_function,
            )
        elif "regression" in self.mode:
            if isinstance(self.interval_learner, list):
                raise ConfigurationError("Fast explanations are not supported in this update path.")
            # update the IntervalRegressor
            self.interval_learner.insert_calibration(xs, ys, bins=bins)
        self.__initialized = True

    def __initialize_interval_learner(self) -> None:
        # Thin delegator kept for backward-compatibility internal calls
        from .calibration_helpers import initialize_interval_learner as _init_il

        _init_il(self)

    # pylint: disable=attribute-defined-outside-init
    def __initialize_interval_learner_for_fast_explainer(self):
        from .calibration_helpers import (
            initialize_interval_learner_for_fast_explainer as _init_fast,
        )

        _init_fast(self)

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
        if calibration_set is None:
            x_cal, y_cal = self.x_cal, self.y_cal
        elif calibration_set is tuple:
            x_cal, y_cal = calibration_set
        else:
            x_cal, y_cal = calibration_set[0], calibration_set[1]
        self.reject_threshold = None
        if self.mode in "regression":
            proba_1, _, _, _ = self.interval_learner.predict_probability(
                x_cal, y_threshold=threshold, bins=self.bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = (y_cal < threshold).astype(int)
            self.reject_threshold = threshold
        elif self.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.interval_learner.predict_proba(x_cal, bins=self.bins)
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = (classes == y_cal).astype(int)
        else:
            proba = self.interval_learner.predict_proba(x_cal, bins=self.bins)
            classes = y_cal
        alphas_cal = hinge(proba, np.unique(classes), classes)
        self.reject_learner = ConformalClassifier().fit(alphas=alphas_cal, bins=classes)
        return self.reject_learner

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
        if self.mode in "regression":
            if self.reject_threshold is None:
                raise ValidationError(
                    "The reject learner is only available for regression with a threshold."
                )
            proba_1, _, _, _ = self.interval_learner.predict_probability(
                x, y_threshold=self.reject_threshold, bins=bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = [0, 1]
        elif self.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.interval_learner.predict_proba(x, bins=bins)
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = [0, 1]
        else:
            proba = self.interval_learner.predict_proba(x, bins=bins)
            classes = np.unique(self.y_cal)
        alphas_test = hinge(proba)

        prediction_set = np.array(
            [
                self.reject_learner.predict_set(
                    alphas_test, np.full(len(alphas_test), classes[c]), confidence=confidence
                )[:, c]
                for c in range(len(classes))
            ]
        ).T
        singleton = np.sum(np.sum(prediction_set, axis=1) == 1)
        empty = np.sum(np.sum(prediction_set, axis=1) == 0)
        n = len(x)

        epsilon = 1 - confidence
        error_rate = (n * epsilon - empty) / singleton
        reject_rate = 1 - singleton / n

        rejected = np.sum(prediction_set, axis=1) != 1
        return rejected, error_rate, reject_rate

    def _preprocess(self):
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
        x = np.array(x)  # Ensure x is a numpy array
        for f in self.discretizer.to_discretize:
            bins = np.concatenate(([-np.inf], self.discretizer.mins[f][1:], [np.inf]))
            x[:, f] = [
                self.discretizer.means[f][np.digitize(x[i, f], bins, right=True) - 1]
                for i in range(len(x))
            ]
        return x

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

    # pylint: disable=too-many-return-statements
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
                Specifies the threshold(s) to get a thresholded prediction for regression tasks (prediction labels such as ``y_hat <= threshold`` or ``y_hat > threshold``). This parameter is ignored for classification tasks.

            - low_high_percentiles : tuple of two floats, optional, default=(5, 95)
                The lower and upper percentiles used to calculate the prediction interval for regression tasks. Determines the breadth of the interval based on the distribution of the predictions. This parameter is ignored for classification tasks.

        Raises
        ------
        RuntimeError
            If the learner has not been fitted prior to making predictions.

        Warning
            If the learner is not calibrated.

        Returns
        -------
        calibrated_prediction : float or array-like, or str
            The calibrated prediction. For regression tasks, this is the median of the conformal predictive system or a thresholded prediction if ``threshold`` is set. For classification tasks, it is the class label with the highest calibrated probability.
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
        # Phase 1B: emit deprecation warnings for aliases and normalize kwargs
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
        # Phase 1B: emit deprecation warnings for aliases and normalize kwargs
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
        """Return whether lime export is enabled.

        If is_enabled is not None, then the lime export is enabled/disabled according to the value of is_enabled.

        Parameters
        ----------
            is_enabled (bool, optional): is used to assign whether lime export is enabled or not. Defaults to None.

        Returns
        -------
            bool: returns whether lime export is enabled
        """
        if is_enabled is not None:
            self.__lime_enabled = is_enabled
        return self.__lime_enabled

    def _is_shap_enabled(self, is_enabled=None) -> bool:
        """Return whether shap export is enabled.

        If is_enabled is not None, then the shap export is enabled/disabled according to the value of is_enabled.

        Parameters
        ----------
            is_enabled (bool, optional): is used to assign whether shap export is enabled or not. Defaults to None.

        Returns
        -------
            bool: returns whether shap export is enabled
        """
        if is_enabled is not None:
            self.__shap_enabled = is_enabled
        return self.__shap_enabled

    def _preload_lime(self, x_cal=None):
        if not (lime := safe_import("lime.lime_tabular", "LimeTabularExplainer")):
            return None, None
        if not self._is_lime_enabled():
            if self.mode == "classification":
                self.lime = lime(
                    self.x_cal[:1, :] if x_cal is None else x_cal,
                    feature_names=self.feature_names,
                    class_names=["0", "1"],
                    mode=self.mode,
                )
                self.lime_exp = self.lime.explain_instance(
                    self.x_cal[0, :], self.learner.predict_proba, num_features=self.num_features
                )
            elif "regression" in self.mode:
                self.lime = lime(
                    self.x_cal[:1, :] if x_cal is None else x_cal,
                    feature_names=self.feature_names,
                    mode="regression",
                )
                self.lime_exp = self.lime.explain_instance(
                    self.x_cal[0, :], self.learner.predict, num_features=self.num_features
                )
            self._is_lime_enabled(True)
        return self.lime, self.lime_exp

    def _preload_shap(self, num_test=None):
        if shap := safe_import("shap"):
            if (
                not self._is_shap_enabled()
                or num_test is not None
                and self.shap_exp.shape[0] != num_test
            ):

                def f(x):
                    return self._predict(x)[0]

                self.shap = shap.Explainer(f, self.x_cal, feature_names=self.feature_names)
                self.shap_exp = (
                    self.shap(self.x_cal[0, :].reshape(1, -1))
                    if num_test is None
                    else self.shap(self.x_cal[:num_test, :])
                )
                self._is_shap_enabled(True)
            return self.shap, self.shap_exp
        return None, None

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
        The confusion matrix is only available for classification tasks. Leave-one-out cross-validation is
        used on the calibration set to generate the confusion matrix.

        Returns
        -------
        array-like
            The calibrated confusion matrix.
        """
        if not (self.mode == "classification"):
            raise ValidationError(
                "The confusion matrix is only available for classification tasks."
            )
        cal_predicted_classes = np.zeros(len(self.y_cal))
        for i in range(len(self.y_cal)):
            va = VennAbers(
                np.concatenate((self.x_cal[:i], self.x_cal[i + 1 :]), axis=0),
                np.concatenate((self.y_cal[:i], self.y_cal[i + 1 :])),
                self.learner,
                bins=np.concatenate((self.bins[:i], self.bins[i + 1 :]))
                if self.bins is not None
                else None,
            )
            _, _, _, predict = va.predict_proba(
                [self.x_cal[i]],
                output_interval=True,
                bins=[self.bins[i]] if self.bins is not None else None,
            )
            cal_predicted_classes[i] = predict[0]
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
