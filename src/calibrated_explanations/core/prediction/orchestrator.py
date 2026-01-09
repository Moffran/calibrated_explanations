"""Orchestration layer for prediction and interval calibration.

This module provides the PredictionOrchestrator class which coordinates
prediction pipeline execution, including interval calibration, difficulty
estimation, uncertainty quantification, and caching.

Part of Phase 1b: Delegate Prediction Orchestration (ADR-001, ADR-004).

Note: All plugin defaults, chaining, and fallback logic has been moved to
PluginManager. This orchestrator delegates all chain-building to PluginManager.
"""

# pylint: disable=protected-access, too-many-lines, invalid-name, import-outside-toplevel
# pylint: disable=line-too-long, unnecessary-pass, broad-except

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

from ...plugins import (
    ClassificationIntervalCalibrator,
    IntervalCalibratorContext,
    RegressionIntervalCalibrator,
    ensure_builtin_plugins,
    find_interval_descriptor,
    find_interval_plugin,
    find_interval_plugin_trusted,
    is_identifier_denied,
)
from ...calibration.interval_wrappers import FastIntervalCalibrator, is_fast_interval_collection
from ...logging import logging_context, update_logging_context
from ...utils import assert_threshold
from ...utils.exceptions import (
    CalibratedError,
    ConfigurationError,
    DataShapeError,
    NotFittedError,
    ValidationError,
)
from .validation import check_interval_runtime_metadata

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


def _freeze_context_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_context_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_context_value(item) for item in value)
    if hasattr(value, "copy") and callable(value.copy):
        try:
            return value.copy()
        except CalibratedError:  # pragma: no cover - defensive
            return value
    return value


def _freeze_context_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_context_value(value) for key, value in mapping.items()})


class PredictionOrchestrator:
    """Orchestrate prediction pipeline execution and interval calibration.

    This class handles the complete prediction workflow including:
    - Interval calibration and plugin resolution
    - Difficulty estimation coordination
    - Uncertainty quantification
    - Prediction caching and performance tracking

    Attributes
    ----------
    explainer : CalibratedExplainer
        Back-reference to the parent explainer instance.
    """

    def __init__(self, explainer: CalibratedExplainer) -> None:
        """Initialize the orchestrator with a back-reference to the explainer.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The parent explainer instance.

        Notes
        -----
        The orchestrator is a thin coordination layer that manages behavior
        using state stored on the parent explainer. It does not hold state itself.
        State is accessed through these explainer fields:
        - explainer._interval_plugin_identifiers
        - explainer._interval_plugin_fallbacks
        - explainer._interval_plugin_hints
        - explainer._interval_context_metadata
        - explainer._telemetry_interval_sources
        - explainer._interval_preferred_identifier
        """
        self.explainer = explainer
        # Initialize interval registry for managing interval learner lifecycle
        from .interval_registry import IntervalRegistry

        self.interval_registry = IntervalRegistry(explainer)
        self._logger = logging.getLogger(__name__)

    def initialize_chains(self) -> None:
        """Delegate to PluginManager for chain initialization.

        PluginManager is now the single source of truth for all plugin
        defaults, chains, and fallbacks. This method delegates to it.

        Notes
        -----
        This method is called during explainer initialization to pre-compute the
        interval calibrator plugin resolution chains for default and fast modes.
        """
        self.explainer.plugin_manager.initialize_chains()

    def predict(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        classes=None,
        bins=None,
        feature=None,
        **kwargs,
    ):
        """Execute a prediction with optional uncertainty quantification.

        Parameters
        ----------
        x : array-like
            Test instances to predict.
        threshold : float, int, array-like, or None
            Threshold for probabilistic regression predictions.
        low_high_percentiles : tuple of floats
            Low and high percentiles for interval calculation.
        classes : array-like or None
            Classes for multiclass predictions.
        bins : array-like or None
            Mondrian categories.
        feature : int or None
            Feature index for fast prediction mode.
        **kwargs : dict
            Additional arguments (show, style_override are stripped).

        Returns
        -------
        tuple
            (prediction, low, high, classes) for classification/regression.
        """
        return self._predict(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            feature=feature,
            **kwargs,
        )

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
        """Cache-aware wrapper around _predict_impl.

        Checks performance cache before delegating to actual prediction logic.
        Stores results in cache after successful prediction.
        """
        cache = getattr(self.explainer, "perf_cache", None)
        cache_enabled = getattr(cache, "enabled", False)
        key_parts = None
        if cache_enabled:
            x_arr = np.asarray(x)
            key_parts = (
                ("mode", self.explainer.mode),
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

        self._validate_prediction_result(result)
        return result

    def _validate_prediction_result(self, result):
        """Enforce low <= predict <= high invariant on prediction results."""
        predict, low, high, _ = result

        # Skip validation if any component is None or empty
        if predict is None or low is None or high is None:
            return

        with contextlib.suppress(TypeError, ValueError):
            # Skip validation if conversion to array fails or types are incompatible
            predict = np.asanyarray(predict)
            low = np.asanyarray(low)
            high = np.asanyarray(high)

            if predict.size == 0 or low.size == 0 or high.size == 0:
                return

            # Check for numeric types
            if not (
                np.issubdtype(predict.dtype, np.number)
                and np.issubdtype(low.dtype, np.number)
                and np.issubdtype(high.dtype, np.number)
            ):
                return

            # Check low <= high
            if not np.all(low <= high):
                warnings.warn(
                    "Prediction interval invariant violated: low > high. This indicates an issue with the underlying estimator.",
                    UserWarning,
                    stacklevel=2,
                )

            # Check low <= predict <= high
            # Allow small floating point tolerance
            epsilon = 1e-9
            if not np.all((low - epsilon <= predict) & (predict <= high + epsilon)):
                warnings.warn(
                    "Prediction invariant violated: predict not in [low, high]. This may indicate poor calibration or inconsistent point predictions.",
                    UserWarning,
                    stacklevel=2,
                )

    # Public alias for testing
    validate_prediction_result = _validate_prediction_result

    def predict_internal(self, *args, **kwargs) -> Any:
        """Public alias for internal predict implementation."""
        return self._predict_impl(*args, **kwargs)

    def _predict_impl(
        self,
        x,
        threshold=None,
        low_high_percentiles=(5, 95),
        classes=None,
        bins=None,
        feature=None,
        **kwargs,
    ):
        """Backwards-compatible alias for predict_impl."""
        return self.predict_impl(
            x,
            threshold=threshold,
            low_high_percentiles=low_high_percentiles,
            classes=classes,
            bins=bins,
            feature=feature,
            **kwargs,
        )

    def predict_impl(
        self,
        x,
        threshold=None,
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
        - Can return probability predictions when threshold is provided (probabilistic regression)

        Parameters
        ----------
        x : A set of test objects to predict
        threshold : float, int or array-like of shape (n_samples,), default=None
            values for which p-values should be returned. Only used for probabilistic regression
            (also called thresholded regression). Returns P(y <= threshold) probabilities.
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
        if bins is not None:
            bins = np.asarray(bins)
        if not self.explainer.initialized:
            raise NotFittedError("The learner must be initialized before calling predict.")
        if feature is None and self.explainer.is_fast():
            feature = self.explainer.num_features  # Use the calibrator defined using X_cal
        if self.explainer.mode == "classification":
            if self.explainer.is_multiclass():
                if self.explainer.is_fast():
                    predict, low, high, new_classes = self.explainer.interval_learner[
                        feature
                    ].predict_proba(x, output_interval=True, classes=classes, bins=bins)
                else:
                    predict, low, high, new_classes = self.explainer.interval_learner.predict_proba(
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

            if self.explainer.is_fast():
                predict, low, high = self.explainer.interval_learner[feature].predict_proba(
                    x, output_interval=True, bins=bins
                )
            else:
                predict, low, high = self.explainer.interval_learner.predict_proba(
                    x, output_interval=True, bins=bins
                )
            return predict[:, 1], low, high, None
        if "regression" in self.explainer.mode:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            if threshold is None:  # normal regression
                if not (low_high_percentiles[0] <= low_high_percentiles[1]):  # pylint: disable=superfluous-parens
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
                    if self.explainer.is_fast():
                        return self.explainer.interval_learner[feature].predict_uncertainty(
                            x, low_high_percentiles, bins=bins
                        )
                    return self.explainer.interval_learner.predict_uncertainty(
                        x, low_high_percentiles, bins=bins
                    )
                except:  # pylint: disable=bare-except
                    # typically crepes broadcasting/shape errors
                    # Use bare except + sys.exc_info to avoid catching 'Exception' explicitly (ADR-002)
                    exc = sys.exc_info()[1]
                    if not isinstance(exc, Exception):
                        raise

                    if self.explainer.suppress_crepes_errors:
                        # Log and return placeholder arrays (caller should handle downstream)
                        # Emit a UserWarning only when fallback chains are enabled
                        # (tests opt-in via the `enable_fallbacks` fixture which
                        # deletes the env var set by the disable fixture). When
                        # fallbacks remain disabled we log info instead to avoid
                        # triggering the runtime fallback enforcement.
                        if os.getenv("CE_INTERVAL_PLUGIN_FALLBACKS") is None:
                            warnings.warn(
                                "crepes produced an unexpected result (likely too-small calibration set); returning zeros as a degraded fallback.",
                                UserWarning,
                                stacklevel=2,
                            )
                        else:
                            self._logger.info(
                                "crepes produced an unexpected result (likely too-small calibration set); returning zeros as a degraded result: %s",
                                exc,
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
                if self.explainer.is_fast():
                    return self.explainer.interval_learner[feature].predict_probability(
                        x, threshold, bins=bins
                    )
                # pylint: disable=unexpected-keyword-arg
                return self.explainer.interval_learner.predict_probability(x, threshold, bins=bins)
            except:  # noqa: E722 - ADR-002: Use bare except + sys.exc_info to avoid catching 'Exception' explicitly
                exc = sys.exc_info()[1]
                if not isinstance(exc, Exception):
                    raise

                if self.explainer.suppress_crepes_errors:
                    if os.getenv("CE_INTERVAL_PLUGIN_FALLBACKS") is None:
                        warnings.warn(
                            "crepes produced an unexpected result while computing probabilities; returning zeros as a degraded fallback.",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        self._logger.info(
                            "crepes produced an unexpected result while computing probabilities; returning zeros as a degraded result: %s",
                            exc,
                        )
                    n = x.shape[0]
                    return np.zeros(n), np.zeros(n), np.zeros(n), None
                # Re-raise as a clearer DataShapeError with guidance
                raise DataShapeError(
                    "Error while computing prediction intervals from the underlying crepes library. "
                    "This commonly occurs when the calibration set is too small for the requested "
                    "percentiles. Consider using a larger calibration set, or instantiate the "
                    "explainer with suppress_crepes_errors=True to return a degraded fallback. "
                    f"(original error: {exc})"
                ) from exc

        return None, None, None, None  # Should never happen

    def ensure_interval_runtime_state(self) -> None:
        """Ensure interval tracking members exist for legacy instances."""
        if not self.explainer.plugin_manager.interval_plugin_hints:
            self.explainer.plugin_manager.interval_plugin_hints = {}
        if not self.explainer.plugin_manager.interval_plugin_fallbacks:
            self.explainer.plugin_manager.interval_plugin_fallbacks = {}
        if not self.explainer.plugin_manager.interval_plugin_identifiers:
            self.explainer.plugin_manager.interval_plugin_identifiers = {
                "default": None,
                "fast": None,
            }
        if not self.explainer.plugin_manager.telemetry_interval_sources:
            self.explainer.plugin_manager.telemetry_interval_sources = {
                "default": None,
                "fast": None,
            }
        if not self.explainer.plugin_manager.interval_preferred_identifier:
            self.explainer.plugin_manager.interval_preferred_identifier = {
                "default": None,
                "fast": None,
            }
        if not self.explainer.plugin_manager.interval_context_metadata:
            self.explainer.plugin_manager.interval_context_metadata = {"default": {}, "fast": {}}

    def gather_interval_hints(self, *, fast: bool) -> Tuple[str, ...]:
        """Return interval dependency hints collected from explanation plugins."""
        if fast:
            return self.explainer.plugin_manager.interval_plugin_hints.get("fast", ())
        ordered: List[str] = []
        seen: set[str] = set()
        for mode in ("factual", "alternative"):
            for identifier in self.explainer.plugin_manager.interval_plugin_hints.get(mode, ()):  # noqa: B020
                if identifier not in seen:
                    ordered.append(identifier)
                    seen.add(identifier)
        return tuple(ordered)

    def check_interval_runtime_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        identifier: str | None,
        fast: bool,
    ) -> str | None:
        """Validate interval plugin metadata for the current execution."""
        return check_interval_runtime_metadata(
            metadata,
            identifier=identifier,
            fast=fast,
            mode=self.explainer.mode,
            bins=self.explainer.bins,
        )

    def resolve_interval_plugin(
        self,
        *,
        fast: bool,
        hints: Sequence[str] = (),
    ) -> Tuple[Any, str | None]:
        """Resolve the interval plugin for the requested execution path."""
        ensure_builtin_plugins()

        raw_override = (
            self.explainer.plugin_manager.fast_interval_plugin_override
            if fast
            else self.explainer.plugin_manager.interval_plugin_override
        )
        override = self.explainer.plugin_manager.coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            identifier = getattr(override, "plugin_meta", {}).get("name")
            return override, identifier

        if isinstance(raw_override, str):
            preferred_identifier = raw_override
        else:
            key = "fast" if fast else "default"
            preferred_identifier = self.explainer.plugin_manager.interval_preferred_identifier.get(
                key
            )
        chain = list(
            self.explainer.plugin_manager.interval_plugin_fallbacks.get(
                "fast" if fast else "default", ()
            )
        )
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
            error = self.check_interval_runtime_metadata(
                meta_source,
                identifier=identifier,
                fast=fast,
            )
            if error:
                if preferred_identifier == identifier:
                    raise ConfigurationError(error)
                errors.append(error)
                continue

            plugin = self.explainer.instantiate_plugin(plugin)
            return plugin, identifier

        raise ConfigurationError(
            "Unable to resolve interval plugin for "
            + ("fast" if fast else "default")
            + " mode. Tried: "
            + ", ".join(chain or ("<none>",))
            + ("; errors: " + "; ".join(errors) if errors else "")
        )

    def build_interval_context(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> IntervalCalibratorContext:
        """Construct the interval calibrator context with mutable metadata for plugin use."""
        calibration_splits: Tuple[Any, ...] = ((self.explainer.x_cal, self.explainer.y_cal),)
        bins = {"calibration": self.explainer.bins}
        difficulty = {"estimator": self.explainer.difficulty_estimator}
        fast_flags = {"fast": fast}
        residuals: Mapping[str, Any] = {}
        key = "fast" if fast else "default"
        stored_metadata = dict(self.explainer.plugin_manager.interval_context_metadata.get(key, {}))
        enriched_metadata = stored_metadata
        enriched_metadata.update(metadata)
        enriched_metadata.setdefault("task", self.explainer.mode)
        enriched_metadata.setdefault("mode", self.explainer.mode)
        enriched_metadata.setdefault(
            "predict_function", getattr(self.explainer, "predict_function", None)
        )
        enriched_metadata.setdefault("difficulty_estimator", self.explainer.difficulty_estimator)
        enriched_metadata.setdefault("explainer", self.explainer)
        enriched_metadata.setdefault(
            "categorical_features", tuple(self.explainer.categorical_features)
        )
        enriched_metadata.setdefault("num_features", self.explainer.num_features)
        enriched_metadata.setdefault(
            "noise_config",
            {
                "noise_type": getattr(self.explainer, "noise_type", None),
                "scale_factor": getattr(self.explainer, "scale_factor", None),
                "severity": getattr(self.explainer, "severity", None),
                "seed": getattr(self.explainer, "seed", None),
                "rng": getattr(self.explainer, "rng", None),
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
            if not existing_fast and is_fast_interval_collection(self.explainer.interval_learner):
                existing_fast = tuple(self.explainer.interval_learner)
            if existing_fast:
                enriched_metadata["existing_fast_calibrators"] = tuple(existing_fast)
        # Freeze nested metadata values for safety but keep the top-level
        # metadata as a plain mutable dict so plugins can add entries during
        # execution. The context will be frozen when persisted by
        # `obtain_interval_calibrator()`.
        metadata_for_plugins = {key: _freeze_context_value(value) for key, value in enriched_metadata.items()}
        return IntervalCalibratorContext(
            learner=self.explainer.learner,
            calibration_splits=calibration_splits,
            bins=_freeze_context_mapping(bins),
            residuals=_freeze_context_mapping(residuals),
            difficulty=_freeze_context_mapping(difficulty),
            metadata=metadata_for_plugins,
            fast_flags=_freeze_context_mapping(fast_flags),
        )

    def obtain_interval_calibrator(
        self,
        *,
        fast: bool,
        metadata: Mapping[str, Any],
    ) -> Tuple[Any, str | None]:
        """Resolve and instantiate the interval calibrator for the active mode."""
        self.ensure_interval_runtime_state()
        hints = self.gather_interval_hints(fast=fast)
        with logging_context(
            explainer_id=getattr(self.explainer, "id", None),
        ):
            plugin, identifier = self.resolve_interval_plugin(fast=fast, hints=hints)
            with logging_context(plugin_identifier=identifier):
                context = self.build_interval_context(fast=fast, metadata=metadata)
                try:
                    calibrator = plugin.create(context, fast=fast)
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    exc = sys.exc_info()[1]
                    raise ConfigurationError(
                        f"Interval plugin execution failed for {'fast' if fast else 'default'} mode: {exc}"
                    ) from exc
                self.validate_interval_calibrator(
                    calibrator=calibrator,
                    context=context,
                    identifier=identifier,
                    fast=fast,
                    plugin=plugin,
            )
        self.capture_interval_calibrators(
            context=context,
            calibrator=calibrator,
            fast=fast,
        )
        key = "fast" if fast else "default"
        self.explainer.plugin_manager.interval_plugin_identifiers[key] = identifier
        self.explainer.plugin_manager.telemetry_interval_sources[key] = identifier
        metadata_dict = dict(context.metadata)
        if fast:
            if isinstance(calibrator, FastIntervalCalibrator):
                calibrators_tuple = calibrator.calibrators
            elif isinstance(calibrator, Sequence) and not isinstance(calibrator, (str, bytes)):
                calibrators_tuple = tuple(calibrator)
            else:
                calibrators_tuple = (calibrator,)
            metadata_dict["fast_calibrators"] = calibrators_tuple
            metadata_dict["existing_fast_calibrators"] = calibrators_tuple
        else:
            metadata_dict["calibrator"] = calibrator
        # persist captured metadata for future invocations without sharing references
        self.explainer.plugin_manager.interval_context_metadata[key] = dict(metadata_dict)
        return calibrator, identifier

    def capture_interval_calibrators(
        self,
        *,
        context: IntervalCalibratorContext,
        calibrator: Any,
        fast: bool,
    ) -> None:
        """Record the returned calibrator inside the interval context metadata."""
        metadata = context.metadata
        # Skip if metadata is immutable (MappingProxyType from frozen context)
        # Calibrators are cached separately in plugin_manager.interval_context_metadata
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

    def validate_interval_calibrator(
        self,
        *,
        calibrator: Any,
        context: IntervalCalibratorContext,
        identifier: str | None,
        fast: bool,
        plugin: Any = None,
    ) -> None:
        """Validate interval calibrator protocol conformance.

        Skips validation for untrusted plugins (they can return anything).
        Enforces protocol compliance for trusted plugins and builtins.
        """
        # Check if plugin is untrusted - if so, skip protocol validation
        if plugin is not None:
            plugin_meta = getattr(plugin, "plugin_meta", {})
            is_trusted = plugin_meta.get("trusted", True)
            if not is_trusted:
                # Untrusted plugins can return anything
                return

        task = str(context.metadata.get("task") or context.metadata.get("mode") or "")
        expected = (
            RegressionIntervalCalibrator
            if "regression" in task
            else ClassificationIntervalCalibrator
        )

        # For FAST mode: allow Sequence, FastIntervalCalibrator, or protocol-compliant single objects
        if fast:
            if calibrator is None:
                mode = "fast"
                label = (
                    identifier or getattr(calibrator, "plugin_meta", {}).get("name") or "<unknown>"
                )
                raise ConfigurationError(
                    f"Interval plugin '{label}' returned None for {mode} mode."
                )

            # Accept FastIntervalCalibrator or Sequence (validate items now)
            if isinstance(calibrator, (FastIntervalCalibrator, list, tuple)):
                # Deep validation: check each item in the sequence
                for idx, item in enumerate(calibrator):
                    if not isinstance(item, expected):
                        mode = "fast"
                        label = f"{identifier or '<unknown>'}[{idx}]"
                        expected_name = expected.__name__
                        actual = type(item).__name__
                        raise ConfigurationError(
                            f"Interval calibrator at index {idx} in '{label}' is non-compliant for {mode} mode "
                            f"(expected {expected_name}, got {actual})."
                        )
                return

            # Also accept protocol-compliant single objects
            if isinstance(calibrator, expected):
                return

            # Otherwise it's invalid
            mode = "fast"
            label = identifier or getattr(calibrator, "plugin_meta", {}).get("name") or "<unknown>"
            expected_name = f"FastIntervalCalibrator | Sequence[{expected.__name__}]"
            actual = type(calibrator).__name__
            raise ConfigurationError(
                f"Interval plugin '{label}' returned a non-compliant calibrator for {mode} mode "
                f"(expected {expected_name}, got {actual})."
            )
        else:
            # For non-FAST mode: single protocol-compliant calibrator
            if calibrator is None or not isinstance(calibrator, expected):
                label = (
                    identifier or getattr(calibrator, "plugin_meta", {}).get("name") or "<unknown>"
                )
                mode = "default"
                expected_name = expected.__name__
                actual = type(calibrator).__name__ if calibrator is not None else "None"
                raise ConfigurationError(
                    f"Interval plugin '{label}' returned a non-compliant calibrator for {mode} mode "
                    f"(expected {expected_name}, got {actual})."
                )
