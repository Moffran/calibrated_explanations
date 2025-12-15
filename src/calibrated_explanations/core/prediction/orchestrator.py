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
import sys
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ...core.config_helpers import coerce_string_tuple
from ...plugins import (
    IntervalCalibratorContext,
    ensure_builtin_plugins,
    find_interval_descriptor,
    find_interval_plugin,
    find_interval_plugin_trusted,
    is_identifier_denied,
)
from ...utils import assert_threshold
from ..exceptions import ConfigurationError, DataShapeError, NotFittedError, ValidationError
from ..explain.feature_task import assign_weight

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


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

        self._interval_registry = IntervalRegistry(explainer)

    def initialize_chains(self) -> None:
        """Delegate to PluginManager for chain initialization.

        PluginManager is now the single source of truth for all plugin
        defaults, chains, and fallbacks. This method delegates to it.

        Notes
        -----
        This method is called during explainer initialization to pre-compute the
        interval calibrator plugin resolution chains for default and fast modes.
        """
        self.explainer._plugin_manager.initialize_chains()

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
        cache = getattr(self.explainer, "_perf_cache", None)
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
        if not self.explainer._CalibratedExplainer__initialized:
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
                        warnings.warn(
                            "crepes produced an unexpected result (likely too-small calibration set); "
                            "returning zeros as a degraded fallback.",
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
                if self.explainer.is_fast():
                    return self.explainer.interval_learner[feature].predict_probability(
                        x, threshold, bins=bins
                    )
                # pylint: disable=unexpected-keyword-arg
                return self.explainer.interval_learner.predict_probability(x, threshold, bins=bins)
            except:  # pylint: disable=bare-except
                # Use bare except + sys.exc_info to avoid catching 'Exception' explicitly (ADR-002)
                exc = sys.exc_info()[1]
                if not isinstance(exc, Exception):
                    raise

                if self.explainer.suppress_crepes_errors:
                    warnings.warn(
                        "crepes produced an unexpected result while computing probabilities; "
                        "returning zeros as a degraded fallback.",
                        UserWarning,
                        stacklevel=2,
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

    def _compute_weight_delta(self, baseline, perturbed) -> np.ndarray:
        """Return the contribution weight delta between *baseline* and *perturbed*."""
        baseline_arr = np.asarray(baseline)
        perturbed_arr = np.asarray(perturbed)

        if baseline_arr.shape == ():
            return np.asarray(baseline_arr - perturbed_arr, dtype=float)

        if baseline_arr.shape != perturbed_arr.shape:
            with contextlib.suppress(ValueError):
                baseline_arr = np.broadcast_to(baseline_arr, perturbed_arr.shape)

        with contextlib.suppress(TypeError, ValueError):
            return np.asarray(baseline_arr - perturbed_arr, dtype=float)

        # Fallback for object arrays or incompatible types
        # Note: If fallback fails, we allow its exception to propagate (ADR-002)
        baseline_flat = np.asarray(baseline, dtype=object).reshape(-1)
        perturbed_flat = np.asarray(perturbed, dtype=object).reshape(-1)
        deltas = np.empty_like(perturbed_flat, dtype=float)
        for idx, (pert_value, base_value) in enumerate(zip(perturbed_flat, baseline_flat)):
            delta_value = assign_weight(pert_value, base_value)
            delta_array = np.asarray(delta_value, dtype=float).reshape(-1)
            deltas[idx] = float(delta_array[0])
        return deltas.reshape(perturbed_arr.shape)

    def _ensure_interval_runtime_state(self) -> None:
        """Ensure interval tracking members exist for legacy instances."""
        if not self.explainer._interval_plugin_hints:
            self.explainer._interval_plugin_hints = {}
        if not self.explainer._interval_plugin_fallbacks:
            self.explainer._interval_plugin_fallbacks = {}
        if not self.explainer._interval_plugin_identifiers:
            self.explainer._interval_plugin_identifiers = {"default": None, "fast": None}
        if not self.explainer._telemetry_interval_sources:
            self.explainer._telemetry_interval_sources = {"default": None, "fast": None}
        if not self.explainer._interval_preferred_identifier:
            self.explainer._interval_preferred_identifier = {"default": None, "fast": None}
        if not self.explainer._interval_context_metadata:
            self.explainer._interval_context_metadata = {"default": {}, "fast": {}}

    def _gather_interval_hints(self, *, fast: bool) -> Tuple[str, ...]:
        """Return interval dependency hints collected from explanation plugins."""
        if fast:
            return self.explainer._interval_plugin_hints.get("fast", ())
        ordered: List[str] = []
        seen: set[str] = set()
        for mode in ("factual", "alternative"):
            for identifier in self.explainer._interval_plugin_hints.get(mode, ()):  # noqa: B020
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

        modes = coerce_string_tuple(metadata.get("modes"))
        if not modes:
            return f"{prefix}: plugin metadata missing modes declaration"
        required_mode = "regression" if "regression" in self.explainer.mode else "classification"
        if required_mode not in modes:
            declared = ", ".join(modes)
            return f"{prefix}: does not support mode '{required_mode}' (modes: {declared})"

        capabilities = set(coerce_string_tuple(metadata.get("capabilities")))
        required_cap = (
            "interval:regression"
            if "regression" in self.explainer.mode
            else "interval:classification"
        )
        if required_cap not in capabilities:
            declared = ", ".join(sorted(capabilities)) or "<none>"
            return f"{prefix}: missing capability '{required_cap}' (capabilities: {declared})"

        if fast and not bool(metadata.get("fast_compatible")):
            return f"{prefix}: not marked fast_compatible"
        if metadata.get("requires_bins") and self.explainer.bins is None:
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
            self.explainer._fast_interval_plugin_override
            if fast
            else self.explainer._interval_plugin_override
        )
        override = self.explainer._plugin_manager.coerce_plugin_override(raw_override)
        if override is not None and not isinstance(override, str):
            identifier = getattr(override, "plugin_meta", {}).get("name")
            return override, identifier

        if isinstance(raw_override, str):
            preferred_identifier = raw_override
        else:
            key = "fast" if fast else "default"
            preferred_identifier = self.explainer._interval_preferred_identifier.get(key)
        chain = list(
            self.explainer._interval_plugin_fallbacks.get("fast" if fast else "default", ())
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

            plugin = self.explainer._instantiate_plugin(plugin)
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
        calibration_splits: Tuple[Any, ...] = ((self.explainer.x_cal, self.explainer.y_cal),)
        bins = {"calibration": self.explainer.bins}
        difficulty = {"estimator": self.explainer.difficulty_estimator}
        fast_flags = {"fast": fast}
        residuals: Mapping[str, Any] = {}
        key = "fast" if fast else "default"
        stored_metadata = dict(self.explainer._interval_context_metadata.get(key, {}))
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
                "noise_type": getattr(self.explainer, "_CalibratedExplainer__noise_type", None),
                "scale_factor": getattr(self.explainer, "_CalibratedExplainer__scale_factor", None),
                "severity": getattr(self.explainer, "_CalibratedExplainer__severity", None),
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
            if not existing_fast and isinstance(self.explainer.interval_learner, list):
                existing_fast = tuple(self.explainer.interval_learner)
            if existing_fast:
                enriched_metadata["existing_fast_calibrators"] = tuple(existing_fast)
        return IntervalCalibratorContext(
            learner=self.explainer.learner,
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
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            exc = sys.exc_info()[1]
            raise ConfigurationError(
                f"Interval plugin execution failed for {'fast' if fast else 'default'} mode: {exc}"
            ) from exc
        self._capture_interval_calibrators(
            context=context,
            calibrator=calibrator,
            fast=fast,
        )
        key = "fast" if fast else "default"
        self.explainer._interval_plugin_identifiers[key] = identifier
        self.explainer._telemetry_interval_sources[key] = identifier
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
        self.explainer._interval_context_metadata[key] = dict(metadata_dict)
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
