"""In-tree plugin implementations shipped with the package.

This module hosts the default interval, explanation, and plot plugins that
mirror the legacy behaviour of :class:`CalibratedExplainer`. They act as thin
wrappers over the existing in-tree implementations so that the new plugin
contracts defined in ADR-013/ADR-014/ADR-015 can be exercised without changing
runtime semantics.

The implementations intentionally keep the adapters lightweight: they delegate
to the original methods and expose the resulting objects through
``ExplanationBatch`` payloads. The batches embed the original collection so that
``CalibratedExplanations.from_batch`` can return the familiar container without
having to rehydrate individual explanations.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .. import __version__ as package_version

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checking
    pass
from ..explanations.explanation import (
    AlternativeExplanation,
    FactualExplanation,
    FastExplanation,
)
from ..explanations.explanation import (
    CalibratedExplanation as _AbstractExplanation,
)
from ..explanations.explanations import CalibratedExplanations
from ..utils.helper import safe_isinstance
from ..utils.perturbation import perturb_dataset
from .explanations import (
    ExplanationBatch,
    ExplanationContext,
    ExplanationPlugin,
    ExplanationRequest,
)
from .intervals import IntervalCalibratorContext, IntervalCalibratorPlugin
from .plots import PlotBuilder, PlotRenderContext, PlotRenderer, PlotRenderResult
from .predict import PredictBridge
from .registry import (
    find_explanation_descriptor,
    find_interval_descriptor,
    register_explanation_plugin,
    register_interval_plugin,
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
)


def _derive_threshold_labels(threshold: Any) -> tuple[str, str]:
    """Produce positive/negative labels for thresholded regression."""
    try:
        if (
            isinstance(threshold, Sequence)
            and not isinstance(threshold, (str, bytes))
            and len(threshold) >= 2
        ):
            lo = float(threshold[0])
            hi = float(threshold[1])
            return (f"{lo:.2f} <= Y < {hi:.2f}", "Outside interval")
    except Exception as exc:
        logging.getLogger(__name__).debug("Failed to parse threshold as interval: %s", exc)
    try:
        value = float(threshold)
    except Exception:
        return ("Target within threshold", "Outside threshold")
    return (f"Y < {value:.2f}", f"Y ≥ {value:.2f}")


class LegacyPredictBridge(PredictBridge):
    """Predict bridge delegating to :class:`CalibratedExplainer` methods."""

    def __init__(self, explainer: Any) -> None:
        """Store the wrapped explainer used for legacy compatibility calls."""
        self._explainer = explainer

    def predict(
        self,
        x: Any,
        *,
        mode: str,
        task: str,
        bins: Any | None = None,
    ) -> Mapping[str, Any]:
        """Return calibrated predictions routed through the wrapped explainer."""
        prediction = self._explainer.predict(x, uq_interval=True, bins=bins)
        if isinstance(prediction, tuple):
            preds, interval = prediction
            low, high = interval
        else:
            preds = prediction
            low = high = None
        payload: dict[str, Any] = {
            "predict": np.asarray(preds),
            "mode": mode,
            "task": task,
        }
        if low is not None and high is not None:
            payload["low"] = np.asarray(low)
            payload["high"] = np.asarray(high)
        if task == "classification":
            payload["classes"] = np.asarray(self._explainer.predict(x, calibrated=True, bins=bins))
        return payload

    def predict_interval(
        self, x: Any, *, task: str, bins: Any | None = None
    ):  # pragma: no cover - passthrough
        """Return calibrated prediction intervals for ``x``."""
        return self._explainer.predict(x, uq_interval=True, calibrated=True, bins=bins)

    def predict_proba(self, x: Any, bins: Any | None = None):  # pragma: no cover - passthrough
        """Return calibrated probabilities for ``x`` when available."""
        return self._explainer.predict_proba(x, uq_interval=True, calibrated=True, bins=bins)


def _supports_calibrated_explainer(model: Any) -> bool:
    """Return ``True`` when *model* is a ``CalibratedExplainer`` instance."""
    return safe_isinstance(
        model, "calibrated_explanations.core.calibrated_explainer.CalibratedExplainer"
    )


def _collection_to_batch(collection: CalibratedExplanations) -> ExplanationBatch:
    """Convert a legacy explanation collection into an :class:`ExplanationBatch`."""
    explanation_cls: type[_AbstractExplanation]
    if collection.explanations:
        explanation_cls = type(collection.explanations[0])
    else:
        explanation_cls = FactualExplanation
    instances = tuple({"explanation": exp} for exp in collection.explanations)
    metadata = {
        "container": collection,
        "mode": getattr(collection, "mode", None),
    }
    return ExplanationBatch(
        container_cls=type(collection),
        explanation_cls=explanation_cls,
        instances=instances,
        collection_metadata=metadata,
    )


class LegacyIntervalCalibratorPlugin(IntervalCalibratorPlugin):
    """Wrapper returning the already-initialised legacy calibrator."""

    plugin_meta = {
        "name": "core.interval.legacy",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["interval:classification", "interval:regression"],
        "modes": ("classification", "regression"),
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "legacy",
        "legacy_compatible": True,
    }

    def create(self, context: IntervalCalibratorContext, *, fast: bool = False) -> Any:
        """Instantiate the legacy interval calibrator for the supplied context."""
        task = str(context.metadata.get("task") or context.metadata.get("mode") or "")
        learner = context.learner
        bins = context.bins.get("calibration")
        difficulty = context.difficulty.get("estimator")
        x_cal, y_cal = context.calibration_splits[0]
        if "regression" in task:
            from ..calibration.interval_regressor import IntervalRegressor

            explainer = context.metadata.get("explainer")
            if explainer is None:
                raise RuntimeError("Legacy interval context missing 'explainer' handle")
            calibrator = IntervalRegressor(explainer)
        else:
            from ..calibration.venn_abers import VennAbers

            predict_function = context.metadata.get("predict_function")
            if predict_function is None:
                explainer = context.metadata.get("explainer")
                if explainer is None:
                    raise RuntimeError("Legacy interval context missing 'predict_function' entry")
                predict_function = getattr(explainer, "predict_function", None)
            calibrator = VennAbers(
                x_cal,
                y_cal,
                learner,
                bins,
                difficulty_estimator=difficulty,
                predict_function=predict_function,
            )
        if isinstance(context.metadata, dict):
            context.metadata.setdefault("calibrator", calibrator)
        return calibrator


@dataclass
class _LegacyExplanationBase(ExplanationPlugin):
    """Shared adapter logic for legacy explanation flows."""

    _mode: str
    _explanation_attr: str
    _expected_cls: type[_AbstractExplanation]

    plugin_meta: Mapping[str, Any]
    _context: ExplanationContext | None = None
    _bridge: PredictBridge | None = None
    _explainer: Any | None = None

    def supports(self, model: Any) -> bool:
        """Return True when the legacy plugin can handle the supplied model instance."""
        return _supports_calibrated_explainer(model)

    def explain(self, model: Any, x: Any, **kwargs: Any) -> Any:  # pragma: no cover - legacy
        """Dispatch to the underlying explainer for single-instance explanations."""
        if not self.supports(model):
            raise ValueError("Unsupported model for legacy plugin")
        explanation_callable = getattr(model, self._explanation_attr)
        return explanation_callable(x, **kwargs)

    def supports_mode(self, mode: str, *, task: str) -> bool:
        """Return True when the plugin implements the requested explanation mode."""
        return mode == self._mode

    def initialize(self, context: ExplanationContext) -> None:
        """Capture context dependencies required by legacy explanation flows."""
        self._context = context
        self._bridge = context.predict_bridge
        self._explainer = context.helper_handles.get("explainer")
        if self._explainer is None:
            raise RuntimeError("Explanation context missing 'explainer' handle")

    def explain_batch(self, x: Any, request: ExplanationRequest) -> ExplanationBatch:
        """Execute the explanation call and adapt legacy collections into batches."""
        if self._context is None or self._bridge is None or self._explainer is None:
            raise RuntimeError("Plugin must be initialised before use")

        # Exercise the predict bridge lifecycle. The results are not used further
        # but calling the bridge ensures the contract is honoured.
        self._bridge.predict(
            x,
            mode=self._mode,
            task=self._context.task,
            bins=request.bins,
        )

        explanation_callable = getattr(self._explainer, self._explanation_attr)

        kwargs = {
            "threshold": request.threshold,
            "low_high_percentiles": request.low_high_percentiles,
            "bins": request.bins,
        }
        if self._mode != "fast":
            kwargs["features_to_ignore"] = request.features_to_ignore
        kwargs["_use_plugin"] = False

        collection: CalibratedExplanations = explanation_callable(x, **kwargs)
        return _collection_to_batch(collection)


class LegacyFactualExplanationPlugin(_LegacyExplanationBase):
    """Plugin wrapping ``CalibratedExplainer.explain_factual``."""

    plugin_meta = {
        "name": "core.explanation.factual",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:classification",
            "task:regression",
        ],
        "modes": ("factual",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
    }

    def __init__(self) -> None:
        """Configure the plugin to proxy factual explanation calls."""
        super().__init__(
            _mode="factual",
            _explanation_attr="explain_factual",
            _expected_cls=FactualExplanation,
            plugin_meta=self.plugin_meta,
        )


class LegacyAlternativeExplanationPlugin(_LegacyExplanationBase):
    """Plugin wrapping ``CalibratedExplainer.explore_alternatives``."""

    plugin_meta = {
        "name": "core.explanation.alternative",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:alternative",
            "task:classification",
            "task:regression",
        ],
        "modes": ("alternative",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
    }

    def __init__(self) -> None:
        """Configure the plugin to proxy alternative explanation calls."""
        super().__init__(
            _mode="alternative",
            _explanation_attr="explore_alternatives",
            _expected_cls=AlternativeExplanation,
            plugin_meta=self.plugin_meta,
        )


class _ExecutionExplanationPluginBase(_LegacyExplanationBase):
    """Base class for wrappers that delegate to execution plugins.

    This class bridges the explanation plugin layer (high-level user API)
    with the execution plugin layer (low-level algorithm selection).
    It handles conversion between ExplanationRequest and ExplainRequest APIs,
    and provides graceful fallback to legacy implementation if execution fails.
    """

    _execution_plugin_class: type | None = None

    def explain_batch(self, x: Any, request: ExplanationRequest) -> ExplanationBatch:
        """Execute the explanation call with fallback to legacy.

        Attempts to use the execution plugin class, then falls back to the
        legacy explanation path if the executor is unavailable or execution fails.
        """
        if self._context is None or self._bridge is None or self._explainer is None:
            raise RuntimeError("Plugin must be initialised before use")

        if self._execution_plugin_class is None:
            raise RuntimeError("Execution plugin class not configured")

        try:
            # Import here to avoid circular imports
            from ..core.explain._shared import ExplainConfig
            from ..core.explain._shared import ExplainRequest as _ExplainRequest

            # Build the execute request from the explanation request
            explain_request = _ExplainRequest(
                x=x,
                threshold=request.threshold,
                low_high_percentiles=request.low_high_percentiles or (5, 95),
                bins=request.bins,
                features_to_ignore=np.asarray(request.features_to_ignore or []),
                use_plugin=False,
                skip_instance_parallel=False,
            )

            # Build execution config from explainer state
            calibration_data = {}
            if hasattr(self._explainer, "_get_calibration_summaries"):
                calibration_data = self._explainer._get_calibration_summaries()[1]

            explain_config = ExplainConfig(
                executor=getattr(self._explainer, "executor", None),
                granularity=getattr(self._explainer, "granularity", "feature"),
                num_features=self._explainer.num_features,
                categorical_features=self._explainer.categorical_features or (),
                feature_values=calibration_data,
                mode=self._context.task,
            )

            # Instantiate and execute the plugin
            plugin = self._execution_plugin_class()
            collection = plugin.execute(explain_request, explain_config, self._explainer)

        except Exception as exc:
            # Fallback to legacy implementation with warning
            _logger = logging.getLogger(__name__)
            _logger.warning(
                "Execution plugin failed for mode '%s': %s; falling back to legacy",
                self._mode,
                exc,
            )
            explanation_callable = getattr(self._explainer, self._explanation_attr)
            kwargs = {
                "threshold": request.threshold,
                "low_high_percentiles": request.low_high_percentiles,
                "bins": request.bins,
            }
            if self._mode != "fast":
                kwargs["features_to_ignore"] = request.features_to_ignore
            kwargs["_use_plugin"] = False
            collection = explanation_callable(x, **kwargs)

        return _collection_to_batch(collection)


class SequentialExplanationPlugin(_ExecutionExplanationPluginBase):
    """Wrapper for sequential execution strategy (factual mode).

    Enables users to explicitly select single-threaded sequential processing
    through the plugin configuration system.
    """

    plugin_meta = {
        "name": "core.explanation.factual.sequential",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:classification",
            "task:regression",
        ],
        "modes": ("factual",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
        "fallbacks": ("core.explanation.factual",),
    }

    def __init__(self) -> None:
        """Configure the plugin to use sequential execution."""
        from ..core.explain.sequential import SequentialExplainExecutor

        self._execution_plugin_class = SequentialExplainExecutor
        super().__init__(
            _mode="factual",
            _explanation_attr="explain_factual",
            _expected_cls=FactualExplanation,
            plugin_meta=self.plugin_meta,
        )


class FeatureParallelExplanationPlugin(_ExecutionExplanationPluginBase):
    """Wrapper for feature-parallel execution strategy (factual mode).

    Enables users to select feature-level parallelism through the plugin
    configuration system. Falls back to sequential if executor unavailable.
    """

    plugin_meta = {
        "name": "core.explanation.factual.feature_parallel",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:classification",
            "task:regression",
        ],
        "modes": ("factual",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
        "fallbacks": ("core.explanation.factual.sequential", "core.explanation.factual"),
    }

    def __init__(self) -> None:
        """Configure the plugin to use feature-parallel execution."""
        from ..core.explain.parallel_feature import FeatureParallelExplainExecutor

        self._execution_plugin_class = FeatureParallelExplainExecutor
        super().__init__(
            _mode="factual",
            _explanation_attr="explain_factual",
            _expected_cls=FactualExplanation,
            plugin_meta=self.plugin_meta,
        )


class InstanceParallelExplanationPlugin(_ExecutionExplanationPluginBase):
    """Wrapper for instance-parallel execution strategy (factual mode).

    Enables users to select instance-level parallelism through the plugin
    configuration system. Falls back to feature-parallel if executor unavailable.
    """

    plugin_meta = {
        "name": "core.explanation.factual.instance_parallel",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:classification",
            "task:regression",
        ],
        "modes": ("factual",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
        "fallbacks": (
            "core.explanation.factual.feature_parallel",
            "core.explanation.factual.sequential",
            "core.explanation.factual",
        ),
    }

    def __init__(self) -> None:
        """Configure the plugin to use instance-parallel execution."""
        from ..core.explain.parallel_instance import InstanceParallelExplainExecutor

        self._execution_plugin_class = InstanceParallelExplainExecutor
        super().__init__(
            _mode="factual",
            _explanation_attr="explain_factual",
            _expected_cls=FactualExplanation,
            plugin_meta=self.plugin_meta,
        )


class SequentialAlternativeExplanationPlugin(_ExecutionExplanationPluginBase):
    """Wrapper for sequential execution strategy (alternative mode).

    Enables users to explicitly select single-threaded sequential processing
    for alternative explanations through the plugin configuration system.
    """

    plugin_meta = {
        "name": "core.explanation.alternative.sequential",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:alternative",
            "task:classification",
            "task:regression",
        ],
        "modes": ("alternative",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
        "fallbacks": ("core.explanation.alternative",),
    }

    def __init__(self) -> None:
        """Configure the plugin to use sequential execution."""
        from ..core.explain.sequential import SequentialExplainExecutor

        self._execution_plugin_class = SequentialExplainExecutor
        super().__init__(
            _mode="alternative",
            _explanation_attr="explore_alternatives",
            _expected_cls=AlternativeExplanation,
            plugin_meta=self.plugin_meta,
        )


class FeatureParallelAlternativeExplanationPlugin(_ExecutionExplanationPluginBase):
    """Wrapper for feature-parallel execution strategy (alternative mode).

    Enables users to select feature-level parallelism for alternative explanations
    through the plugin configuration system. Falls back to sequential if executor unavailable.
    """

    plugin_meta = {
        "name": "core.explanation.alternative.feature_parallel",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:alternative",
            "task:classification",
            "task:regression",
        ],
        "modes": ("alternative",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
        "fallbacks": ("core.explanation.alternative.sequential", "core.explanation.alternative"),
    }

    def __init__(self) -> None:
        """Configure the plugin to use feature-parallel execution."""
        from ..core.explain.parallel_feature import FeatureParallelExplainExecutor

        self._execution_plugin_class = FeatureParallelExplainExecutor
        super().__init__(
            _mode="alternative",
            _explanation_attr="explore_alternatives",
            _expected_cls=AlternativeExplanation,
            plugin_meta=self.plugin_meta,
        )


class InstanceParallelAlternativeExplanationPlugin(_ExecutionExplanationPluginBase):
    """Wrapper for instance-parallel execution strategy (alternative mode).

    Enables users to select instance-level parallelism for alternative explanations
    through the plugin configuration system. Falls back to feature-parallel if executor unavailable.
    """

    plugin_meta = {
        "name": "core.explanation.alternative.instance_parallel",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": [
            "explain",
            "explanation:alternative",
            "task:classification",
            "task:regression",
        ],
        "modes": ("alternative",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.legacy", "legacy"),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": True,
        "trust": {"trusted": True},
        "fallbacks": (
            "core.explanation.alternative.feature_parallel",
            "core.explanation.alternative.sequential",
            "core.explanation.alternative",
        ),
    }

    def __init__(self) -> None:
        """Configure the plugin to use instance-parallel execution."""
        from ..core.explain.parallel_instance import InstanceParallelExplainExecutor

        self._execution_plugin_class = InstanceParallelExplainExecutor
        super().__init__(
            _mode="alternative",
            _explanation_attr="explore_alternatives",
            _expected_cls=AlternativeExplanation,
            plugin_meta=self.plugin_meta,
        )


class LegacyPlotBuilder(PlotBuilder):
    """Minimal plot builder that keeps legacy behaviour."""

    plugin_meta = {
        "name": "core.plot.legacy.builder",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["plot:builder"],
        "style": "legacy",
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "output_formats": ["png"],
        "legacy_compatible": True,
    }

    def build(self, context: PlotRenderContext) -> Mapping[str, Any]:
        """Return a legacy-compatible payload representing the plot request."""
        return {"context": context}


class LegacyPlotRenderer(PlotRenderer):
    """Minimal renderer mirroring the legacy matplotlib pathway."""

    plugin_meta = {
        "name": "core.plot.legacy.renderer",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["plot:renderer"],
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "output_formats": ["png"],
        "supports_interactive": False,
    }

    def render(
        self, artifact: Mapping[str, Any], *, context: PlotRenderContext
    ) -> PlotRenderResult:
        """Render the placeholder legacy artefact and return an empty result."""
        return PlotRenderResult(artifact=artifact, figure=None, saved_paths=(), extras={})


class PlotSpecDefaultBuilder(PlotBuilder):
    """PlotSpec-first builder for global plots."""

    plugin_meta = {
        "name": "core.plot.plot_spec.default.builder",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["plot:builder"],
        "style": "plot_spec.default",
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "legacy_compatible": True,
        "output_formats": ["png", "svg", "pdf"],
    }

    def build(self, context: PlotRenderContext) -> Mapping[str, Any]:
        """Construct a PlotSpec-compatible payload for global plots."""
        intent = context.intent if isinstance(context.intent, Mapping) else {}
        intent_type = intent.get("type") if isinstance(intent, Mapping) else None
        if intent_type == "global":
            options = context.options if isinstance(context.options, Mapping) else {}
            payload = options.get("payload", {})
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    "PlotSpec default builder expected payload mapping for global plot"
                )
            payload_dict = dict(payload)
            payload_dict.pop("threshold", None)  # legacy-only field
            if "y" in payload_dict and "y_test" not in payload_dict:
                payload_dict["y_test"] = payload_dict.pop("y")
            from ..viz.builders import build_global_plotspec_dict

            title = intent.get("title") if isinstance(intent, Mapping) else None
            return build_global_plotspec_dict(title=title, **payload_dict)

        if intent_type == "alternative":
            options = context.options if isinstance(context.options, Mapping) else {}
            payload = options.get("payload", {})
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    "PlotSpec default builder expected payload mapping for alternative plot"
                )

            feature_payload = payload.get("feature_predict")
            if feature_payload is None:
                feature_payload = payload.get("feature_weights")
            if feature_payload is None:
                raise RuntimeError(
                    "Alternative plot payload must supply 'feature_predict' or 'feature_weights'"
                )

            predict_payload = payload.get("predict", {})
            if not isinstance(predict_payload, Mapping):
                predict_payload = dict(predict_payload or {})
            else:
                predict_payload = dict(predict_payload)

            def _safe_float(value: Any) -> float | None:
                try:
                    val = float(value)
                except Exception:
                    return None
                if not np.isfinite(val):
                    return None
                return val

            y_minmax = payload.get("y_minmax")
            if y_minmax is None and context.explanation is not None:
                with contextlib.suppress(Exception):
                    candidate = getattr(context.explanation, "y_minmax", None)
                    if candidate is not None:
                        y_minmax = candidate
            normalised_y_minmax: tuple[float, float] | None = None
            if isinstance(y_minmax, Sequence) and len(y_minmax) >= 2:
                try:
                    y0, y1 = float(y_minmax[0]), float(y_minmax[1])
                except Exception:
                    normalised_y_minmax = None
                else:
                    if np.isfinite(y0) and np.isfinite(y1):
                        normalised_y_minmax = (y0, y1)

            base_pred = _safe_float(predict_payload.get("predict"))
            if base_pred is None:
                base_pred = 0.0

            low_default = base_pred
            high_default = base_pred
            if normalised_y_minmax is not None:
                low_default = normalised_y_minmax[0]
                high_default = normalised_y_minmax[1]

            pred_low = _safe_float(predict_payload.get("low"))
            pred_high = _safe_float(predict_payload.get("high"))

            predict_payload["predict"] = base_pred
            predict_payload["low"] = pred_low if pred_low is not None else low_default
            predict_payload["high"] = pred_high if pred_high is not None else high_default

            def _normalise_sequence(values: Any, fallback: float) -> list[float]:
                if isinstance(values, np.ndarray):
                    seq = values.tolist()
                elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    seq = list(values)
                else:
                    seq = [values]
                normalised: list[float] = []
                for item in seq:
                    val = _safe_float(item)
                    normalised.append(val if val is not None else float(fallback))
                return normalised

            feature_count = 0
            fallback_map = {
                "predict": predict_payload["predict"],
                "low": predict_payload["low"],
                "high": predict_payload["high"],
            }
            if isinstance(feature_payload, Mapping):
                sanitised: dict[str, list[float]] = {}
                for key, values in feature_payload.items():
                    fallback_value = fallback_map.get(key, predict_payload["predict"])
                    sanitised[key] = _normalise_sequence(values, fallback_value)
                    feature_count = max(feature_count, len(sanitised[key]))
                feature_payload = sanitised
            else:
                normalised_seq = _normalise_sequence(feature_payload, predict_payload["predict"])
                feature_payload = normalised_seq
                feature_count = len(normalised_seq)

            features_to_plot_raw = payload.get("features_to_plot", ())
            features_to_plot: list[int] = []
            if feature_count:
                for idx in features_to_plot_raw:
                    try:
                        value = int(idx)
                    except Exception as exc:
                        logging.getLogger(__name__).debug(
                            "Failed to convert feature index to int: %s", exc
                        )
                        continue
                    if 0 <= value < feature_count:
                        features_to_plot.append(value)
                if not features_to_plot:
                    features_to_plot = list(range(feature_count))

            column_names = payload.get("column_names")
            if column_names is None:
                column_names = payload.get("feature_names")
            if column_names is not None:
                column_names = list(column_names)
            elif feature_count:
                column_names = [str(idx) for idx in range(feature_count)]

            rule_labels = payload.get("rule_labels")
            rule_labels = list(rule_labels) if rule_labels is not None else column_names

            instance_values = payload.get("instance")
            if instance_values is None:
                instance_values = payload.get("instance_values")

            interval = payload.get("interval")
            if interval is None:
                interval = (
                    isinstance(feature_payload, Mapping)
                    and "low" in feature_payload
                    and "high" in feature_payload
                )

            sort_by = payload.get("sort_by")
            ascending = bool(payload.get("ascending", False))
            legacy_behavior = payload.get("legacy_solid_behavior", True)
            uncertainty_color = payload.get("uncertainty_color")
            uncertainty_alpha = payload.get("uncertainty_alpha")
            is_thresholded = False
            threshold_label: str | None = None
            if context.explanation is not None:
                with contextlib.suppress(Exception):
                    is_thresholded = bool(context.explanation.is_thresholded())
                if is_thresholded:
                    y_threshold = getattr(context.explanation, "y_threshold", None)
                    if isinstance(y_threshold, tuple) and len(y_threshold) >= 2:
                        try:
                            lo_val = float(y_threshold[0])
                            hi_val = float(y_threshold[1])
                        except Exception:
                            threshold_label = None
                        else:
                            threshold_label = (
                                f"Probability of target being between {lo_val:.3f} and {hi_val:.3f}"
                            )
                    elif y_threshold is not None:
                        try:
                            thr = float(y_threshold)
                        except Exception:
                            threshold_label = None
                        else:
                            threshold_label = f"Probability of target being below {thr:.2f}"

            variant_hint = str(
                intent.get("mode")
                or intent.get("variant")
                or intent.get("task")
                or payload.get("mode")
                or payload.get("variant")
                or payload.get("task")
                or ""
            ).lower()
            if not variant_hint and context.explanation is not None:
                with contextlib.suppress(Exception):
                    maybe_mode = context.explanation.get_mode()
                    if isinstance(maybe_mode, str):
                        variant_hint = maybe_mode.lower()

            from ..viz.builders import (
                build_alternative_probabilistic_spec,
                build_alternative_regression_spec,
            )

            builder_kwargs = {
                "title": intent.get("title") if isinstance(intent, Mapping) else None,
                "predict": predict_payload,
                "feature_weights": feature_payload,
                "features_to_plot": features_to_plot,
                "column_names": column_names,
                "rule_labels": rule_labels,
                "instance": instance_values,
                "y_minmax": normalised_y_minmax,
                "interval": bool(interval),
                "sort_by": sort_by,
                "ascending": ascending,
                "legacy_solid_behavior": bool(legacy_behavior),
                "uncertainty_color": uncertainty_color,
                "uncertainty_alpha": uncertainty_alpha,
                "threshold_value": getattr(context.explanation, "y_threshold", None)
                if is_thresholded
                else None,
                "is_thresholded": is_thresholded,
                "threshold_label": threshold_label,
            }

            if "regression" in variant_hint:
                thresholded = builder_kwargs.pop("is_thresholded")
                threshold_label_text = builder_kwargs.pop("threshold_label")
                threshold_value = builder_kwargs.pop("threshold_value")
                if thresholded:
                    pos_label, neg_label = _derive_threshold_labels(threshold_value)
                    classification_kwargs = builder_kwargs.copy()
                    classification_kwargs.pop("xlabel", None)
                    classification_kwargs.pop("xlim", None)
                    classification_kwargs.pop("xticks", None)
                    classification_kwargs["neg_label"] = neg_label
                    classification_kwargs["pos_label"] = pos_label
                    classification_kwargs["xlabel"] = threshold_label_text or "Probability"
                    classification_kwargs["xlim"] = (0.0, 1.0)
                    classification_kwargs["xticks"] = [float(x) for x in np.linspace(0.0, 1.0, 11)]
                    classification_kwargs["y_minmax"] = None
                    return build_alternative_probabilistic_spec(**classification_kwargs)
                else:
                    builder_kwargs.pop("threshold_value", None)
                    return build_alternative_regression_spec(**builder_kwargs)

            builder_kwargs.update(
                {
                    "neg_label": payload.get("neg_label"),
                    "pos_label": payload.get("pos_label"),
                }
            )
            builder_kwargs.pop("threshold_value", None)
            builder_kwargs.pop("is_thresholded", None)
            builder_kwargs.pop("threshold_label", None)
            return build_alternative_probabilistic_spec(**builder_kwargs)

        raise RuntimeError("PlotSpec default builder currently supports only global plots")


class PlotSpecDefaultRenderer(PlotRenderer):
    """Renderer delegating PlotSpec payloads to the matplotlib adapter."""

    plugin_meta = {
        "name": "core.plot.plot_spec.default.renderer",
        "schema_version": 1,
        "version": package_version,
        "provider": "calibrated_explanations",
        "capabilities": ["plot:renderer"],
        "dependencies": (),
        "trusted": True,
        "trust": {"trusted": True},
        "output_formats": ["png", "svg", "pdf"],
        "supports_interactive": False,
    }

    def render(
        self, artifact: Mapping[str, Any], *, context: PlotRenderContext
    ) -> PlotRenderResult:
        """Render a PlotSpec artefact via the matplotlib adapter."""
        from ..viz.matplotlib_adapter import render as render_plotspec

        saved: list[str] = []
        save_ext = context.save_ext
        base_path = context.path

        try:
            if save_ext:
                for ext in save_ext if isinstance(save_ext, (list, tuple)) else (save_ext,):
                    target = f"{base_path}{ext}" if base_path else ext
                    render_plotspec(artifact, show=False, save_path=target)
                    saved.append(target)
                if context.show:
                    render_plotspec(artifact, show=True, save_path=None)
            else:
                render_plotspec(artifact, show=context.show, save_path=base_path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError(f"PlotSpec renderer failed: {exc}") from exc
        return PlotRenderResult(artifact=artifact, figure=None, saved_paths=tuple(saved), extras={})


def _register_builtin_fast_plugins() -> None:
    """Register in-tree fallbacks for FAST plugins when extras are unavailable."""
    if find_interval_descriptor("core.interval.fast") is None:

        class BuiltinFastIntervalCalibratorPlugin(IntervalCalibratorPlugin):
            """FAST interval plugin mirroring the external implementation."""

            plugin_meta = {
                "name": "core.interval.fast",
                "schema_version": 1,
                "version": package_version,
                "provider": "calibrated_explanations",
                "capabilities": [
                    "interval:classification",
                    "interval:regression",
                ],
                "modes": ("classification", "regression"),
                "dependencies": (),
                "trusted": True,
                "trust": {"trusted": True},
                "fast_compatible": True,
                "requires_bins": False,
                "confidence_source": "fast",
                "legacy_compatible": True,
            }

            def create(
                self,
                context: IntervalCalibratorContext,
                *,
                fast: bool = True,
            ) -> Any:
                """Return the FAST calibrator list prepared from the explainer state."""
                metadata = context.metadata
                task = str(metadata.get("task") or metadata.get("mode") or "")
                explainer = metadata.get("explainer")
                if explainer is None:
                    raise RuntimeError("FAST interval context missing 'explainer' handle")

                x_cal, y_cal = context.calibration_splits[0]
                bins = context.bins.get("calibration")
                learner = context.learner
                difficulty = metadata.get("difficulty_estimator")
                categorical_features = tuple(metadata.get("categorical_features", ()))
                noise_cfg = metadata.get("noise_config", {})
                (
                    explainer.fast_x_cal,
                    explainer.scaled_x_cal,
                    explainer.scaled_y_cal,
                    scale_factor,
                ) = perturb_dataset(
                    x_cal,
                    y_cal,
                    categorical_features,
                    noise_type=noise_cfg.get("noise_type"),
                    scale_factor=noise_cfg.get("scale_factor"),
                    severity=noise_cfg.get("severity"),
                    seed=noise_cfg.get("seed"),
                    rng=noise_cfg.get("rng"),
                )
                expanded_bins = np.tile(bins.copy(), scale_factor) if bins is not None else None
                original_bins = explainer.bins
                original_x_cal = explainer.x_cal
                original_y_cal = explainer.y_cal
                explainer.bins = expanded_bins

                calibrators: list[Any] = []
                num_features = int(metadata.get("num_features", 0) or 0)
                if "classification" in task:
                    from ..calibration.venn_abers import VennAbers

                    for f in range(num_features):
                        fast_x_cal = explainer.scaled_x_cal.copy()
                        fast_x_cal[:, f] = explainer.fast_x_cal[:, f]
                        calibrators.append(
                            VennAbers(
                                fast_x_cal,
                                explainer.scaled_y_cal,
                                learner,
                                explainer.bins,
                                difficulty_estimator=difficulty,
                            )
                        )
                else:
                    from ..calibration.interval_regressor import IntervalRegressor

                    for f in range(num_features):
                        fast_x_cal = explainer.scaled_x_cal.copy()
                        fast_x_cal[:, f] = explainer.fast_x_cal[:, f]
                        explainer.x_cal = fast_x_cal
                        explainer.y_cal = explainer.scaled_y_cal
                        calibrators.append(IntervalRegressor(explainer))

                explainer.x_cal = original_x_cal
                explainer.y_cal = original_y_cal
                explainer.bins = original_bins

                if "classification" in task:
                    from ..calibration.venn_abers import VennAbers

                    calibrators.append(
                        VennAbers(
                            x_cal,
                            y_cal,
                            learner,
                            bins,
                            difficulty_estimator=difficulty,
                            predict_function=(
                                metadata.get("predict_function")
                                if metadata.get("predict_function") is not None
                                else getattr(explainer, "predict_function", None)
                            ),
                        )
                    )
                else:
                    from ..calibration.interval_regressor import IntervalRegressor

                    calibrators.append(IntervalRegressor(explainer))

                if isinstance(metadata, dict):
                    metadata.setdefault("fast_calibrators", tuple(calibrators))
                return calibrators

        register_interval_plugin("core.interval.fast", BuiltinFastIntervalCalibratorPlugin())

    if find_explanation_descriptor("core.explanation.fast") is None:

        class BuiltinFastExplanationPlugin(_LegacyExplanationBase):
            """Legacy wrapper delegating fast explanations to the explainer."""

            plugin_meta = {
                "name": "core.explanation.fast",
                "schema_version": 1,
                "version": package_version,
                "provider": "calibrated_explanations",
                "capabilities": [
                    "explain",
                    "explanation:fast",
                    "task:classification",
                    "task:regression",
                ],
                "modes": ("fast",),
                "tasks": ("classification", "regression"),
                "dependencies": ("core.interval.fast", "legacy"),
                "interval_dependency": "core.interval.fast",
                "plot_dependency": "legacy",
                "trusted": True,
                "trust": {"trusted": True},
            }

            def __init__(self) -> None:
                super().__init__(
                    _mode="fast",
                    _explanation_attr="explain_fast",
                    _expected_cls=FastExplanation,
                    plugin_meta=self.plugin_meta,
                )

        register_explanation_plugin("core.explanation.fast", BuiltinFastExplanationPlugin())


def _register_builtins() -> None:
    """Register in-tree plugins with the shared registry."""
    register_interval_plugin("core.interval.legacy", LegacyIntervalCalibratorPlugin())

    # Register execution strategy wrappers first (with higher priority in fallback chain)
    register_explanation_plugin(
        "core.explanation.factual.sequential", SequentialExplanationPlugin()
    )
    register_explanation_plugin(
        "core.explanation.factual.feature_parallel", FeatureParallelExplanationPlugin()
    )
    register_explanation_plugin(
        "core.explanation.factual.instance_parallel", InstanceParallelExplanationPlugin()
    )

    register_explanation_plugin(
        "core.explanation.alternative.sequential", SequentialAlternativeExplanationPlugin()
    )
    register_explanation_plugin(
        "core.explanation.alternative.feature_parallel",
        FeatureParallelAlternativeExplanationPlugin(),
    )
    register_explanation_plugin(
        "core.explanation.alternative.instance_parallel",
        InstanceParallelAlternativeExplanationPlugin(),
    )

    # Register legacy plugins as fallback defaults
    register_explanation_plugin("core.explanation.factual", LegacyFactualExplanationPlugin())
    register_explanation_plugin(
        "core.explanation.alternative", LegacyAlternativeExplanationPlugin()
    )

    try:
        import sys
        from pathlib import Path

        # Ensure the repository root is in the path
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from external_plugins.fast_explanations import register as _register_fast_plugins
    except ImportError:
        _register_fast_plugins = None

    if _register_fast_plugins is not None:
        _register_fast_plugins()

    _register_builtin_fast_plugins()

    legacy_builder = LegacyPlotBuilder()
    legacy_renderer = LegacyPlotRenderer()
    register_plot_builder("core.plot.legacy", legacy_builder)
    register_plot_renderer("core.plot.legacy", legacy_renderer)
    register_plot_style(
        "legacy",
        metadata={
            "style": "legacy",
            "builder_id": "core.plot.legacy",
            "renderer_id": "core.plot.legacy",
            "fallbacks": (),
            "legacy_compatible": True,
            "is_default": False,
            "default_for": (),
        },
    )

    plotspec_builder = PlotSpecDefaultBuilder()
    plotspec_renderer = PlotSpecDefaultRenderer()
    register_plot_builder("core.plot.plot_spec.default", plotspec_builder)
    register_plot_renderer("core.plot.plot_spec.default", plotspec_renderer)
    register_plot_style(
        "plot_spec.default",
        metadata={
            "style": "plot_spec.default",
            "builder_id": "core.plot.plot_spec.default",
            "renderer_id": "core.plot.plot_spec.default",
            "fallbacks": ("legacy",),
            "legacy_compatible": True,
            "is_default": True,
            "default_for": ("global", "alternative"),
        },
    )


_register_builtins()

__all__ = [
    "LegacyIntervalCalibratorPlugin",
    "LegacyFactualExplanationPlugin",
    "LegacyAlternativeExplanationPlugin",
    "SequentialExplanationPlugin",
    "FeatureParallelExplanationPlugin",
    "InstanceParallelExplanationPlugin",
    "SequentialAlternativeExplanationPlugin",
    "FeatureParallelAlternativeExplanationPlugin",
    "InstanceParallelAlternativeExplanationPlugin",
    "LegacyPlotBuilder",
    "LegacyPlotRenderer",
    "PlotSpecDefaultBuilder",
    "PlotSpecDefaultRenderer",
    "LegacyPredictBridge",
]

