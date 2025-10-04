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

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..explanations.explanation import (
    AlternativeExplanation,
    CalibratedExplanation as _AbstractExplanation,
    FactualExplanation,
    FastExplanation,
)
from ..explanations.explanations import CalibratedExplanations
from ..utils.helper import safe_isinstance
from .explanations import (
    ExplanationBatch,
    ExplanationContext,
    ExplanationPlugin,
    ExplanationRequest,
)
from .intervals import IntervalCalibratorContext, IntervalCalibratorPlugin
from .plots import PlotBuilder, PlotRenderContext, PlotRenderResult, PlotRenderer
from .predict import PredictBridge
from .registry import (
    register_explanation_plugin,
    register_interval_plugin,
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
)


class LegacyPredictBridge(PredictBridge):
    """Predict bridge delegating to :class:`CalibratedExplainer` methods."""

    def __init__(self, explainer: Any) -> None:
        self._explainer = explainer

    def predict(
        self,
        X: Any,
        *,
        mode: str,
        task: str,
        bins: Any | None = None,
    ) -> Mapping[str, Any]:
        prediction = self._explainer.predict(X, uq_interval=True, bins=bins)
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
            payload["classes"] = np.asarray(
                self._explainer.predict(X, calibrated=True, bins=bins)
            )
        return payload

    def predict_interval(
        self, X: Any, *, task: str, bins: Any | None = None
    ):  # pragma: no cover - passthrough
        return self._explainer.predict(X, uq_interval=True, calibrated=True, bins=bins)

    def predict_proba(self, X: Any, bins: Any | None = None):  # pragma: no cover - passthrough
        return self._explainer.predict_proba(X, uq_interval=True, calibrated=True, bins=bins)


def _supports_calibrated_explainer(model: Any) -> bool:
    """Best-effort runtime check for :class:`CalibratedExplainer`."""

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
        "capabilities": ["interval:classification", "interval:regression"],
        "modes": ("classification", "regression"),
        "dependencies": (),
        "trust": {"trusted": True},
        "fast_compatible": False,
        "requires_bins": False,
        "confidence_source": "legacy",
        "legacy_compatible": True,
    }

    def create(
        self, context: IntervalCalibratorContext, *, fast: bool = False
    ) -> Any:
        calibrator = context.metadata.get("calibrator")
        if calibrator is None:
            raise RuntimeError("Legacy interval context missing 'calibrator' entry")
        return calibrator


class FastIntervalCalibratorPlugin(IntervalCalibratorPlugin):
    """FAST adapter returning the precomputed list of interval learners."""

    plugin_meta = {
        "name": "core.interval.fast",
        "schema_version": 1,
        "capabilities": ["interval:classification", "interval:regression"],
        "modes": ("classification", "regression"),
        "dependencies": (),
        "trust": {"trusted": True},
        "fast_compatible": True,
        "requires_bins": False,
        "confidence_source": "fast",
        "legacy_compatible": True,
    }

    def create(
        self, context: IntervalCalibratorContext, *, fast: bool = True
    ) -> Any:
        calibrator = context.metadata.get("fast_calibrators")
        if calibrator is None:
            raise RuntimeError("FAST interval context missing 'fast_calibrators' entry")
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
        return _supports_calibrated_explainer(model)

    def explain(self, model: Any, X: Any, **kwargs: Any) -> Any:  # pragma: no cover - legacy
        if not self.supports(model):
            raise ValueError("Unsupported model for legacy plugin")
        explanation_callable = getattr(model, self._explanation_attr)
        return explanation_callable(X, **kwargs)

    def supports_mode(self, mode: str, *, task: str) -> bool:
        return mode == self._mode

    def initialize(self, context: ExplanationContext) -> None:
        self._context = context
        self._bridge = context.predict_bridge
        self._explainer = context.helper_handles.get("explainer")
        if self._explainer is None:
            raise RuntimeError("Explanation context missing 'explainer' handle")

    def explain_batch(self, X: Any, request: ExplanationRequest) -> ExplanationBatch:
        if self._context is None or self._bridge is None or self._explainer is None:
            raise RuntimeError("Plugin must be initialised before use")

        # Exercise the predict bridge lifecycle. The results are not used further
        # but calling the bridge ensures the contract is honoured.
        self._bridge.predict(
            X,
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

        collection: CalibratedExplanations = explanation_callable(X, **kwargs)
        return _collection_to_batch(collection)


class LegacyFactualExplanationPlugin(_LegacyExplanationBase):
    """Plugin wrapping ``CalibratedExplainer.explain_factual``."""

    plugin_meta = {
        "name": "core.explanation.factual",
        "schema_version": 1,
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
        "plot_dependency": "legacy",
        "trust": {"trusted": True},
    }

    def __init__(self) -> None:
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
        "plot_dependency": "legacy",
        "trust": {"trusted": True},
    }

    def __init__(self) -> None:
        super().__init__(
            _mode="alternative",
            _explanation_attr="explore_alternatives",
            _expected_cls=AlternativeExplanation,
            plugin_meta=self.plugin_meta,
        )


class FastExplanationPlugin(_LegacyExplanationBase):
    """Plugin wrapping ``CalibratedExplainer.explain_fast``."""

    plugin_meta = {
        "name": "core.explanation.fast",
        "schema_version": 1,
        "capabilities": ["explain", "explanation:fast", "task:classification", "task:regression"],
        "modes": ("fast",),
        "tasks": ("classification", "regression"),
        "dependencies": ("core.interval.fast", "legacy"),
        "interval_dependency": "core.interval.fast",
        "plot_dependency": "legacy",
        "trust": {"trusted": True},
    }

    def __init__(self) -> None:
        super().__init__(
            _mode="fast",
            _explanation_attr="explain_fast",
            _expected_cls=FastExplanation,
            plugin_meta=self.plugin_meta,
        )


class LegacyPlotBuilder(PlotBuilder):
    """Minimal plot builder that keeps legacy behaviour."""

    plugin_meta = {
        "name": "core.plot.legacy.builder",
        "schema_version": 1,
        "capabilities": ["plot:builder"],
        "style": "legacy",
        "dependencies": (),
        "trust": {"trusted": True},
        "output_formats": ["png"],
        "legacy_compatible": True,
    }

    def build(self, context: PlotRenderContext) -> Mapping[str, Any]:
        return {"context": context}


class LegacyPlotRenderer(PlotRenderer):
    """Minimal renderer mirroring the legacy matplotlib pathway."""

    plugin_meta = {
        "name": "core.plot.legacy.renderer",
        "schema_version": 1,
        "capabilities": ["plot:renderer"],
        "dependencies": (),
        "trust": {"trusted": True},
        "output_formats": ["png"],
        "supports_interactive": False,
    }

    def render(
        self, artifact: Mapping[str, Any], *, context: PlotRenderContext
    ) -> PlotRenderResult:
        return PlotRenderResult(artifact=artifact, figure=None, saved_paths=tuple(), extras={})


def _register_builtins() -> None:
    """Register in-tree plugins with the shared registry."""

    register_interval_plugin("core.interval.legacy", LegacyIntervalCalibratorPlugin())
    register_interval_plugin("core.interval.fast", FastIntervalCalibratorPlugin())

    register_explanation_plugin("core.explanation.factual", LegacyFactualExplanationPlugin())
    register_explanation_plugin(
        "core.explanation.alternative", LegacyAlternativeExplanationPlugin()
    )
    register_explanation_plugin("core.explanation.fast", FastExplanationPlugin())

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
        },
    )


_register_builtins()


__all__ = [
    "LegacyIntervalCalibratorPlugin",
    "FastIntervalCalibratorPlugin",
    "LegacyFactualExplanationPlugin",
    "LegacyAlternativeExplanationPlugin",
    "FastExplanationPlugin",
    "LegacyPlotBuilder",
    "LegacyPlotRenderer",
    "LegacyPredictBridge",
]
