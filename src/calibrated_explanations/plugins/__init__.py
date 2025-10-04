"""Explainer plugin interfaces and registry.

This subpackage exposes core plugin protocols and registry utilities covering
explanation, interval, and plotting integrations. The foundational
``ExplainerPlugin`` protocol and ``validate_plugin_meta`` helper derive from
ADR-006, while ADR-013/ADR-014/ADR-015 extend the surface with richer
contracts for calibrated explanations.

Security note: Registering/using third-party plugins executes arbitrary code.
Only use plugins you trust. This API is opt-in and intentionally explicit.
"""

from . import registry  # re-export module for convenience
from .base import ExplainerPlugin, PluginMeta, validate_plugin_meta  # noqa: F401
from .explanations import (  # noqa: F401
    ExplanationBatch,
    ExplanationContext,
    ExplanationPlugin,
    ExplanationRequest,
)
from .intervals import (  # noqa: F401
    ClassificationIntervalCalibrator,
    IntervalCalibratorContext,
    IntervalCalibratorPlugin,
    RegressionIntervalCalibrator,
)
from .plots import (  # noqa: F401
    PlotArtifact,
    PlotBuilder,
    PlotRenderContext,
    PlotRenderResult,
    PlotRenderer,
)
from .predict import PredictBridge  # noqa: F401
from .registry import (  # noqa: F401
    find_for_trusted,
    trust_plugin,
    untrust_plugin,
    clear_explanation_plugins,
    clear_interval_plugins,
    clear_plot_plugins,
)

__all__ = [
    "ExplainerPlugin",
    "ClassificationIntervalCalibrator",
    "ExplanationBatch",
    "ExplanationContext",
    "ExplanationPlugin",
    "ExplanationRequest",
    "IntervalCalibratorContext",
    "IntervalCalibratorPlugin",
    "PlotArtifact",
    "PlotBuilder",
    "PlotRenderContext",
    "PlotRenderResult",
    "PlotRenderer",
    "PluginMeta",
    "PredictBridge",
    "validate_plugin_meta",
    "registry",
    "trust_plugin",
    "untrust_plugin",
    "find_for_trusted",
    "clear_explanation_plugins",
    "clear_interval_plugins",
    "clear_plot_plugins",
]
