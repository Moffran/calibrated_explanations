"""Plugin interfaces, registry utilities, and in-tree plugin registrations.

The package exposes protocol definitions (`base`, `explanations`, `intervals`,
`plots`) plus the shared registry machinery described in ADR-006/013/014/015.
Importing this module eagerly registers the built-in plugins so callers simply
need to `import calibrated_explanations.plugins` before resolving identifiers.
"""

from . import registry  # re-export module for convenience
from .base import ExplainerPlugin, PluginMeta, validate_plugin_meta  # noqa: F401
from .explanations import (  # noqa: F401
    ExplanationBatch,
    ExplanationContext,
    ExplanationPlugin,
    ExplanationRequest,
    validate_explanation_batch,
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
    PlotRenderer,
    PlotRenderResult,
)
from .predict import PredictBridge  # noqa: F401
from .registry import (  # noqa: F401
    clear_explanation_plugins,
    clear_interval_plugins,
    clear_plot_plugins,
    find_for_trusted,
    trust_plugin,
    untrust_plugin,
)

__all__ = [
    "ExplainerPlugin",
    "ClassificationIntervalCalibrator",
    "ExplanationBatch",
    "ExplanationContext",
    "ExplanationPlugin",
    "ExplanationRequest",
    "validate_explanation_batch",
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

# Eagerly register in-tree plugins so they are available without explicit import.
# Import at the end to avoid circular import during core initialisation.
try:  # pragma: no cover - import-order guard
    from . import builtins as _builtins  # noqa: F401
except Exception:
    # If core is importing plugins while builtins also imports core, avoid
    # raising on partially initialised modules. The builtins can be imported
    # explicitly by consumers after initialisation if needed.
    pass
