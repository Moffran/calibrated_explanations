"""Plugin interfaces, registry utilities, and in-tree plugin registrations.

The package exposes protocol definitions (`base`, `explanations`, `intervals`,
`plots`) plus the shared registry machinery described in ADR-006/013/014/015.
Importing this module eagerly registers the built-in plugins so callers simply
need to `import calibrated_explanations.plugins` before resolving identifiers.
"""

import contextlib

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
from .manager import PluginManager  # noqa: F401
from .plots import (  # noqa: F401
    PlotArtifact,
    PlotBuilder,
    PlotRenderContext,
    PlotRenderer,
    PlotRenderResult,
)
from .predict import PredictBridge  # noqa: F401
from .registry import (  # noqa: F401
    EXPLANATION_PROTOCOL_VERSION,
    _EXPLANATION_PLUGINS,
    _INTERVAL_PLUGINS,
    _PLOT_BUILDERS,
    _PLOT_RENDERERS,
    _PLOT_STYLES,
    clear_explanation_plugins,
    clear_interval_plugins,
    clear_plot_plugins,
    ensure_builtin_plugins,
    find_explanation_descriptor,
    find_explanation_plugin,
    find_explanation_plugin_trusted,
    find_for_trusted,
    find_interval_descriptor,
    find_interval_plugin,
    find_interval_plugin_trusted,
    find_plot_plugin,
    find_plot_plugin_trusted,
    is_identifier_denied,
    list_explanation_descriptors,
    mark_explanation_trusted,
    mark_explanation_untrusted,
    register_explanation_plugin,
    register_interval_plugin,
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
    trust_plugin,
    untrust_plugin,
    unregister,
    validate_explanation_metadata,
)

__all__ = [
    "ExplainerPlugin",
    "ClassificationIntervalCalibrator",
    "EXPLANATION_PROTOCOL_VERSION",
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
    "PluginManager",
    "PredictBridge",
    "_EXPLANATION_PLUGINS",
    "_INTERVAL_PLUGINS",
    "_PLOT_BUILDERS",
    "_PLOT_RENDERERS",
    "_PLOT_STYLES",
    "clear_explanation_plugins",
    "clear_interval_plugins",
    "clear_plot_plugins",
    "ensure_builtin_plugins",
    "find_explanation_descriptor",
    "find_explanation_plugin",
    "find_explanation_plugin_trusted",
    "find_for_trusted",
    "find_interval_descriptor",
    "find_interval_plugin",
    "find_interval_plugin_trusted",
    "find_plot_plugin",
    "find_plot_plugin_trusted",
    "is_identifier_denied",
    "list_explanation_descriptors",
    "mark_explanation_trusted",
    "mark_explanation_untrusted",
    "register_explanation_plugin",
    "register_interval_plugin",
    "register_plot_builder",
    "register_plot_renderer",
    "register_plot_style",
    "registry",
    "trust_plugin",
    "untrust_plugin",
    "unregister",
    "validate_explanation_batch",
    "validate_explanation_metadata",
    "validate_plugin_meta",
]

# Eagerly register in-tree plugins so they are available without explicit import.
# Import at the end to avoid circular import during core initialisation.
with contextlib.suppress(Exception):  # pragma: no cover - import-order guard
    from . import builtins as _builtins  # noqa: F401
