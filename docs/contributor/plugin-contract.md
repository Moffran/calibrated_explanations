# Plugin contract

Calibrated explanations stay extensible without sacrificing the calibrated-first
workflow. Every plugin must honour the contract below so binary and multiclass
classification plus probabilistic and interval regression remain aligned with
the core library.

## Overview: Plugin types

This document covers three complementary plugin types used throughout the calibrated
explanations pipeline:

1. **Interval calibrator plugins** – Produce prediction intervals for uncertainty quantification
2. **Explanation plugins** – Generate factual, alternative, or fast explanations from calibrated models
3. **Plot plugins** – Render explanation visualizations in different styles and formats

## Hello, interval calibrator plugin

Follow these steps to build an interval calibrator plugin that respects the contract defined by
ADR-026.

### 1. Scaffold the interval calibrator class

Interval calibrators implement the :class:`calibrated_explanations.plugins.intervals.IntervalCalibratorPlugin`
protocol. The example below shows a minimal implementation that computes simple bounds; replace the
body with your calibration logic while preserving calibrated prediction intervals.

```python
from __future__ import annotations

from typing import Any, Mapping

from calibrated_explanations.plugins.intervals import (
    IntervalCalibratorPlugin,
    IntervalCalibratorContext,
    ClassificationIntervalCalibrator,
    RegressionIntervalCalibrator,
)


class HelloIntervalCalibratorPlugin(IntervalCalibratorPlugin):
    """Minimal interval calibrator example that computes fixed-width bounds."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.interval.calibrator",
        "version": "0.1.0",
        "provider": "example-team",
        "capabilities": ["interval:classification", "interval:regression"],
        "modes": ("classification", "regression"),
        "dependencies": (),
        "trusted": False,
        "trust": False,
        "confidence_source": "hello",
        "requires_bins": False,
        "fast_compatible": False,
    }

    def create(self, context: IntervalCalibratorContext, **kwargs: Any) -> Any:
        """Return a list of calibrators for feature-wise and model-level intervals."""
        task = str(context.metadata.get("task") or context.metadata.get("mode") or "")

        # For demonstration, return mock calibrators
        # In practice, fit calibrators on context.calibration_splits
        if "classification" in task:
            return [ClassificationIntervalCalibrator() for _ in range(
                int(context.metadata.get("num_features", 1))
            )] + [ClassificationIntervalCalibrator()]  # +1 for model-level
        else:
            return [RegressionIntervalCalibrator() for _ in range(
                int(context.metadata.get("num_features", 1))
            )] + [RegressionIntervalCalibrator()]
```

**Note:** `IntervalCalibratorContext` is frozen by design. `context.metadata` is exposed as an immutable mapping so plugins cannot mutate shared explainer state, while `context.plugin_state` provides a mutable dictionary for storing transient, per-execution scratch data (e.g., caching intermediate summaries or heuristics) without leaking back into the explainer.

### 2. Validate and register the interval calibrator

```python
from calibrated_explanations.plugins.registry import register_interval_plugin
from calibrated_explanations.plugins.base import validate_plugin_meta

plugin = HelloIntervalCalibratorPlugin()
validate_plugin_meta(dict(plugin.plugin_meta))
register_interval_plugin("external.hello.interval", plugin)
```

---

## Hello, explanation plugin

Follow these steps to build an explanation plugin that respects the contract defined by
ADR-026.

### 1. Scaffold the explanation plugin class

Explanation plugins implement the :class:`calibrated_explanations.plugins.explanations.ExplanationPlugin`
protocol. The example below emits a fixed explanation payload;
replace the body with your logic while preserving calibrated outputs.

```python
from __future__ import annotations

from typing import Any, Mapping
from dataclasses import dataclass

from calibrated_explanations.plugins.explanations import (
    ExplanationPlugin,
    ExplanationContext,
    ExplanationRequest,
    ExplanationBatch,
)
from calibrated_explanations.explanations import CalibratedExplanations


@dataclass
class HelloExplanationPlugin(ExplanationPlugin):
    """Minimal example that returns calibrated explanations for binary classification."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.explanation.plugin",
        "version": "0.1.0",
        "provider": "example-team",
        "capabilities": ["explain", "explanation:factual", "task:classification"],
        "modes": ("factual",),
        "tasks": ("classification",),
        "dependencies": (),
        "interval_dependency": "core.interval.legacy",
        "plot_dependency": "plot_spec.default",
        "trusted": False,
    }

    def supports(self, model: Any) -> bool:
        """Return whether the plugin can work with *model*."""
        return hasattr(model, "predict_proba")

    def initialize(self, context: ExplanationContext) -> None:
        """Initialize the plugin with explainer context."""
        self.context = context

    def explain_batch(self, x: Any, request: ExplanationRequest) -> ExplanationBatch:
        """Produce a batch of calibrated explanations."""
        # Generate explanations using your logic
        # Store results in ExplanationBatch with collection_metadata
        explanations = CalibratedExplanations(None, 0, x, {}, {}, {}, {})
        return ExplanationBatch(
            explanations=explanations,
            collection_metadata={
                "mode": self.plugin_meta.get("modes", ("factual",))[0],
                "task": self.context.task,
            },
        )
```

### 2. Validate and register the explanation plugin

```python
from calibrated_explanations.plugins.registry import register_explanation_plugin
from calibrated_explanations.plugins.base import validate_plugin_meta

plugin = HelloExplanationPlugin()
validate_plugin_meta(dict(plugin.plugin_meta))
register_explanation_plugin("external.hello.explanation", plugin)
```

---

## Hello, plot plugin

Follow these steps to build a plot plugin (builder and renderer pair) that respects the
contract defined by ADR-024/025.

### 1. Scaffold the plot builder and renderer

Plot builders and renderers implement the :class:`calibrated_explanations.plugins.plots.PlotBuilder`
and :class:`calibrated_explanations.plugins.plots.PlotRenderer` protocols.

```python
from __future__ import annotations

from typing import Any, Mapping
from calibrated_explanations.plugins.plots import (
    PlotBuilder,
    PlotRenderer,
    PlotRenderContext,
    PlotRenderResult,
)


class HelloPlotBuilder(PlotBuilder):
    """Minimal plot builder that creates a simple PlotSpec payload."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.plot.builder",
        "version": "0.1.0",
        "provider": "example-team",
        "style": "hello",
        "output_formats": ("png", "svg"),
        "capabilities": ["plot:builder"],
        "dependencies": (),
        "trusted": False,
        "legacy_compatible": False,
    }

    def build(self, context: PlotRenderContext) -> Mapping[str, Any]:
        """Return a PlotSpec-compatible payload for the explanation."""
        # Build a visualization specification from the explanation context
        # This example returns a minimal structure; implement your visualization logic
        return {
            "plot_spec": {
                "title": "Hello Plot",
                "description": "A minimal plot builder example",
                "primitives": [],
            }
        }


class HelloPlotRenderer(PlotRenderer):
    """Minimal plot renderer that materializes PlotSpec payloads."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.plot.renderer",
        "version": "0.1.0",
        "provider": "example-team",
        "output_formats": ("png", "svg"),
        "capabilities": ["plot:renderer"],
        "supports_interactive": False,
        "dependencies": (),
        "trusted": False,
        "legacy_compatible": False,
    }

    def render(
        self, artifact: Mapping[str, Any], *, context: PlotRenderContext
    ) -> PlotRenderResult:
        """Render the PlotSpec artifact and return visualization result."""
        # Materialize the plot using matplotlib, plotly, or other rendering backend
        # This example returns a minimal result; implement your rendering logic
        return PlotRenderResult(
            artifact=artifact,
            figure=None,
            saved_paths=(),
            extras={},
        )
```

### 2. Register the plot builder, renderer, and style

```python
from calibrated_explanations.plugins.registry import (
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
)
from calibrated_explanations.plugins.base import validate_plugin_meta

builder = HelloPlotBuilder()
renderer = HelloPlotRenderer()

validate_plugin_meta(dict(builder.plugin_meta))
validate_plugin_meta(dict(renderer.plugin_meta))

register_plot_builder("hello.plot.builder", builder)
register_plot_renderer("hello.plot.renderer", renderer)

# Map the style identifier to builder and renderer
register_plot_style(
    "hello",
    metadata={
        "style": "hello",
        "builder_id": "hello.plot.builder",
        "renderer_id": "hello.plot.renderer",
        "fallbacks": ("plot_spec.default", "legacy"),  # Fallback chain
        "is_default": False,
    },
)
```

---

## Wiring plugins into your explainer and explanations

Once a plugin is registered, you can wire it into your workflow using two primary methods:

### Method A: Via CalibratedExplainer parameter

Pass the desired plugin identifier (or style for plot plugins) to the explainer at construction time:

```python
from calibrated_explanations import CalibratedExplainer

# Wire interval and explanation plugins
explainer = CalibratedExplainer(
    model, x_cal, y_cal,
    factual_plugin="external.hello.explanation",  # Explanation plugin
    default_interval_plugin="external.hello.interval",  # Interval plugin
    plot_style="hello",  # Plot style (builder + renderer pair)
)
```

This approach ensures consistent plugin selection across all explanations generated by the explainer.
The explainer stores the plugin identifiers and uses them during explanation generation.

### Method B: Via explanation.plot() parameter

Pass the plot style override when calling `.plot()` on explanation batches:

```python
explanations = explainer.explain_factual(x_test)

# Use the registered plot style
explanations.plot(style_override="hello")

# Or fall back to defaults (plot_spec.default -> legacy)
explanations.plot()
```

This method allows dynamic plot style selection after explanations are generated, enabling
comparison of different visualization styles for the same explanations.

**Note:** The `style` parameter in `.plot()` controls plot **type** (regular, triangular),
not the plugin style. Use `style_override` to select plot plugins.

---

## Advanced wiring methods

For environment variable configuration, pyproject.toml settings, and plugin dependency propagation,
see {doc}`./extending/plugin-advanced-contract`.

---

## Packaging and entry points

To distribute your plugins with pip, expose entry points under ``calibrated_explanations.plugins``:

```toml
[project.entry-points."calibrated_explanations.plugins"]
external.hello.explanation = "your_package.plugins:HelloExplanationPlugin"
external.hello.interval = "your_package.plugins:HelloIntervalCalibratorPlugin"

[project.entry-points."calibrated_explanations.plugins.plot_builders"]
hello.plot.builder = "your_package.plugins:HelloPlotBuilder"

[project.entry-points."calibrated_explanations.plugins.plot_renderers"]
hello.plot.renderer = "your_package.plugins:HelloPlotRenderer"
```

When your package is installed, these entry points are discovered automatically.
Users can then register your plugins by importing your module:

```python
import your_package.plugins  # Triggers registration via entry points
# or register manually:
from your_package.plugins import HelloExplanationPlugin
from calibrated_explanations.plugins.registry import register_explanation_plugin
register_explanation_plugin("external.hello.explanation", HelloExplanationPlugin())
```

Document how probabilistic and interval regression stay calibrated after your
extension by linking back to the practitioner quickstarts and interpretation
guides.

## Guardrails and ADR references

Use these decision records when designing new plugins:

- [ADR-024 - legacy plot input contracts](../improvement/adrs/superseded%20ADR-024-legacy-plot-input-contracts.md)
  defines PlotSpec and legacy plot inputs that plugins must honour.
- [ADR-025 - legacy plot rendering semantics](../improvement/adrs/superseded%20ADR-025-legacy-plot-rendering-semantics.md)
  documents rendering semantics so PlotSpec builders remain interchangeable.
- [ADR-026 – explanation plugin semantics](../improvement/adrs/ADR-026-explanation-plugin-semantics.md)
  captures the calibrated explanation contract for explanation and interval
  plugins.

### Runtime performance toggles

v0.9.0 introduces an opt-in calibrator cache and parallel executor. Plugin
authors should treat these as shared infrastructure:

- Avoid mutating request-level objects captured by the cache key. Values stored
  in the cache must be deterministic, pickle-safe, and process independent.
- When you spawn subprocesses (for example inside a plugin), call
  ``explainer._perf_cache.forksafe_reset()`` or reuse the provided
  :class:`~calibrated_explanations.parallel.ParallelExecutor` to inherit the
  built-in fork guards. The ``calibrated_explanations.perf.parallel`` and
  ``calibrated_explanations.perf.cache`` shims remain temporarily for
  compatibility and will be removed after v1.1.0.
- Respect ``CE_CACHE``/``CE_PARALLEL`` overrides documented in
  {doc}`../foundations/how-to/tune_runtime_performance`. Plugins should not force
  their own worker settings—delegate to the executor attached to the explainer
  instead.
- Emit telemetry sparingly and honour the ``perf_telemetry`` callback when
  provided so operators can aggregate cache hit/miss data consistently.

### Fast explanations and external bundles

The fast explanations implementation now ships as an external plugin. Install
the aggregated extra and register the bundle explicitly when you need it:

```bash
pip install "calibrated-explanations[external-plugins]"
python -m external_plugins.fast_explanations register
```

See {doc}`../appendices/external_plugins` for community listings and governance
notes.

## Denylist and trust controls

The registry honours both ``CE_TRUST_PLUGIN`` and ``CE_DENY_PLUGIN`` environment
variables.

- ``CE_TRUST_PLUGIN`` lists identifiers that should be trusted on discovery.
- ``CE_DENY_PLUGIN`` blocks specific identifiers while you iterate on them.

Use the CLI to inspect the resulting state after registration:

```bash
python -m calibrated_explanations.plugins.cli list all
```

The CLI always prints an opt-in reminder so core calibrated explanations remain
usable without it.

### CLI discovery (optional)

The :mod:`calibrated_explanations.plugins.cli` module surfaces plugin metadata
for auditing. It is optional tooling—set ``CE_DENY_PLUGIN`` during local testing
to simulate trust boundaries before promoting a plugin.

### Telemetry instrumentation (optional)

If your plugin emits telemetry, document the signals and link back to
{doc}`../foundations/governance/optional_telemetry`. Keep emission disabled by
default so calibrated explanations remain privacy-preserving out of the box.
