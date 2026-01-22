# Plugin contract

Calibrated explanations stay extensible without sacrificing the calibrated-first
workflow. Every plugin must honour the contract below so binary and multiclass
classification plus probabilistic and interval regression remain aligned with
the core library.

## Overview: Plugin types

This document covers three complementary plugin types used throughout the calibrated
explanations pipeline:

1. **Interval calibrator plugins** - Produce prediction intervals for uncertainty quantification
2. **Explanation plugins** - Generate factual, alternative, or fast explanations from calibrated models
3. **Plot plugins** - Render explanation visualizations in different styles and formats

## Hello, calibrated plugin (minimal explanation plugin)

Use this minimal example when you want to wrap a model that already exposes
`predict_proba` and return a calibrated payload directly.

```python
from __future__ import annotations

import numpy as np


def build_dummy_model():
    class DummyModel:
        def predict_proba(self, x):
            return np.column_stack([1 - x, x])

    return DummyModel()


class HelloCalibratedPlugin:
    """Minimal example that returns calibrated outputs for a wrapped model."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.calibrated.plugin",
        "version": "0.1.0",
        "provider": "example-team",
        "capabilities": ("binary-classification", "probabilistic-regression"),
        "dependencies": (),
        "modes": ("factual", "alternative"),
        "tasks": ("classification", "regression"),
        "trusted": False,
    }

    def supports(self, model):
        """Return whether the plugin can work with *model*."""

        return hasattr(model, "predict_proba")

    def explain(self, model, x, **kwargs):
        """Produce a calibrated explanation payload for ``x``."""

        probabilities = model.predict_proba(x)
        return {
            "prediction": probabilities[:, 1],
            "uncertainty_interval": (probabilities[:, 0], probabilities[:, 1]),
            "modes": self.plugin_meta["modes"],
        }
```

## Hello, interval calibrator plugin

Follow these steps to build an interval calibrator plugin that respects the contract defined by
ADR-026.

### 1. Scaffold the interval calibrator class

Interval calibrators implement the :class:`calibrated_explanations.plugins.intervals.IntervalCalibratorPlugin`
protocol. The example below shows a minimal implementation that computes simple bounds; replace the
body with your calibration logic while preserving calibrated prediction intervals.

```python
from __future__ import annotations

from typing import Any

from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin


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

    def create(self, context, **kwargs: Any) -> Any:
        """Return a list of calibrators for feature-wise and model-level intervals."""
        task = str(context.metadata.get("task") or context.metadata.get("mode") or "")
        if "classification" in task:
            return []
        else:
            return []
```

**Note:** `IntervalCalibratorContext` is frozen by design. `context.metadata` is exposed as an immutable mapping so plugins cannot mutate shared explainer state, while `context.plugin_state` provides a mutable dictionary for storing transient, per-execution scratch data (e.g., caching intermediate summaries or heuristics) without leaking back into the explainer.

### 2. Validate and register the interval calibrator

```python
from calibrated_explanations.plugins.base import validate_plugin_meta
from calibrated_explanations.plugins import register_interval_plugin

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

from dataclasses import dataclass
from typing import Any

from calibrated_explanations.plugins.explanations import ExplanationPlugin
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
        return hasattr(model, "predict_proba")

    def initialize(self, context):
        self.context = context

    def explain_batch(self, x, request):
        explanations = CalibratedExplanations(None, 0, x, {}, {}, {}, {})
        return type(
            "ExplanationBatch",
            (),
            {
                "explanations": explanations,
                "collection_metadata": {
                    "mode": self.plugin_meta.get("modes", ("factual",))[0],
                    "task": self.context.task,
                },
            },
        )()
```

### 2. Validate and register the explanation plugin

```python
from calibrated_explanations.plugins.base import validate_plugin_meta
from calibrated_explanations.plugins import register_explanation_plugin

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

from calibrated_explanations.plugins.plots import (
    PlotBuilder,
    PlotRenderer,
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

    def build(self, context):
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

    def render(self, artifact, *, context):
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
from calibrated_explanations.plugins.base import validate_plugin_meta
from calibrated_explanations.plugins.plots import (
    PlotBuilder,
    PlotRenderer,
    PlotRenderResult,
)
from calibrated_explanations.plugins import (
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
)


class HelloPlotBuilder(PlotBuilder):
    plugin_meta = {
        "schema_version": 1,
        "name": "hello.plot.builder.test",
        "version": "0.1.0",
        "provider": "example-team",
        "style": "hello.test",
        "output_formats": ("png", "svg"),
        "capabilities": ["plot:builder"],
        "dependencies": (),
        "trusted": False,
        "legacy_compatible": False,
    }

    def build(self, context):
        return {"plot_spec": {"title": "Test", "primitives": []}}


class HelloPlotRenderer(PlotRenderer):
    plugin_meta = {
        "schema_version": 1,
        "name": "hello.plot.renderer.test",
        "version": "0.1.0",
        "provider": "example-team",
        "output_formats": ("png", "svg"),
        "capabilities": ["plot:renderer"],
        "supports_interactive": False,
        "dependencies": (),
        "trusted": False,
        "legacy_compatible": False,
    }

    def render(self, artifact, *, context):
        return PlotRenderResult(
            artifact=artifact,
            figure=None,
            saved_paths=(),
            extras={},
        )


builder = HelloPlotBuilder()
renderer = HelloPlotRenderer()

validate_plugin_meta(dict(builder.plugin_meta))
validate_plugin_meta(dict(renderer.plugin_meta))

builder_id = "hello.plot.builder.test"
renderer_id = "hello.plot.renderer.test"
style_id = "hello.test"

register_plot_builder(builder_id, builder)
register_plot_renderer(renderer_id, renderer)

register_plot_style(
    style_id,
    metadata={
        "style": style_id,
        "builder_id": builder_id,
        "renderer_id": renderer_id,
        "fallbacks": ("plot_spec.default", "legacy"),
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
from tests.helpers.model_utils import get_classification_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from calibrated_explanations import CalibratedExplainer

# Prepare data
data = load_breast_cancer()
x = data.data
y = data.target
x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_cal, y_train, y_cal = train_test_split(
    x_temp, y_temp, test_size=0.4, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_cal = scaler.transform(x_cal)
x_test = scaler.transform(x_test)

model, _ = get_classification_model("RF", x_train, y_train)

# Parameter wiring
explainer = CalibratedExplainer(
    model,
    x_cal,
    y_cal,
    plot_style="plot_spec.default",
)

chain = explainer.plugin_manager.plot_style_chain
assert chain[0] == "plot_spec.default"
```

This approach ensures consistent plugin selection across all explanations generated by the explainer.
The explainer stores the plugin identifiers and uses them during explanation generation.

### Method B: Via explanation.plot() parameter

Pass the plot style override when calling `.plot()` on explanation batches:

```python
from tests.helpers.model_utils import get_classification_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from calibrated_explanations import CalibratedExplainer

# Prepare data
data = load_breast_cancer()
x = data.data
y = data.target
x_temp, x_test, y_temp, _ = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_cal, y_train, y_cal = train_test_split(
    x_temp, y_temp, test_size=0.4, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_cal = scaler.transform(x_cal)
x_test = scaler.transform(x_test)

model, _ = get_classification_model("RF", x_train, y_train)

explainer = CalibratedExplainer(
    model,
    x_cal,
    y_cal,
)

explanations = explainer.explain_factual(x_test[:3])

assert hasattr(explanations, "plot")
assert "style_override" in explanations.plot.__code__.co_varnames
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
from calibrated_explanations.plugins import register_explanation_plugin
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
