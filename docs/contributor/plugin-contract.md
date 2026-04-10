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

## Plugin override precedence

All plugin types follow a consistent override hierarchy for selection and configuration:

1. **Explainer parameters** (highest priority): Direct kwargs to `CalibratedExplainer` (e.g., `factual_plugin`, `interval_plugin`, `plot_style`).
2. **Environment variables**: Mode-specific variables (e.g., `CE_EXPLANATION_PLUGIN_FACTUAL`, `CE_INTERVAL_PLUGIN`, `CE_PLOT_STYLE`).
3. **pyproject.toml**: Project-level defaults under `[tool.calibrated_explanations.plugins]` or mode sections.
4. **Plugin-declared dependencies** (lowest): Automatic seeding from plugin metadata (e.g., `interval_dependency`, `plot_dependency`).

For detailed wiring examples, see {doc}`./extending/plugin-advanced-contract`.

## Modality contract

ADR-033 defines the modality metadata required for entry-point plugins and the
resolver behavior used by modality-aware selection helpers.

### Canonical modalities and aliases

`data_modalities` values are normalised to canonical names:

- `tabular`
- `vision` (aliases: `image`, `images`, `img`)
- `audio`
- `text`
- `multimodal` (aliases: `multi-modal`, `multi_modal`)

Custom extension names must use an `x-<vendor>-<name>` namespace.

### Metadata requirements

Entry-point explanation plugins should publish these metadata fields:

- `data_modalities`: tuple of canonical modality names
- `plugin_api_version`: `MAJOR.MINOR` or `MAJOR.MINOR.PATCH`

Compatibility policy is major-hard/minor-soft:

- major mismatch is rejected at discovery time
- higher minor/patch can load with `UserWarning` plus governance logging

### v0.11.1 deprecation timeline

- `v0.11.0`: missing `data_modalities` defaults to `("tabular",)` silently
- `v0.11.1`: missing `data_modalities` on entry-point metadata emits
    `DeprecationWarning`
- `v0.12.0/v1.0.0-rc`: explicit `data_modalities` is required; fallback default
    is removed

See {doc}`../practitioner/advanced/modality-plugins` for user-facing selection
examples and migration guidance.

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

### Override precedence for interval calibrator plugins

Interval plugins are selected globally or per-mode:

- **Explainer parameters**: `interval_plugin`, `fast_interval_plugin`
- **Environment variables**: `CE_INTERVAL_PLUGIN`, `CE_INTERVAL_PLUGIN_FAST`
- **pyproject.toml**: Under `[tool.calibrated_explanations.intervals]` with keys `default`, `fast`
- **Dependencies**: Seeded from explanation plugin metadata `interval_dependency`

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

### Pitfall: two prediction paths — do not mix them

Inside `explain_batch` there are **two separate prediction paths** with different
responsibilities. Confusing them produces a `TypeError` that surfaces as an opaque
`ENGINE_FAILURE` to the caller.

| Path | How to call | What it does |
|---|---|---|
| `bridge.predict(x, mode=..., task=..., bins=...)` | Via `context.predict_bridge` | Lifecycle contract check; uses **default percentiles (5, 95)** internally; return value is often discarded |
| `explainer.predict(x, uq_interval=True, low_high_percentiles=..., bins=...)` | Via `context.helper_handles["explainer"]` | Full inference; honours any custom `low_high_percentiles` from the `ExplanationRequest` |

**The `PredictBridge` protocol does not accept `low_high_percentiles`.** If a
caller forwards `request.low_high_percentiles` into `bridge.predict()` it gets a
`TypeError` on every call where that field is non-`None`. The fix is to send
interval-shaping parameters to the explainer handle instead:

```python
def explain_batch(self, x, request):
    # Step 1 — honour the bridge lifecycle contract (protocol-defined params only)
    self._bridge.predict(x, mode=self._mode, task=self.context.task, bins=request.bins)

    # Step 2 — full inference with custom percentiles goes to the explainer handle
    explainer = self.context.helper_handles["explainer"]
    prediction, (low, high) = explainer.predict(
        x,
        uq_interval=True,
        low_high_percentiles=request.low_high_percentiles,  # safe here
        bins=request.bins,
    )
    ...
```

This distinction is not obvious because the bridge's `predict()` return value
contains `low` and `high` arrays, which makes it look like the interval-shaping
surface. It is not — it is a narrowly-scoped lifecycle shim.

### Override precedence for explanation plugins

Explanation plugins support mode-specific selection:

- **Explainer parameters**: `factual_plugin`, `alternative_plugin`, `fast_plugin`
- **Environment variables**: `CE_EXPLANATION_PLUGIN_FACTUAL`, `CE_EXPLANATION_PLUGIN_ALTERNATIVE`, `CE_EXPLANATION_PLUGIN_FAST`
- **pyproject.toml**: Under `[tool.calibrated_explanations.explanations]` with keys `factual`, `alternative`, `fast`
- **Dependencies**: Plugin metadata `interval_dependency` and `plot_dependency` seed related plugins automatically

---

## Hello, plot plugin

Follow these steps to build a plot plugin (builder and renderer pair) that respects the
contract defined by ADR-037 and ADR-036.

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

### Override precedence for plot plugins

Plot styles control visualization rendering:

- **Explainer parameters**: `plot_style`
- **Explanation.plot() parameters**: `style_override` for dynamic selection
- **Environment variables**: `CE_PLOT_STYLE`, `CE_PLOT_STYLE_FALLBACKS`
- **pyproject.toml**: Under `[tool.calibrated_explanations.plots]` with key `style`
- **Dependencies**: Seeded from explanation plugin metadata `plot_dependency`

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

- [ADR-006 - plugin registry trust model](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-006-plugin-registry-trust-model.md)
  defines explicit trust controls (`CE_TRUST_PLUGIN`, `CE_DENY_PLUGIN`) and governance expectations for third-party plugins.
- [ADR-013 - interval calibrator plugin strategy](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-013-interval-calibrator-plugin-strategy.md)
  defines the architecture for interval calibrator plugins and their integration with core calibrators.
- [ADR-037 - visualization extension and rendering governance](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-037-visualization-extension-and-rendering-governance.md)
  defines builder/renderer governance, deterministic metadata requirements, and default behavior.
- [ADR-015 - explanation plugin architecture](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-015-explanation-plugin.md)
  specifies the plugin orchestration, resolution, and mode-aware selection for explanation plugins.
- [ADR-036 - PlotSpec canonical contract and validation boundary](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-036-plot-spec-canonical-contract-and-validation-boundary.md)
  documents canonical PlotSpec semantics, validation boundaries, and compatibility rules.
- [ADR-026 – explanation plugin semantics](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-026-explanation-plugin-semantics.md)
  captures the calibrated explanation contract for explanation and interval
  plugins.

### Trust and governance checklist

Before shipping a plugin, verify all items:

- Document trust onboarding (how operators should set `CE_TRUST_PLUGIN` or pyproject allowlists).
- Document emergency deny flow (`CE_DENY_PLUGIN`) for incident mitigation.
- Verify plugin appears in discovery diagnostics (`python -m calibrated_explanations.plugins.cli report`).
- Verify trusted-only views behave as expected (`python -m calibrated_explanations.plugins.cli list all --trusted-only`).

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
