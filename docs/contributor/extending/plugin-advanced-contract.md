# Advanced plugin wiring and configuration

This document covers advanced plugin wiring methods beyond basic parameter passing.
For introductory plugin examples and Methods A & B wiring, see {doc}`../plugin-contract`.

## Overview: Plugin wiring methods

There are five primary methods to wire plugins into calibrated explanations:

| Method | Location | Priority | Use Case |
|--------|----------|----------|----------|
| **A** | CalibratedExplainer parameter | 1 (highest) | Development; consistent plugin selection across explainer lifetime |
| **B** | Explanation.plot() parameter | 1 (highest) | Dynamic comparison; different visualizations for same explanations |
| **C** | Environment variables | 2 | CI/CD; operator configuration without code changes |
| **D** | pyproject.toml | 3 | Project-level defaults; distribution with package |
| **E** | Explanation plugin metadata | 4 | Plugin-declared preferences; automatic dependency seeding |

---

## Method C: Environment variable configuration

Environment variables allow runtime plugin selection without code modifications,
making them ideal for CI/CD pipelines and operational settings.

### Configuration variables

For **plot plugins**, use:

```bash
# Primary plot style selector
export CE_PLOT_STYLE="my.custom.plot"

# Comma-separated fallback chain (optional)
export CE_PLOT_STYLE_FALLBACKS="fallback.plot,legacy"
```

For **explanation plugins**, use:

```bash
# Set fallback chain for a specific mode
export CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS="external.hello.explanation,core.explanation.factual"
export CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS="external.hello.alternative"
export CE_EXPLANATION_PLUGIN_FAST_FALLBACKS="external.hello.fast"

# Set fallback chain for interval plugins
export CE_INTERVAL_PLUGIN_FALLBACKS="external.hello.interval,core.interval.legacy"
```

### Example: Testing with alternative plot plugin

```python
import os
os.environ["CE_PLOT_STYLE"] = "legacy"

from calibrated_explanations import CalibratedExplainer

# The explainer automatically uses the environment-configured plot style
explainer = CalibratedExplainer(model, x_cal, y_cal)
explanations = explainer.explain_factual(x_test)

# Plot style is driven by CE_PLOT_STYLE
explanations.plot()
```

### Example: CI/CD workflow with plugin validation

```bash
#!/bin/bash
# test-plugins.sh - Validate plugins in different configurations

# Test with legacy plots
export CE_PLOT_STYLE="legacy"
pytest tests/ -v

# Test with plotspec plots
export CE_PLOT_STYLE="plot_spec.default"
pytest tests/ -v

# Test with custom plugin
export CE_PLOT_STYLE="my.custom.plot"
export CE_PLOT_STYLE_FALLBACKS="legacy"
pytest tests/ -v
```

---

## Method D: pyproject.toml configuration

Project-level defaults can be specified in `pyproject.toml` for consistent
behavior across development and distribution.

### Configuration structure

```toml
[tool.calibrated-explanations.plots]
# Primary plot style selector
style = "my.custom.plot"

# Fallback chain when primary plugin fails
style_fallbacks = [
    "fallback.plot",
    "plot_spec.default",
    "legacy",
]

[tool.calibrated-explanations.intervals]
# Primary interval calibrator plugin
plugin = "external.hello.interval"

# Fallback chain
fallbacks = [
    "core.interval.legacy",
]

[tool.calibrated-explanations.explanations]
# Per-mode plugin configuration
factual = "external.hello.explanation"
alternative = "core.explanation.alternative"
fast = "core.explanation.fast"
```

### Example: Distribution with custom plot plugin

When your package includes a plot plugin, document project defaults in `pyproject.toml`:

```toml
[project]
name = "my-calibrated-plots"
dependencies = ["calibrated-explanations>=0.9.0"]

[tool.calibrated-explanations.plots]
style = "my.beautiful.plot"
style_fallbacks = ["plot_spec.default", "legacy"]
```

Users of your package can override these defaults using environment variables
(Method C) or explainer parameters (Method A).

### Priority order for plot style selection

When multiple configuration methods are active, this priority order applies:

1. Explainer parameter: `CalibratedExplainer(..., plot_style="explicit")`
2. Environment variable: `CE_PLOT_STYLE="from_env"`
3. pyproject.toml: `[tool.calibrated-explanations.plots] style = "from_project"`
4. Explanation plugin metadata: `"plot_dependency": "from_plugin"`
5. Default fallback: `"plot_spec.default"` → `"legacy"`

### Example: Priority demonstration

```python
import os

# Set environment variable
os.environ["CE_PLOT_STYLE"] = "env.plot"

# Create explainer with explicit parameter (highest priority)
explainer = CalibratedExplainer(
    model, x_cal, y_cal,
    plot_style="explicit.plot"  # ← This takes precedence
)

explanations = explainer.explain_factual(x_test)
# Plot style used: "explicit.plot" (from parameter, not env var)
explanations.plot()
```

---

## Method E: Explanation plugin metadata dependencies

Explanation plugins can declare their preferred plot plugins through metadata.
This enables automatic plugin chain seeding without explicit user configuration.

### Declaring plot dependencies

In your explanation plugin metadata, specify the preferred plot plugin:

```python
class MyExplanationPlugin(ExplanationPlugin):
    plugin_meta = {
        "schema_version": 1,
        "name": "my.explanation",
        "version": "0.1.0",
        "provider": "my-team",
        "modes": ("factual",),
        "tasks": ("classification",),
        "capabilities": ["explain", "explanation:factual", "task:classification"],
        "dependencies": (),
        # ↓ Declare preferred plot plugin
        "plot_dependency": "my.plot.style",
        "interval_dependency": "my.interval.calibrator",
        "trusted": False,
    }
```

### How dependencies are propagated

When an explanation plugin is registered with dependencies, the calibrated explainer
automatically seeds these dependencies into the fallback chain:

```python
from calibrated_explanations.plugins.registry import register_explanation_plugin

plugin = MyExplanationPlugin()
register_explanation_plugin("my.explanation", plugin)

# Later, when using the plugin:
explainer = CalibratedExplainer(
    model, x_cal, y_cal,
    factual_plugin="my.explanation",
    # plot_style not specified; uses dependency from plugin metadata
)

# The explainer internally creates fallback chain:
# ["my.plot.style", "plot_spec.default", "legacy"]
```

### Example: Self-contained plugin package

A plugin package can declare all its dependencies in metadata:

```python
# my_plugin/explanation.py
from calibrated_explanations.plugins.explanations import ExplanationPlugin

class MyExplanationPlugin(ExplanationPlugin):
    plugin_meta = {
        "name": "my.explanation.advanced",
        "plot_dependency": "my.plot.advanced",  # ← Use our custom plot plugin
        "interval_dependency": "my.interval.advanced",  # ← Use our calibrator
        # ... other metadata ...
    }

# my_plugin/plot.py
from calibrated_explanations.plugins.plots import PlotBuilder, PlotRenderer

class MyPlotBuilder(PlotBuilder):
    plugin_meta = {"name": "my.plot.builder", "style": "my.plot.advanced"}

class MyPlotRenderer(PlotRenderer):
    plugin_meta = {"name": "my.plot.renderer"}

# my_plugin/__init__.py
from calibrated_explanations.plugins.registry import (
    register_explanation_plugin,
    register_plot_builder,
    register_plot_renderer,
    register_plot_style,
)

register_explanation_plugin("my.explanation.advanced", MyExplanationPlugin())
register_plot_builder("my.plot.builder", MyPlotBuilder())
register_plot_renderer("my.plot.renderer", MyPlotRenderer())
register_plot_style(
    "my.plot.advanced",
    metadata={
        "style": "my.plot.advanced",
        "builder_id": "my.plot.builder",
        "renderer_id": "my.plot.renderer",
        "fallbacks": ("plot_spec.default", "legacy"),
    },
)
```

Users of this package can simply import it, and all dependencies are automatically wired:

```python
import my_plugin  # Triggers registration and dependency seeding

explainer = CalibratedExplainer(
    model, x_cal, y_cal,
    factual_plugin="my.explanation.advanced",
    # Automatically uses my.plot.advanced and my.interval.advanced
)
```

---

## Guardrails and ADR references

Refer to these decision records when designing advanced plugin configurations:

- [ADR-006 – plugin registry](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-006-plugin-registry-trust-model.md)
  – Registry design and plugin lifecycle.
- [ADR-013 – explanation plugin semantics](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-026-explanation-plugin-semantics.md)
  – Explanation plugin contracts and output format.
- [ADR-014 – plot plugin strategy](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-014-plot-plugin-strategy.md)
  – Plot builder/renderer architecture and fallback chains.
- [ADR-024 – legacy plot input contracts](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-024-legacy-plot-input-contracts.md)
  – PlotSpec and legacy plot input formats.
- [ADR-025 – legacy plot rendering semantics](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-025-legacy-plot-rendering-semantics.md)
  – Plot rendering semantics and compatibility.

## Runtime performance toggles

See {doc}`../../foundations/how-to/tune_runtime_performance` for guidance on
cache and parallel executor configuration when plugins use these optional features.

## Denylist and trust controls

The registry honours both ``CE_TRUST_PLUGIN`` and ``CE_DENY_PLUGIN`` environment
variables.

- ``CE_TRUST_PLUGIN`` lists identifiers that should be trusted on discovery.
- ``CE_DENY_PLUGIN`` blocks specific identifiers while you iterate on them.

Use the CLI to inspect the resulting state after registration:

```bash
python -m calibrated_explanations.plugins.cli list all
```

See {doc}`../plugin-contract` for additional registry and telemetry guidance.
