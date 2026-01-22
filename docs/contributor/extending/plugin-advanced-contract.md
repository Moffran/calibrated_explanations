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
# Top-level defaults (applies across modes; fast has its own default)
export CE_EXPLANATION_PLUGIN="external.hello.explanation"
export CE_EXPLANATION_PLUGIN_FAST="external.hello.fast"

# Set fallback chain for a specific mode
export CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS="external.hello.explanation,core.explanation.factual"
export CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS="external.hello.alternative"
export CE_EXPLANATION_PLUGIN_FAST_FALLBACKS="external.hello.fast"

# Set fallback chain for interval plugins
export CE_INTERVAL_PLUGIN_FALLBACKS="external.hello.interval,core.interval.legacy"
```

### Example: Explanation plugin fallbacks

```python
import os
from dataclasses import dataclass
from calibrated_explanations.plugins import (
    register_explanation_plugin,
    _EXPLANATION_PLUGINS,
)
from calibrated_explanations.plugins.explanations import ExplanationPlugin


@dataclass
class TestExplanationPlugin(ExplanationPlugin):
    plugin_meta = {
        "schema_version": 1,
        "name": "test.explanation.fallback",
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["explain"],
        "modes": ("factual",),
        "tasks": ("classification",),
        "dependencies": (),
        "trusted": False,
    }

    def supports(self, model):
        return hasattr(model, "predict_proba")

    def initialize(self, context):
        pass

    def explain_batch(self, x, request):
        return None


os.environ["CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS"] = (
    "core.explanation.factual,test.explanation.fallback"
)

plugin = TestExplanationPlugin()
register_explanation_plugin("test.explanation.fallback", plugin)
_EXPLANATION_PLUGINS.pop("test.explanation.fallback", None)
```

### Example: Testing with alternative plot plugin

```python
import os
from calibrated_explanations.plotting import resolve_plot_style_chain
from tests.helpers.model_utils import get_classification_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from calibrated_explanations import CalibratedExplainer

# Prepare data and explainer
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
explainer = CalibratedExplainer(model, x_cal, y_cal)

os.environ["CE_PLOT_STYLE"] = "legacy"
chain = resolve_plot_style_chain(explainer, explicit_style=None)
assert "legacy" in chain
```

### Example: Configure plot fallbacks via environment variables

```python
import os
from calibrated_explanations.plotting import resolve_plot_style_chain
from tests.helpers.model_utils import get_classification_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from calibrated_explanations import CalibratedExplainer

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
explainer = CalibratedExplainer(model, x_cal, y_cal)

os.environ["CE_PLOT_STYLE_FALLBACKS"] = "plot_spec.default,legacy"
chain = resolve_plot_style_chain(explainer, explicit_style=None)
assert len(chain) > 0
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

# Primary interval calibrator plugin (default/fast mode)
[tool.calibrated-explanations.intervals]
default = "external.hello.interval"
fast = "core.interval.fast"

# Fallback chain
default_fallbacks = [
    "core.interval.legacy",
]
fast_fallbacks = ["core.interval.fast", "core.interval.legacy"]

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
5. Default fallback: `"plot_spec.default"` -> `"legacy"`

### Example: Priority demonstration

```python
import os
from calibrated_explanations.plotting import resolve_plot_style_chain
from tests.helpers.model_utils import get_classification_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from calibrated_explanations import CalibratedExplainer

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

os.environ["CE_PLOT_STYLE"] = "env.plot"

explainer = CalibratedExplainer(
    model,
    x_cal,
    y_cal,
    plot_style="plot_spec.default",
)

chain = resolve_plot_style_chain(explainer, explicit_style="plot_spec.default")
assert chain[0] == "plot_spec.default"
```

---

## Method E: Explanation plugin metadata dependencies

Explanation plugins can declare their preferred plot plugins through metadata.
This enables automatic plugin chain seeding without explicit user configuration.

### Declaring plot dependencies

In your explanation plugin metadata, specify the preferred plot plugin:

```python
from dataclasses import dataclass
from calibrated_explanations.plugins.explanations import ExplanationPlugin


@dataclass
class HelloFactualPlugin(ExplanationPlugin):
    """Example explanation plugin with declared dependencies."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.explanation.factual",
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

    def supports(self, model):
        return hasattr(model, "predict_proba")

    def initialize(self, context):
        pass

    def explain_batch(self, x, request):
        return None
```

### How dependencies are propagated

When an explanation plugin is registered with dependencies, the calibrated explainer
automatically seeds these dependencies into the fallback chain:

```python
from calibrated_explanations.plugins import register_explanation_plugin

plugin = HelloFactualPlugin()
register_explanation_plugin("hello.explanation.factual", plugin)

explainer = CalibratedExplainer(
    model,
    x_cal,
    y_cal,
    factual_plugin="hello.explanation.factual",
)

# The explainer internally creates fallback chain:
# ["plot_spec.default", "legacy"]
```

### Example: Self-contained plugin package

A plugin package can declare all its dependencies in metadata:

```python
# my_plugin/explanation.py
from calibrated_explanations.plugins.explanations import ExplanationPlugin

class MyExplanationPlugin(ExplanationPlugin):
    plugin_meta = {
        "name": "my.explanation.advanced",
        "plot_dependency": "my.plot.advanced",  # Use our custom plot plugin
        "interval_dependency": "my.interval.advanced",  # Use our calibrator
        # ... other metadata ...
    }

# my_plugin/plot.py
from calibrated_explanations.plugins.plots import PlotBuilder, PlotRenderer

class MyPlotBuilder(PlotBuilder):
    plugin_meta = {"name": "my.plot.builder", "style": "my.plot.advanced"}

class MyPlotRenderer(PlotRenderer):
    plugin_meta = {"name": "my.plot.renderer"}

# my_plugin/__init__.py
from calibrated_explanations.plugins import (
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

- [ADR-006 – plugin registry](../../improvement/adrs/ADR-006-plugin-registry-trust-model.md)
  – Registry design and plugin lifecycle.
- [ADR-013 – explanation plugin semantics](../../improvement/adrs/ADR-026-explanation-plugin-semantics.md)
  – Explanation plugin contracts and output format.
- [ADR-014 – plot plugin strategy](../../improvement/adrs/ADR-014-plot-plugin-strategy.md)
  – Plot builder/renderer architecture and fallback chains.
- [ADR-024 - legacy plot input contracts](../../improvement/adrs/superseded%20ADR-024-legacy-plot-input-contracts.md)
  – PlotSpec and legacy plot input formats.
- [ADR-025 - legacy plot rendering semantics](../../improvement/adrs/superseded%20ADR-025-legacy-plot-rendering-semantics.md)
  – Plot rendering semantics and compatibility.

## Runtime performance toggles

See {doc}`../../foundations/how-to/tune_runtime_performance` for guidance on
cache and parallel executor configuration when plugins use these optional features.

## Denylist and trust controls

The registry honours both ``CE_TRUST_PLUGIN`` and ``CE_DENY_PLUGIN`` environment
variables.

- ``CE_TRUST_PLUGIN`` lists identifiers that should be trusted on discovery.
- ``[tool.calibrated-explanations.plugins] trusted = ["id"]`` in ``pyproject.toml`` can
  be used for a versioned allowlist of trusted identifiers.
- ``CE_DENY_PLUGIN`` blocks specific identifiers while you iterate on them.

Use the CLI to inspect the resulting state after registration:

```bash
python -m calibrated_explanations.plugins.cli list all
```

See {doc}`../plugin-contract` for additional registry and telemetry guidance.
