# Plugin guide

Calibrated explanations stay extensible without sacrificing the calibrated-first
workflow. Start with the minimal plugin walkthrough below, then apply the
guardrails that keep binary and multiclass classification plus probabilistic and
interval regression aligned with the core library.

## Hello, calibrated plugin

Follow these steps to build a plugin that honours calibrated semantics and
ships with the metadata required by ADR-024/025/026.

### 1. Scaffold the plugin class

Use the :class:`calibrated_explanations.plugins.base.ExplainerPlugin` protocol to
document the surface area. The example below emits a fixed explanation payload;
replace the body with your logic while preserving calibrated outputs.

```python
from __future__ import annotations

from typing import Any

from calibrated_explanations.plugins.base import ExplainerPlugin


class HelloCalibratedPlugin:
    """Minimal example that returns calibrated outputs for a wrapped model."""

    plugin_meta = {
        "schema_version": 1,
        "name": "hello.calibrated.plugin",
        "version": "0.1.0",
        "provider": "example-team",
        "capabilities": ("binary-classification", "probabilistic-regression"),
        "modes": ("factual", "alternative"),
        "tasks": ("classification", "probabilistic-regression"),
        "trusted": False,
    }

    def supports(self, model: Any) -> bool:
        """Return whether the plugin can work with *model*."""

        return hasattr(model, "predict_proba")

    def explain(self, model: Any, x: Any, **kwargs: Any) -> Any:
        """Produce a calibrated explanation payload for ``x``."""

        probabilities = model.predict_proba(x)
        return {
            "prediction": probabilities[:, 1],
            "uncertainty_interval": (probabilities[:, 0], probabilities[:, 1]),
            "modes": self.plugin_meta["modes"],
        }
```

### 2. Validate metadata before registering

Run :func:`calibrated_explanations.plugins.base.validate_plugin_meta` to confirm
required keys match ADR-026. Then register the plugin with an identifier that
follows the dot-separated naming guidelines.

```python
from calibrated_explanations.plugins.base import validate_plugin_meta
from calibrated_explanations.plugins.registry import register_explanation_plugin

plugin = HelloCalibratedPlugin()
validate_plugin_meta(dict(plugin.plugin_meta))
register_explanation_plugin("external.hello.calibrated", plugin)
```

The registration helper updates trust metadata automatically. Optional interval
or plot plugins follow the same pattern using
:func:`calibrated_explanations.plugins.registry.register_interval_plugin` and the
plot builder/renderer helpers.

### 3. Wire the plugin into your package

Expose an entry point under ``calibrated_explanations.plugins`` so discovery can
load your plugin. Fast explanations and other performance-focused options now
live in :mod:`external_plugins`, keeping the core runtime telemetry-optional.

```toml
[project.entry-points."calibrated_explanations.plugins"]
external.hello.calibrated = "your_package.plugins:HelloCalibratedPlugin"
```

Document how probabilistic and interval regression stay calibrated after your
extension by linking back to the practitioner quickstarts and interpretation
guides.

## Guardrails and ADR references

Use these decision records when designing new plugins:

- [ADR-024 – legacy plot input contracts](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-024-legacy-plot-input-contracts.md)
  defines PlotSpec and legacy plot inputs that plugins must honour.
- [ADR-025 – legacy plot rendering semantics](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-025-legacy-plot-rendering-semantics.md)
  documents rendering semantics so PlotSpec builders remain interchangeable.
- [ADR-026 – explanation plugin semantics](https://github.com/Moffran/calibrated_explanations/blob/main/improvement_docs/adrs/ADR-026-explanation-plugin-semantics.md)
  captures the calibrated explanation contract for explanation and interval
  plugins.

### Fast explanations and external bundles

The fast explanations implementation now ships as an external plugin. Install
the aggregated extra and register the bundle explicitly when you need it:

```bash
pip install "calibrated-explanations[external-plugins]"
python -m external_plugins.fast_explanations register
```

See :doc:`external_plugins/index` for community listings and governance notes.

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

{{ optional_extras_template }}

### CLI discovery (optional)

The :mod:`calibrated_explanations.plugins.cli` module surfaces plugin metadata
for auditing. It is optional tooling—set ``CE_DENY_PLUGIN`` during local testing
to simulate trust boundaries before promoting a plugin.

### Telemetry instrumentation (optional)

If your plugin emits telemetry, document the signals and link back to
:doc:`concepts/telemetry`. Keep emission disabled by default so calibrated
explanations remain privacy-preserving out of the box.

### Aggregated install extra (optional)

`pip install "calibrated-explanations[external-plugins]"` installs vetted
external bundles, including fast explanations. Import the module and call its
``register()`` helper to populate the registry explicitly.
