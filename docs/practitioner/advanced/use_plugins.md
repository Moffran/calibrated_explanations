# Use external plugins

Plugins are optional. Start with the calibrated explanations workflow first; only opt in to plugins (like FAST) when you need speed-ups or custom plots. This page shows how to install, register, and wire plugins without changing your core modeling code.

## When to opt in

- Your calibrated factual/alternative runs are validated and you want lower latency.
- You need a different plot style or visualization backend via PlotSpec.
- Your team maintains a vetted plugin aligned to the CE contract.

Note: FAST-mode plugins are opt-in only. By design the runtime will not include
FAST implementations in the default fallback chains for `factual` or
`alternative` explanations — you must explicitly request FAST via the
`fast_plugin` argument or `CE_EXPLANATION_PLUGIN_FAST` / `CE_INTERVAL_PLUGIN_FAST`.

## FAST Explanations

```{admonition} Status: Experimental
:class: warning

FAST explanations are an **experimental** feature. Use them only after validating
your standard factual/alternative workflows. They are not recommended for production
without thorough testing.
```

FAST explanations provide accelerated rule generation through sampling strategies,
trading some detail for significantly improved speed.

### When to Use FAST

- **Large batch processing**: When explaining thousands of instances
- **Real-time applications**: When latency constraints are tight
- **Exploratory analysis**: When approximate explanations suffice

### Trade-offs

| Aspect | Standard Explanations | FAST Explanations |
| :--- | :--- | :--- |
| Speed | Baseline | 2-10x faster |
| Detail | Full perturbation analysis | Sampled perturbations |
| Calibration | Full conformal guarantees | Approximate coverage |
| Stability | Deterministic | May vary between runs |

### Quick Start

```python
# Method 1: Use fast=True kwarg (preferred)
explanation = explainer.explain_factual(x_test, fast=True)

# Method 2: Use explain_fast directly
fast_explanation = explainer.explain_fast(x_test)
```

### Research Background

FAST explanations are documented in:

> Löfström, H., et al. (2024). Fast Calibrated Explanations: Efficient and
> Uncertainty-Aware Explanations for Machine Learning Models.
> In: xAI 2024. Lecture Notes in Computer Science.
> [DOI: 10.1007/978-3-031-44064-9_18](https://link.springer.com/chapter/10.1007/978-3-032-08333-3_16)

## Install and register

Install the curated bundle and register the FAST plugins when needed:

```bash
pip install "calibrated-explanations[external-plugins]"
python -m external_plugins.fast_explanations register
```

See {doc}`../../appendices/external_plugins` for listings and notes.

## Wire it into your run

Prefer explicit, local configuration for development, and use environment variables for CI/CD.

### Method A — Explainer parameters (highest priority)

For modality-aware plugin selection (for example `vision` and `audio` plugin
discovery), see {doc}`./modality-plugins`.

```python
from calibrated_explanations import CalibratedExplainer

explainer = CalibratedExplainer(
    model, x_cal, y_cal,
    # Explanation plugin choice by mode
    factual_plugin="core.explanation.factual",   # or an external id
    alternative_plugin="core.explanation.alternative",
    # Interval calibrator and plot style
    interval_plugin="core.interval.legacy",  # or external interval id
    plot_style="plot_spec.default",                 # or a custom style id
)

explanations = explainer.explain_factual(x_test)
explanations.plot()  # uses plot_style and fallbacks
```

```python
from calibrated_explanations import CalibratedExplainer

explainer = CalibratedExplainer(
    model, x_cal, y_cal,
    # To activate FAST mode globally
    fast=True,
    # or select a specific FAST plugin
    fast_plugin="core.explanation.fast",
    # Interval calibrator and plot style
    interval_plugin="core.interval.fast",  # or external interval id
    plot_style="plot_spec.default",                 # or a custom style id
)

# Use the fast mode to invoke the fast execution path
fast_batch = explainer.explain_fast(x_test)
fast_batch.plot()
```

Notes

- If your chosen explanation plugin declares `plot_dependency` or `interval_dependency`, the explainer seeds those into the fallback chain automatically.
- You can override plot style at render time: `explanations.plot(style_override="legacy")`.

### Method C — Environment variables

Use when you need to switch plugins without code changes (e.g., CI/CD):

```bash
# Plot style and fallbacks
$env:CE_PLOT_STYLE = "plot_spec.default"
$env:CE_PLOT_STYLE_FALLBACKS = "legacy"

# Explanation plugin fallbacks by mode
$env:CE_EXPLANATION_PLUGIN = "core.explanation.factual"
$env:CE_EXPLANATION_PLUGIN_FAST = "core.explanation.fast"
$env:CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS = "core.explanation.factual"
$env:CE_EXPLANATION_PLUGIN_ALTERNATIVE_FALLBACKS = "core.explanation.alternative"
$env:CE_EXPLANATION_PLUGIN_FAST_FALLBACKS = "core.explanation.fast"

# Interval plugin defaults + fallback chain
$env:CE_INTERVAL_PLUGIN = "core.interval.fast"
$env:CE_INTERVAL_PLUGIN_FAST = "core.interval.fast"
$env:CE_INTERVAL_PLUGIN_FALLBACKS = "core.interval.fast,core.interval.legacy"
```

Then construct your explainer without explicit plugin arguments and rely on the configured defaults.

## Troubleshooting and governance

- Trust/deny controls: set `CE_TRUST_PLUGIN` and `CE_DENY_PLUGIN` to allow or block specific identifiers during discovery.
- CLI discovery and audits:

```bash
python -m calibrated_explanations.plugins.cli list all
```

- Keep telemetry optional and off by default; see {doc}`../../foundations/governance/optional_telemetry`.

## Related links

- External plugin index: {doc}`../../appendices/external_plugins`
- Developer contract (for authors): {doc}`../../contributor/plugin-contract`
- Interpretation guide: {doc}`../../foundations/how-to/interpret_explanations`
