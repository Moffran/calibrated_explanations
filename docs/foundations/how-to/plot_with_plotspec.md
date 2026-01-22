# Plot explanations with PlotSpec

```{admonition} Optional plotting extra
:class: tip

PlotSpec stays opt-in for v0.9.0. The quickstarts and notebooks render calibrated explanations without it; enable the PlotSpec extras only when you need uncertainty-aware visuals for reports or audits.
```

PlotSpec is the optional plotting backend introduced alongside the governance refresh. It produces telemetry-aware renderings that document interval sources and fallbacks while keeping the calibrated workflow unchanged.

## Requirements

```bash
pip install "calibrated_explanations[viz]"
```

## Render a regression explanation

```python
from pathlib import Path
from calibrated_explanations.viz import (
    build_regression_bars_spec,
    matplotlib_adapter,
)

predict = {"predict": 3.2, "low": 2.7, "high": 3.8}
feature_weights = {
    "predict": [0.8, -0.4, 0.2],
    "low": [0.5, -0.7, 0.0],
    "high": [1.1, -0.1, 0.4],
}
features_to_plot = [0, 1, 2]
column_names = ["feat_a", "feat_b", "feat_c"]
instance = [1.2, 0.3, -0.1]

spec = build_regression_bars_spec(
    title="Regression feature contributions",
    predict=predict,
    feature_weights=feature_weights,
    features_to_plot=features_to_plot,
    column_names=column_names,
    instance=instance,
    y_minmax=(2.0, 4.0),
    interval=True,
    sort_by="abs",  # one of: none|value|abs|width|label
    ascending=False,
)

output = Path("plotspec_example.png")
matplotlib_adapter.render(spec, show=False, save_path=str(output))
```

`sort_by` accepts `none`, `value`, `abs`, `width`, or `label`. When intervals
cross zero the adapter falls back to translucent bars that match legacy output.

## Runtime telemetry

When you opt into PlotSpec, the library documents the chosen builder and renderer. PlotSpec identifiers are recorded on both the batch telemetry and
`explainer.runtime_telemetry` as `plot_source` and `plot_fallbacks`. Use these
fields to verify whether the PlotSpec adapter or a legacy fallback rendered the
chart.

## CLI discovery

Use the plugin CLI when you need to audit optional plot routes. List registered plot styles and identify defaults:

```bash
python -m calibrated_explanations.plugins.cli list plots
python -m calibrated_explanations.plugins.cli show plot_spec.default.factual --kind plots
```

The CLI output includes builder/renderer IDs, fallback chains, and compatibility
flags.
