# Visualization with PlotSpec (MVP)

The PlotSpec abstraction provides a lightweight, backend-agnostic way to describe plots.
For now, it supports a minimal regression-style bar chart with an optional header interval.

Requirements

- Optional extra for plotting: pip install "calibrated_explanations[viz]"

Quick example

```python
from calibrated_explanations.viz import build_regression_bars_spec, matplotlib_adapter

# Minimal fake data similar to the regression plot
predict = {"predict": 3.2, "low": 2.7, "high": 3.8}
feature_weights = {
    "predict": [0.8, -0.4, 0.2],
    "low":     [0.5, -0.7, 0.0],
    "high":    [1.1, -0.1, 0.4],
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
    sort_by="abs",        # one of: none|value|abs|width|label
    ascending=False,
)

# Render with matplotlib adapter (no-op if both show=False and save_path=None)
matplotlib_adapter.render(spec, show=False, save_path="plotspec_example.png")
```

Notes

- Sorting options: none, value, abs, width (interval), label; ascending defaults to False.
- When intervals cross zero for a bar, a transparent interval is drawn without an opaque core bar, matching legacy behavior.
- If matplotlib isn’t installed, a clear error will suggest installing the viz extra.

API reference

- calibrated_explanations.viz.plotspec.PlotSpec
- calibrated_explanations.viz.builders.build_regression_bars_spec
- calibrated_explanations.viz.matplotlib_adapter.render
