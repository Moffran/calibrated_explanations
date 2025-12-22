# Legacy Plotting Reference (Maintenance Only)

This document captures the **legacy plotting contracts** and **rendering semantics** for maintenance and debugging. It is **not** a normative architecture decision and is not required for new rendering backends. The legacy matplotlib renderer remains the canonical reference for pixel-level behaviour.

Source of truth:
- `src/calibrated_explanations/legacy/plotting.py`

## Input contracts (summary)
- Legacy entry points: `_plot_probabilistic`, `_plot_regression`, `_plot_triangular`, `_plot_alternative`, `_plot_global`.
- `show=False` and missing `path/title` short-circuits rendering and bypasses matplotlib dependency checks.
- `save_ext` defaults to `("svg", "pdf", "png")`; filenames are built as `path + title + ext` without separators.
- Instance plots require aligned `features_to_plot` and `instance` arrays and optional `column_names` for labels.
- Interval rendering depends on `interval=True`, non-null `idx`, and non-one-sided explanations.
- Global plots rely on the explainer’s `predict`/`predict_proba` and build uncertainty as `high - low`.

## Rendering semantics (summary)
- Figure width is fixed at 10 inches; height scales with `num_to_show`.
- Probability and interval gauges are drawn with layered `fill_betweenx` bands.
- Classification and regression swap colour palettes and axis labels as described in the legacy code.
- Alternative and triangular plots rely on helper routines (`__plot_proba_triangle`, `__color_brew`, `__get_fill_color`).
- `_plot_global` always calls `plt.show()` unless `show=False` is provided.

## When to consult this doc
- Regression or bugfix work on legacy plots.
- Debugging parity regressions against historical images.
- Supporting tests that inspect exported primitives from the legacy renderer.

For detailed behaviour, consult `src/calibrated_explanations/legacy/plotting.py` and the in-code comments around each helper.
