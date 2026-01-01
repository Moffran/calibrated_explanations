# Legacy Plotting Reference (Maintenance Only)

This document captures the **legacy plotting contracts** and **rendering semantics** for maintenance and debugging. It is **not** a normative architecture decision and is not required for new rendering backends. The legacy matplotlib renderer remains the canonical reference for pixel-level behaviour.

Source of truth:
- `src/calibrated_explanations/legacy/plotting.py`

## Status
- ADR-024 / ADR-025: superseded — do not reintroduce ADR-enforced contract changes for legacy plotting. This maintenance reference documents the current observed behaviors and conventions and is the canonical source for stability-focused fixes.

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

## Fallback & visibility policy
- Legacy plot code follows the repo-wide fallback visibility policy: any runtime fallback or simplification MUST emit an `INFO` log and a `warnings.warn(..., UserWarning)` (see `.github/copilot-instructions.md` and `docs/improvement/RELEASE_PLAN_v1.md`). Tests that rely on fallbacks should assert the warning via `pytest.warns(UserWarning)`.

## Testing guidance
- Existing legacy plotting regression tests live in `tests/legacy/test_plotting.py`. Prefer extending these for fixes to preserve image parity.
- For new tests that exercise fallback behavior, use `pytest.warns(UserWarning)` to assert visible fallbacks.
- When comparing images, prefer pixel-tolerant assertions or checksum comparisons produced by the existing image-fixture tooling (see `tests/legacy/README.md` if present).

## When to consult this doc
- Regression or bugfix work on legacy plots.
- Debugging parity regressions against historical images.
- Supporting tests that inspect exported primitives from the legacy renderer.

For detailed behaviour, consult `src/calibrated_explanations/legacy/plotting.py` and the in-code comments around each helper.
