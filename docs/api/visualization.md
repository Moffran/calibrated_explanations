# Visualization & PlotSpec reference

Calibrated Explanations renders plots through PlotSpec objects and visualization
adapters.

## Current default posture

As of **v0.11.3 Task 6**, PlotSpec is the default user-facing plotting path for:

- `plot_probabilistic`
- `plot_regression`
- `plot_alternative`
- `plot_triangular`
- `plot_global`
- factual, alternative, and batch explanation `.plot()` entrypoints that route
  into those functions

This default change follows the v0.11.2 semantic/visual mending evidence and
human review recorded under `docs/improvement/plot_spec/`.

## Legacy opt-out and fallback

Legacy rendering remains available as an explicit opt-out and fallback path:

- `use_legacy=True`
- `style_override="legacy"` where the plotting function accepts style overrides
- explicit legacy style configuration in plugin/style resolution surfaces

When PlotSpec rendering fails and execution falls back to legacy, the fallback
must remain visible through a `UserWarning` and an INFO log entry.

PlotSpec can still be requested explicitly with:

- `use_legacy=False`
- `style_override="plot_spec.default"`
- `return_plot_spec=True`
- configuration that explicitly prefers `plot_spec.default`

## v0.11.2 Task 9 mending status

The opt-in PlotSpec path now has semantic parity coverage for all six planned
families: factual probabilistic, factual regression, alternative probabilistic,
alternative regression, triangular, and global. Triangular and global PlotSpec
dataclasses render through concrete matplotlib adapter branches rather than
placeholder primitives. Global PlotSpec payloads preserve threshold and
class-label semantics so thresholded and class-conditioned axis labels remain
available to renderers.

This evidence is the basis for the v0.11.3 default promotion.

## Core modules

- `src/calibrated_explanations/viz/plotspec.py` – PlotSpec dataclasses and
  required metadata fields.
- `src/calibrated_explanations/viz/serializers.py` – JSON serialization helpers
  and validation utilities.
- `src/calibrated_explanations/viz/matplotlib_adapter.py` – Matplotlib rendering
  and headless export support.
- `src/calibrated_explanations/viz/builders.py` – PlotSpec builders for factual,
  alternative, and global views.

## Plot kinds

Plot kinds are validated by `viz.serializers.PlotKindRegistry` using the
built-in `_SUPPORTED_KINDS` contract in this release. The PlotSpec
`kind`/`mode` metadata determines which renderer and validation rules apply.
Runtime `kind` registration remains out of scope under ADR-037.
Any change to that policy requires a later ADR/amendment and the explicit
follow-up decision in the release plan.

## Related ADRs

- ADR-036 (PlotSpec canonical contract and validation boundary)
- ADR-037 (visualization extension and rendering governance)

Entry-point tier: Tier 3.
