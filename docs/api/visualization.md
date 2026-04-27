# Visualization & PlotSpec reference

Calibrated Explanations renders plots through PlotSpec objects and visualization
adapters.

## Current default posture

As of **v0.11.2 task 7 (decision recorded 2026-04-23)**, PlotSpec remains
**non-default** for user-facing plotting. Legacy plotting is still the default
route while PlotSpec visual semantics are hardened under v0.11.2 Task 9.

The v0.11.2 decision is a deliberate **1A-family deferral**:

- `plot_probabilistic`, `plot_regression`, and `plot_alternative` remain
  non-default for end users.
- `plot_triangular` and `plot_global` also remain non-default and are deferred
  to the explicit v0.11.3 promotion re-evaluation/finalization task.

This means PlotSpec may be structurally correct and usable, but is not yet the
default user-facing visual path.

## Opt-in usage

PlotSpec can still be exercised explicitly in v0.11.2:

- `use_legacy=False`
- `style_override="plot_spec.default"`
- `return_plot_spec=True`
- configuration that explicitly prefers `plot_spec.default`

Task 9 in `docs/improvement/v0.11.2_plan.md` is the active semantic/visual
mending task for the opt-in PlotSpec path. The default-promotion question is
re-opened in v0.11.3 Task 6 only after that mending work is complete.

## v0.11.2 Task 9 mending status

The opt-in PlotSpec path now has semantic parity coverage for all six planned
families: factual probabilistic, factual regression, alternative probabilistic,
alternative regression, triangular, and global. Triangular and global PlotSpec
dataclasses render through concrete matplotlib adapter branches rather than
placeholder primitives. Global PlotSpec payloads preserve threshold and
class-label semantics so thresholded and class-conditioned axis labels remain
available to renderers.

This does not change the default route. Human side-by-side review and sign-off
remain required before any later default-promotion decision.

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
Runtime `kind` registration is out of scope in v0.11.1 under ADR-037.
Any change to that policy requires a later ADR/amendment and the explicit
v0.11.2/v0.11.3 follow-up decisions in the release plan.

## Related ADRs

- ADR-036 (PlotSpec canonical contract and validation boundary)
- ADR-037 (visualization extension and rendering governance)
