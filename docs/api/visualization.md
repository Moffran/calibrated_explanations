# Visualization & PlotSpec reference

Calibrated Explanations renders plots through PlotSpec objects and visualization
adapters.

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
v0.11.2 follow-up decision in the release plan.

## Related ADRs

- ADR-036 (PlotSpec canonical contract and validation boundary)
- ADR-037 (visualization extension and rendering governance)
