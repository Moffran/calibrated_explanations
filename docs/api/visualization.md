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

Plot kinds are registered and validated through the plot plugin registry. The
PlotSpec `kind`/`mode` metadata determines which renderer and validation rules
apply.

## Related ADRs

- ADR-007 (visualization abstraction)
- ADR-014 (plot plugin strategy)
- ADR-016 (PlotSpec separation)
