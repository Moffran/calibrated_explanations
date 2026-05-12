# v0.11.2 PlotSpec Side-by-side Artifacts

This directory contains the Task 9 review artifacts that are actually generated
by `scripts/plot_spec/generate_side_by_side.py`.

Generated image pairs:

- `factual_probabilistic_legacy.png` / `factual_probabilistic_plotspec.png`
- `factual_regression_legacy.png` / `factual_regression_plotspec.png`
- `alternative_probabilistic_legacy.png` / `alternative_probabilistic_plotspec.png`
- `alternative_regression_legacy.png` / `alternative_regression_plotspec.png`
- `triangular_legacy.png` / `triangular_plotspec.png`
- `global_legacy.png` / `global_plotspec.png`
- `global_legacy_regression.png` / `global_plotspec_regression.png`
- `conjunction_legacy.png` / `conjunction_plotspec.png`

Interpretation rule:

- The PlotSpec image in each pair is the current review candidate.
- If a family needed no additional mend after direct verification, there is no separate `plotspec_current` vs `plotspec_mended` split; the delta is recorded in the mending report instead.

Automated semantic verification lives in `tests/unit/viz/`. Human review must
still mark each generated pair as either accepted or needing more mending before
Task 9 can close.
