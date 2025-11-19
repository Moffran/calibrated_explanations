# Deprecation & Migration Guide

This guide documents the deprecations introduced as part of the ADR-011 policy work and provides concrete migration steps, timelines, and a status table to help library users and downstream integrators.

## Goals
- Centralise deprecation emission and behaviour via `deprecate()` so messages are consistent and can be toggled to raise in CI (`CE_DEPRECATIONS`).
- Provide clear migration examples for common deprecated symbols and aliases.
- Inform maintainers about the two-minor-release deprecation window and CI checks.

## Where the helper lives

The new helper is implemented at:

```
src/calibrated_explanations/utils/deprecations.py
```

Use `from calibrated_explanations.utils.deprecations import deprecate, deprecate_alias`.

## Recommended migration steps for callers

1. Replace use of deprecated APIs as documented below. Where you control the calling code, update to the canonical API.
2. If you rely on third-party libraries that emit deprecation warnings, pin those libraries or file an issue requesting they adopt the central helper.
3. For CI enforcement, set `CE_DEPRECATIONS=error` temporarily to catch any remaining deprecation uses during migration.

## Common deprecated items and migration examples

- `CalibratedExplanations.get_explanation(index)` → Use indexing: `explanations[index]`.

  Example:

  ```py
  # old
  e = explanations.get_explanation(0)

  # new
  e = explanations[0]
  ```

- `CalibratedExplainer.explain_counterfactual(...)` → `explore_alternatives(...)`.

  ```py
  # old
  alt = explainer.explain_counterfactual(x)

  # new
  alt = explainer.explore_alternatives(x)
  ```

- `calibrated_explanations.core` legacy import → use the package façade (e.g. `from calibrated_explanations.core import CalibratedExplainer`). The legacy module import emits a single deprecation warning on first import.

- Parameter aliases: `alpha` / `alphas` → `low_high_percentiles` (canonical key). Use `canonicalize_kwargs` or supply the canonical keyword. The alias helper will emit a deprecation warning when present.

- `register_plot_plugin(...)` → use `register_plot_builder(...)` and `register_plot_renderer(...)` separately.

## Migration timeline and policy

- Deprecation messages are emitted once-per-session by default and can be elevated to errors by setting `CE_DEPRECATIONS=error` in CI.
- The project follows a two-minor-release deprecation window: a message introduced in `vX.Y.Z` will remain for at least `vX.(Y+2).0` before removal unless explicitly called out in an ADR.

## Status table

| Deprecated symbol | Replacement | Introduced | Removal ETA | Notes |
|---|---|---:|---:|---|
| `get_explanation` (CalibratedExplanations) | indexing (`[i]`) | v0.9.0 | v0.11.0 | Helper converted to use `deprecate()`; update callsites.
| `explain_counterfactual` | `explore_alternatives` | v0.9.0 | v0.11.0 | Backward-compatible delegator emits deprecation; replace directly.
| `calibrated_explanations.core` legacy module import | package façade | v0.9.0 | v0.11.0 | Single-session DeprecationWarning emitted on import.
| `register_plot_plugin` | `register_plot_builder` + `register_plot_renderer` | v0.9.0 | v0.11.0 | Converted to call `deprecate()`; update plugin registration code.
| Parameter aliases (`alpha`, `alphas`) | `low_high_percentiles` | v0.9.0 | v0.11.0 | Use `canonicalize_kwargs` or pass canonical key; library will warn on aliases.

## For maintainers

- When introducing a deprecation, use `deprecate(message, key="unique:key", stacklevel=3)` and prefer a stable `key` value.
- Add a line to this document and update the release plan (`improvement_docs/RELEASE_PLAN_V1.md`) under ADR-011 when new items are introduced.
- Add a unit test in `tests/unit/` validating the desired behaviour of `deprecate()` if you change its semantics.

## Troubleshooting

- If CI shows a `DeprecationWarning` raised due to `CE_DEPRECATIONS=error`, run locally with that env var set to reproduce and update callsites accordingly.

## Contact

If you're unsure about a migration, open an issue.
