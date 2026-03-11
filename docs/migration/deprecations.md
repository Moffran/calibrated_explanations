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

- `CalibratedExplanations.get_explanation(index)` was removed → Use indexing: `explanations[index]`.

  Example:

  ```py
  # old
  e = explanations.get_explanation(0)

  # new
  e = explanations[0]
  ```

- `WrapCalibratedExplainer.explain_counterfactual(...)` was removed → `explore_alternatives(...)`.

  ```py
  # old
  alt = explainer.explain_counterfactual(x)

  # new
  alt = explainer.explore_alternatives(x)
  ```

- `calibrated_explanations.core` no longer emits the legacy module deprecation warning.

- Parameter aliases `alpha` / `alphas` were removed in v0.11.0 → use `low_high_percentiles`.

- `register_plot_plugin(...)` was removed → use `register_plot_builder(...)` and `register_plot_renderer(...)` separately.

- Parameter alias `n_jobs` was removed in v0.11.0 → use `parallel_workers`:

  ```py
  # old
  explainer.explain_factual(x, n_jobs=4)

  # new
  explainer.explain_factual(x, parallel_workers=4)
  ```

- `calibrated_explanations.core.calibration` import → top-level:

  ```py
  # old
  from calibrated_explanations.core.calibration import IntervalRegressor, VennAbers

  # new
  from calibrated_explanations.calibration import IntervalRegressor, VennAbers
  ```

- `RejectPolicy` renamed aliases:

  ```py
  # old
  from calibrated_explanations import RejectPolicy
  policy = RejectPolicy.PREDICT_AND_FLAG   # or EXPLAIN_ALL
  policy = RejectPolicy.EXPLAIN_REJECTS
  policy = RejectPolicy.EXPLAIN_NON_REJECTS  # or SKIP_ON_REJECT

  # new
  policy = RejectPolicy.FLAG
  policy = RejectPolicy.ONLY_REJECTED
  policy = RejectPolicy.ONLY_ACCEPTED
  ```

## Migration timeline and policy

- Deprecation messages are emitted once-per-session by default and can be elevated to errors by setting `CE_DEPRECATIONS=error` in CI.
- The project follows a two-minor-release deprecation window: a message introduced in `vX.Y.Z` will remain for at least `vX.(Y+2).0` before removal unless explicitly called out in an ADR.

## Status table

### Active deprecations

Symbols listed here still emit warnings. Stop using them — they will be removed on the date shown.

| Deprecated symbol | Replacement | Deprecated since | Removal ETA | Notes |
|---|---|---:|---:|---|
| `calibrated_explanations.core.calibration` (package) | `calibrated_explanations.calibration` | v0.10.x | v1.0.0 | ADR-001 Stage 1a. Shims in `core/calibration/__init__.py` and submodule files (`interval_learner`, `interval_regressor`, `state`, `summaries`, `venn_abers`). |
| `calibrated_explanations.perf.cache` | `calibrated_explanations.cache` | v0.10.x | v1.0.0 | Shim in `perf/cache.py`. |
| `calibrated_explanations.perf.parallel` | `calibrated_explanations.parallel` | v0.10.x | v1.0.0 | Shim in `perf/parallel.py`. |
| Imports from `core.calibration_helpers` (`assign_threshold`, `initialize_interval_learner`, `initialize_interval_learner_for_fast_explainer`, `update_interval_learner`) | `calibrated_explanations.calibration.interval_learner` | v0.10.x | v1.0.0 | Lazy `__getattr__` shim in `core/calibration_helpers.py`. |
| `RejectPolicy.PREDICT_AND_FLAG`, `RejectPolicy.EXPLAIN_ALL` | `RejectPolicy.FLAG` | v0.10.x | v1.0.0 | Aliases in `explanations/reject.py` (`_missing_`) and `core/reject/policy.py` (`__getattr__`). |
| `RejectPolicy.EXPLAIN_REJECTS` | `RejectPolicy.ONLY_REJECTED` | v0.10.x | v1.0.0 | Same files. |
| `RejectPolicy.EXPLAIN_NON_REJECTS`, `RejectPolicy.SKIP_ON_REJECT` | `RejectPolicy.ONLY_ACCEPTED` | v0.10.x | v1.0.0 | Same files. |
| Plugin `modes` value `"explanation:factual"` | `"factual"` | v0.10.x | v1.0.0 | `plugins/registry.py` validates and warns on old mode aliases. |
| Plugin `modes` value `"explanation:alternative"` | `"alternative"` | v0.10.x | v1.0.0 | Same. |
| Plugin `modes` value `"explanation:fast"` | `"fast"` | v0.10.x | v1.0.0 | Same. |
| `ParallelConfig(granularity="feature")` | `granularity="instance"` | v0.10.x | v1.0.0 | `parallel/parallel.py` silently upgrades the value and warns. |
| `CalibratedExplainer.initialize_reject_learner(...)` | `explainer.reject_orchestrator.initialize_reject_learner(...)` | v0.11.1 | v0.13.0/v1.0.0 | Compatibility wrapper retained for migration; emits `deprecate()` warning. |
| `CalibratedExplainer.predict_reject(...)` | `explainer.reject_orchestrator.predict_reject(...)` | v0.11.1 | v0.13.0/v1.0.0 | Compatibility wrapper retained for migration; emits `deprecate()` warning. |
| `WrapCalibratedExplainer.initialize_reject_learner(...)` | `wrapper.explainer.reject_orchestrator.initialize_reject_learner(...)` | v0.11.1 | v0.13.0/v1.0.0 | Wrapper parity deprecation aligned with explainer-level deprecation. |
| `WrapCalibratedExplainer.predict_reject(...)` | `wrapper.explainer.reject_orchestrator.predict_reject(...)` | v0.11.1 | v0.13.0/v1.0.0 | Wrapper parity deprecation aligned with explainer-level deprecation. |

### Removed deprecations (history)

Symbols listed here have been deleted. Any remaining usage will raise `AttributeError` or `ImportError`.

| Deprecated symbol | Replacement | Deprecated since | Removed in | Notes |
|---|---|---:|---:|---|
| `get_explanation(index)` (CalibratedExplanations) | `explanations[index]` | v0.9.0 | v0.11.0 | Removed from the base collection API. |
| `explain_counterfactual(...)` (WrapCalibratedExplainer) | `explore_alternatives(...)` | v0.9.0 | v0.11.0 | Removed alias from wrapper API. |
| `calibrated_explanations.core` legacy module warning path | package façade | v0.9.0 | v0.11.0 | Legacy deprecation warning removed. |
| `register_plot_plugin(...)` | `register_plot_builder(...)` + `register_plot_renderer(...)` | v0.9.0 | v0.11.0 | Compatibility shim removed. |
| Parameter aliases `alpha`, `alphas` | `low_high_percentiles` | v0.9.0 | v0.11.0 | Removed alias mapping/warning path; calls now fail fast with `ConfigurationError`. |
| Parameter alias `n_jobs` | `parallel_workers` | v0.9.0 | v0.11.0 | Removed alias mapping/warning path; calls now fail fast with `ConfigurationError`. |
| Top-level package exports (`AlternativeExplanation`, `FactualExplanation`, `FastExplanation`, `AlternativeExplanations`, `CalibratedExplanations`, `BinaryEntropyDiscretizer`, `BinaryRegressorDiscretizer`, `EntropyDiscretizer`, `RegressorDiscretizer`, `IntervalRegressor`, `VennAbers`) | Import from respective submodules | v0.9.0 | v0.11.0 | Verified by `tests/unit/test_package_init_deprecation.py`. |
| `calibrated_explanations.perf` root facade | `calibrated_explanations.cache` + `calibrated_explanations.parallel` | v0.10.x | v0.11.0 | `perf/__init__.py` is now empty. |

## Breaking changes

### Reject NCF public contract simplified (v0.11.1+)

Reject NCF user-facing inputs are now limited to `default` and `ensured`.

- `ncf="default"`: task-dependent internal score (`hinge` for binary + thresholded regression, `margin` for multiclass).
- `ncf="ensured"`: `score = (1 - w) * interval_width + w * default_score`.
- Legacy `ncf="entropy"` remains accepted and is silently normalized to `ncf="default"`.
- Explicit `ncf="hinge"` and `ncf="margin"` are no longer accepted and now raise `ValidationError`.

### Default `condition_source` changed to `"prediction"` (v0.11.0)

Starting in v0.11.0, the default value for the `condition_source` parameter in `CalibratedExplainer` has changed from `"observed"` to `"prediction"`. This change enhances the consistency of calibrated explanations by basing condition labels on model predictions rather than observed labels.

**Migration:**

If your code previously relied on the default behavior (condition labels derived from observed labels), you must now explicitly set `condition_source="observed"` when initializing the explainer:

```python
# Before (implicitly used "observed")
explainer = CalibratedExplainer(model, x_cal, y_cal)

# After (explicitly retain old behavior)
explainer = CalibratedExplainer(model, x_cal, y_cal, condition_source="observed")

# New default behavior (recommended)
explainer = CalibratedExplainer(model, x_cal, y_cal)  # Uses "prediction"
```

A warning is issued when `condition_source` is not provided, guiding users to the new default. This change does not affect existing code that explicitly sets `condition_source="observed"`. For more details, see the [API documentation](../api/calibrated_explainer.md).

## For maintainers

- When introducing a deprecation, use `deprecate(message, key="unique:key", stacklevel=3)` and prefer a stable `key` value.
- Add a line to this document and update the release plan (`docs/improvement/RELEASE_PLAN_v1.md`) under ADR-011 when new items are introduced.
- Add a unit test in `tests/unit/` validating the desired behaviour of `deprecate()` if you change its semantics.

## Troubleshooting

- If CI shows a `DeprecationWarning` raised due to `CE_DEPRECATIONS=error`, run locally with that env var set to reproduce and update callsites accordingly.

## Contact

If you're unsure about a migration, open an issue.
