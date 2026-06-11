# Deprecation & Migration Guide

This guide documents the deprecations introduced as part of the ADR-011 policy work and provides concrete migration steps, timelines, and a status table to help library users and downstream integrators.

## Goals
- Centralise deprecation emission and behaviour via `deprecate()` so messages are consistent and can be toggled to raise in CI (`CE_DEPRECATIONS`).
- Provide clear migration examples for common deprecated symbols and aliases.
- Inform maintainers about the default two-minor-release deprecation window, plus the binding pre-v1.0 finalization exception that requires full closure by v0.11.3.

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

### ADR-034 centralized configuration migration (v0.11.1)

- Runtime modules now resolve environment and `pyproject.toml` config through
  `ConfigManager` snapshots, not live ad-hoc reads.
- Snapshot behavior is intentional: changes to process env vars after manager
  construction are only visible after reconstructing the owning runtime object.
- CLI configuration diagnostics are available via:
  - `ce config show`
  - `ce config export`

### Plugin registration list-path API (closed by v0.11.3)

The legacy list-path plugin APIs were deprecated in `v0.11.1` and removed in
`v0.11.3`.

| Legacy API | Replacement |
|---|---|
| `register(plugin)` | `register_explanation_plugin(identifier, plugin, metadata)` |
| `trust_plugin(plugin)` | Register with `metadata={"trusted": True, ...}` via `register_explanation_plugin(...)` |
| `find_for(model)` | `find_explanation_plugin_for(..., model=model, trusted_only=False)` |
| `find_for_trusted(model)` | `find_explanation_plugin_for(..., model=model, trusted_only=True)` |

```py
# register(plugin)
# before
registry.register(plugin)

# after
registry.register_explanation_plugin(
    identifier=plugin.plugin_meta["name"],
    plugin=plugin,
    metadata=plugin.plugin_meta,
)
```

```py
# trust_plugin(plugin)
# before
registry.register(plugin)
registry.trust_plugin(plugin)

# after
meta = dict(plugin.plugin_meta)
meta["trusted"] = True
registry.register_explanation_plugin(
    identifier=meta["name"],
    plugin=plugin,
    metadata=meta,
)
```

```py
# find_for(model)
# before
plugins = registry.find_for(model)

# after
identifier, plugin = registry.find_explanation_plugin_for(
    "tabular",
    mode="factual",
    task="classification",
    model=model,
    trusted_only=False,
)
plugins = (plugin,)
```

```py
# find_for_trusted(model)
# before
trusted_plugins = registry.find_for_trusted(model)

# after
identifier, trusted_plugin = registry.find_explanation_plugin_for(
    "tabular",
    mode="factual",
    task="classification",
    model=model,
    trusted_only=True,
)
trusted_plugins = (trusted_plugin,)
```

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
- Default policy: a deprecation introduced in `vX.Y.Z` remains for at least two minor releases before removal.
- Finalization override: for the v1.0.0 cleanup window, all active deprecations must be removed by v0.11.3. No deprecation remains active in v1.0.0.

## Status table

**Binding rule for this table:** every row in **Active deprecations** must move to **Removed deprecations (history)** by the end of v0.11.3.

### Task 21 inventory (v0.11.1): core-surface LIME/SHAP deprecations

The v0.11.1 API-bloat removal program inventories ten LIME/SHAP core-surface entry points.
All ten now emit `deprecate()` warnings and are assigned to explicit pre-v1.0 removal milestones.

### Active deprecations

Symbols listed here still emit warnings. Stop using them — they will be removed on the date shown.

| Deprecated symbol | Replacement | Deprecated since | Removal ETA | Notes |
|---|---|---:|---:|---|

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
| `CalibratedExplainer.preload_lime(...)` | `external_plugins.integrations.lime_pipeline.LimePipeline(explainer).preload(...)` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; core helper preload path deleted. |
| `CalibratedExplainer.preload_shap(...)` | `external_plugins.integrations.shap_pipeline.ShapPipeline(explainer).preload(...)` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; core helper preload path deleted. |
| `CalibratedExplainer.explain_lime(...)` | `external_plugins.integrations.lime_pipeline.LimePipeline(explainer).explain(...)` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; runtime explanation path is plugin-only. |
| `CalibratedExplainer.explain_shap(...)` | `external_plugins.integrations.shap_pipeline.ShapPipeline(explainer).explain(...)` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; runtime explanation path is plugin-only. |
| `CalibratedExplainer.is_lime_enabled(...)` | `external_plugins.integrations.lime_pipeline.LimePipeline(explainer).is_enabled()` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; core helper toggle removed. |
| `CalibratedExplainer.is_shap_enabled(...)` | `external_plugins.integrations.shap_pipeline.ShapPipeline(explainer).is_enabled()` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; core helper toggle removed. |
| `WrapCalibratedExplainer.explain_lime(...)` | `external_plugins.integrations.lime_pipeline.LimePipeline(wrapper).explain(...)` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; wrapper forwarding hook deleted. |
| `WrapCalibratedExplainer.explain_shap(...)` | `external_plugins.integrations.shap_pipeline.ShapPipeline(wrapper).explain(...)` | v0.11.1 | v0.11.2 | Task-21 inventory item removed in Task 5A; wrapper forwarding hook deleted. |
| `calibrated_explanations.perf.cache` | `calibrated_explanations.cache` | v0.10.x | v0.11.3 | Shim `perf/cache.py` deleted; `ImportError` on import. |
| `calibrated_explanations.perf.parallel` | `calibrated_explanations.parallel` | v0.10.x | v0.11.3 | Shim `perf/parallel.py` deleted; `ImportError` on import. |
| Imports from `core.calibration_helpers` (`assign_threshold`, `initialize_interval_learner`, `initialize_interval_learner_for_fast_explainer`, `update_interval_learner`) | `calibrated_explanations.calibration.interval_learner` | v0.10.x | v0.11.3 | Lazy `__getattr__` shim removed from `core/calibration_helpers.py`; `AttributeError` on access. |
| Plugin `modes` value `"explanation:factual"` | `"factual"` | v0.10.x | v0.11.3 | `_EXPLANATION_MODE_ALIASES` removed from `plugins/registry.py`; `ValidationError` on registration with old mode string. |
| Plugin `modes` value `"explanation:alternative"` | `"alternative"` | v0.10.x | v0.11.3 | Same. |
| Plugin `modes` value `"explanation:fast"` | `"fast"` | v0.10.x | v0.11.3 | Same. |
| `ParallelConfig(granularity="feature")` | `granularity="instance"` | v0.10.x | v0.11.3 | `CE_PARALLEL granularity=feature` now raises `ConfigurationError`; `perf_parallel_granularity` config field accepts only `"instance"`. |
| `RejectPolicy.PREDICT_AND_FLAG`, `RejectPolicy.EXPLAIN_ALL` | `RejectPolicy.FLAG` | v0.10.x | v0.11.3 | `_missing_` removed from `RejectPolicy`; `__getattr__` removed from `core/reject/policy.py`. Old string values raise `ValueError`; module attribute access raises `AttributeError`. |
| `RejectPolicy.EXPLAIN_REJECTS` | `RejectPolicy.ONLY_REJECTED` | v0.10.x | v0.11.3 | Same. |
| `RejectPolicy.EXPLAIN_NON_REJECTS`, `RejectPolicy.SKIP_ON_REJECT` | `RejectPolicy.ONLY_ACCEPTED` | v0.10.x | v0.11.3 | Same. |
| `calibrated_explanations.core.calibration` (package) | `calibrated_explanations.calibration` | v0.10.x | v0.11.3 | ADR-001 Stage 1a shims removed; import from the canonical calibration package. |
| `CalibratedExplainer.initialize_reject_learner(...)` | `explainer.reject_orchestrator.initialize_reject_learner(...)` | v0.11.1 | v0.11.3 | Public reject delegator removed; use the reject orchestrator directly. |
| `CalibratedExplainer.predict_reject(...)` | `explainer.reject_orchestrator.predict_reject(...)` | v0.11.1 | v0.11.3 | Public reject delegator removed; use the reject orchestrator directly. |
| `WrapCalibratedExplainer.initialize_reject_learner(...)` | `wrapper.explainer.reject_orchestrator.initialize_reject_learner(...)` | v0.11.1 | v0.11.3 | Wrapper delegator removed; use the wrapped explainer's reject orchestrator. |
| `WrapCalibratedExplainer.predict_reject(...)` | `wrapper.explainer.reject_orchestrator.predict_reject(...)` | v0.11.1 | v0.11.3 | Wrapper delegator removed; use the wrapped explainer's reject orchestrator. |
| `CalibratedExplainer.build_plot_style_chain(...)` | `explainer.plugin_manager.build_plot_chain(...)` | v0.11.1 | v0.11.3 | Non-essential plugin-manager delegator removed. |
| `CalibratedExplainer.instantiate_plugin(...)` | `explainer.plugin_manager.explanation_orchestrator.instantiate_plugin(...)` | v0.11.1 | v0.11.3 | Non-essential plugin-manager delegator removed. |
| `CalibratedExplainer.invoke_explanation_plugin(...)` | `explainer.explanation_orchestrator.invoke(...)` | v0.11.1 | v0.11.3 | Non-essential plugin-manager delegator removed. |
| `CalibratedExplainer.ensure_interval_runtime_state(...)` | `explainer.prediction_orchestrator.ensure_interval_runtime_state(...)` | v0.11.1 | v0.11.3 | Non-essential prediction-orchestrator delegator removed. |
| `CalibratedExplainer.gather_interval_hints(...)` | `explainer.prediction_orchestrator.gather_interval_hints(...)` | v0.11.1 | v0.11.3 | Non-essential prediction-orchestrator delegator removed. |
| `CalibratedExplainer.interval_plugin_hints`, `.interval_plugin_fallbacks`, `.interval_preferred_identifier`, `.telemetry_interval_sources`, `.interval_plugin_identifiers`, `.interval_context_metadata` | `explainer.plugin_manager.<same name>` | v0.11.1 | v0.11.3 | Public plugin-manager state aliases removed from `CalibratedExplainer`. |
| `CalibratedExplainer.explanation_plugin_overrides`, `.interval_plugin_override`, `.fast_interval_plugin_override`, `.plot_style_override` | `explainer.plugin_manager.<same name>` | v0.11.1 | v0.11.3 | Public plugin-manager override aliases removed from `CalibratedExplainer`. |
| `CalibratedExplanations.as_lime(...)` | `external_plugins.integrations.lime_pipeline.LimePipeline(...).explain(...)` | v0.11.1 | v0.11.3 | Collection adapter removed after v0.11.2 core hook deletion. |
| `CalibratedExplanations.as_shap(...)` | `external_plugins.integrations.shap_pipeline.ShapPipeline(...).explain(...)` | v0.11.1 | v0.11.3 | Collection adapter removed after v0.11.2 core hook deletion. |
| `plugins.registry.register(plugin)` | `register_explanation_plugin(identifier, plugin, metadata)` | v0.11.1 | v0.11.3 | List-path API removed; use identifier-based registry APIs. |
| `plugins.registry.trust_plugin(plugin)` | `register_explanation_plugin(..., metadata={"trusted": True, ...})` or `mark_explanation_trusted(identifier)` | v0.11.1 | v0.11.3 | List-path API removed; use identifier-based trust APIs. |
| `plugins.registry.find_for(model)` | `find_explanation_plugin_for(..., trusted_only=False)` | v0.11.1 | v0.11.3 | List-path API removed; use descriptor-based resolution. |
| `plugins.registry.find_for_trusted(model)` | `find_explanation_plugin_for(..., trusted_only=True)` | v0.11.1 | v0.11.3 | List-path API removed; use descriptor-based resolution. |
| `RejectResult` active deprecation warning path in `reject_result_v2_to_legacy()` | `RejectResult` remains stable in v1.0.0; use `RejectResultV2` as opt-in strict schema | v0.11.x | v0.11.3 | Group L resolved via deprecation reset path: removed active warning targeting v1.0.0-rc to comply with ADR-011 finalization exception. |
| `CalibratedExplainer.explain_guarded_factual(...)` | `explainer.explain_factual(..., guarded=True)` | v0.11.3 | v1.0.0 | Guarded normalized as a boolean policy flag on `explain_factual`; separate method removed (ADR-032). |
| `CalibratedExplainer.explore_guarded_alternatives(...)` | `explainer.explore_alternatives(..., guarded=True)` | v0.11.3 | v1.0.0 | Guarded normalized as a boolean policy flag on `explore_alternatives`; separate method removed (ADR-032). |
| `WrapCalibratedExplainer.explain_guarded_factual(...)` | `wrapper.explain_factual(..., guarded=True)` | v0.11.3 | v1.0.0 | Same as above — wrapper delegates to `explain_factual(guarded=True)`. |
| `WrapCalibratedExplainer.explore_guarded_alternatives(...)` | `wrapper.explore_alternatives(..., guarded=True)` | v0.11.3 | v1.0.0 | Same as above — wrapper delegates to `explore_alternatives(guarded=True)`. |

## Breaking changes

### Guarded entrypoints now fail on calibration-feature divergence (v0.11.1+)

`explain_factual(guarded=True)` and `explore_alternatives(guarded=True)` now raise
`ValidationError` when the active prediction backend is not using the same
calibration feature matrix as `explainer.x_cal`.

**Why:** Guarded filtering and interval predictions must share the same
calibration-feature values to preserve the guarded exchangeability assumption.

**Migration:**

- Recalibrate the explainer before calling guarded entrypoints if you have rebuilt or swapped interval learners.
- Do not mutate or replace the backend calibration features independently of `explainer.x_cal`.
- Use `explain_factual(..., guarded=True)` / `explore_alternatives(..., guarded=True)` — the old `explain_guarded_factual` / `explore_guarded_alternatives` methods are deprecated as of v0.11.3.

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
- In the v0.11.x finalization window, each new/remaining deprecation entry must include explicit removal ownership in v0.11.2 or v0.11.3.
- Add a unit test in `tests/unit/` validating the desired behaviour of `deprecate()` if you change its semantics.

## Troubleshooting

- If CI shows a `DeprecationWarning` raised due to `CE_DEPRECATIONS=error`, run locally with that env var set to reproduce and update callsites accordingly.

## Contact

If you're unsure about a migration, open an issue.
