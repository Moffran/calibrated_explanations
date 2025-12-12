# Internal FAST-based feature filtering for factual/alternative explanations

This document outlines how to use FAST explanations internally to filter features **per batch and per instance** before running the more expensive `explain_factual` / `explore_alternatives` paths.

The aim is to reduce compute by skipping unimportant features while preserving per-instance nuance and keeping the public `CalibratedExplainer` API unchanged.

## Goals and constraints

- **Per-batch, per-instance filtering**  
  - For each call to `explain_factual(x, ...)` or `explore_alternatives(x, ...)`, run an internal FAST pass on the *same* batch `x`.  
  - Use the per-instance feature weights from this FAST batch to decide which features to keep for that batch; no global or cross-batch ranking.

- **No new helpers on `CalibratedExplainer`**  
  - All new logic lives in orchestrators, plugins, or new internal modules (e.g. `core.explain._feature_filter`).  
  - `CalibratedExplainer` methods (`explain_factual`, `explore_alternatives`, `explain_fast`) remain thin delegators.

- **Coexistence of modes**  
  - A single explainer must continue to support `explain_factual`, `explore_alternatives`, and `explain_fast` concurrently.  
  - The internal filter uses FAST via the plugin/orchestrator system; it must not preclude user-facing FAST calls.

- **Executor lifetime & performance**  
  - Reuse the same `ParallelExecutor` for both the internal FAST pass and the factual/alternative execution, matching the “enter once, reuse pool” guidance from `parallel_parallelism_analysis.md`.  
  - Never spin up a new pool per `map` call inside the filter.

- **Graceful degradation**  
  - If the FAST plugin is unavailable (not installed, denied via `CE_DENY_PLUGIN`, misconfigured) or fails at runtime, fall back to the current behaviour with no filtering and omit a UserWarning and log message.  
  - Feature filtering must never break or change the semantics of existing calls beyond skipping unimportant features.

- **Telemetry and testing**  
  - Add unit tests for the new filter module and plugin integration.  
  - Add integration tests to verify executor reuse across both stages.  
  - Optionally log telemetry events:
    - feature filtering is enabled,
    - skipped due to errors,
    - features retained.

## High-level design

1. **Configuration surface**
   - Add a small, opt-in feature-filter config:
     - Builder: `ExplainerBuilder.perf_feature_filter(enabled: bool, *, per_instance_top_k: int = 8)` (extensible later with additional knobs).
     - Env var: `CE_FEATURE_FILTER="enable,top_k=8"` parsed similarly to `CE_PARALLEL` / `CE_CACHE`.
   - Expose this as an internal config object attached to the explainer/plugin context (e.g. `explainer._feature_filter_config`), not as a public API.

2. **New internal module for filtering**
   - Introduce `calibrated_explanations.core.explain._feature_filter` with helpers that:
     - Accept a FAST `CalibratedExplanations` instance and a `per_instance_top_k` value.
     - For each instance `i`, compute per-feature importance (e.g. `importance[i, f] = abs(weight_predict[i, f])` or a small extension that also considers `low` / `high` weights) and stack these into a `(n_instances, n_features)` matrix.
     - Aggregate per-feature importance across instances (e.g. `max_i importance[i, f]`) to obtain a *batch-local* global score per feature.
     - Select the global top-`k` features by this aggregated score (excluding any already-ignored features) so that the number of non-ignored features for the batch is always `<= k`.
     - Given an existing `features_to_ignore` (constants + user-specified), derive a new `features_to_ignore_filtered = complement(top_k_features) ∪ existing_ignore`.
   - This module remains independent of `CalibratedExplainer`; it works only with explanation objects and indices.

3. **Plugin-layer integration (execution wrappers)**
   - Extend `_ExecutionExplanationPluginBase.explain_batch` in `plugins/builtins.py` for modes `"factual"` and `"alternative"`:
     - Detect whether feature filtering is enabled via the attached config.
     - Compute a base `features_to_ignore_base` from `request.features_to_ignore` and `explainer.features_to_ignore`.
     - Inside the existing executor context (`with explainer._perf_parallel or nullcontext():` in the caller), perform an internal FAST pass on the current batch:
       - Call `explainer._explanation_orchestrator.invoke("fast", x, threshold=request.threshold, low_high_percentiles=request.low_high_percentiles or (5, 95), bins=request.bins, features_to_ignore=features_to_ignore_base, extras={"mode": "fast", "invoked_by": "feature_filter"})`.
     - Pass the resulting FAST explanations to `_feature_filter` to obtain `features_to_ignore_filtered`.
     - Construct a *filtered* `ExplanationRequest` with `features_to_ignore=features_to_ignore_filtered` and the same `threshold`, `low_high_percentiles`, `bins`, `extras`.
     - Call `build_explain_execution_plan(self._explainer, x, filtered_request)` and dispatch to the configured execution plugin (sequential / feature-parallel / instance-parallel) as today.
   - Ensure that this path is only used for `"factual"` and `"alternative"`; `"fast"` remains unchanged.

4. **Executor reuse and lifetime**
   - The public methods already wrap explanation calls in the parallel context:
     - `ctx = self._perf_parallel or contextlib.nullcontext(); with ctx: self._explanation_orchestrator.invoke_factual(...)`.
   - Because `_ExecutionExplanationPluginBase.explain_batch` is invoked *inside* this `with ctx:` block:
     - The internal FAST pass (`invoke("fast", ...)`) and the subsequent factual/alternative execution share the same `ParallelExecutor` instance.
     - No additional pool is created for the filter; this respects the process-backend guidance (single entry, reuse pool).

5. **Error handling and fallbacks**
   - Wrap the internal FAST call in `try/except`:
     - Catch `ConfigurationError` and generic `Exception` (ensure ADR-002 compliance).
     - On error, log a warning (e.g. “FAST filter disabled for this run: …”) and proceed with the original `ExplanationRequest` (no feature filtering).
   - Optionally memoise a “FAST unavailable” flag on the plugin context to avoid retrying on every call in the same process.

6. **Testing strategy**
   - **Unit tests (plugins / filter module)**
     - For a toy model and small dataset:
       - Enable filtering with a small `per_instance_top_k` and assert that:
         - `ExplainRequest.features_to_ignore` passed to the execution plugin matches the complement of the union of per-instance FAST top-`k` features plus constants.
         - The final factual/alternative explanations contain weights only for the kept features (other columns zero or absent).
     - Simulate missing FAST plugin (deny or unregister) and ensure that factual/alternative runs still succeed with the unfiltered feature set.
   - **Integration tests (parallel backends)**
     - Run small end-to-end tests under thread and process backends, checking:
       - The same `ParallelExecutor` metrics (`submitted`, `completed`) increase across both the FAST and factual/alternative stages.
       - No pool re-initialisation occurs between the two stages.

7. **Documentation updates**
   - Add a short section to `docs/foundations/how-to/tune_runtime_performance.md` describing:
     - The new feature-filter toggle and its intended use (large-`p` settings where many features are uninformative).
     - How it interacts with `CE_PARALLEL` and the FAST plugin (internal use, no change to user-facing FAST API).

