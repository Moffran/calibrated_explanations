# Anti-Pattern Remediation: API Changes Summary

This document summarizes the API changes, refactorings, and remediations performed as part of the anti-pattern remediation project (specifically Pattern 1: Internal Logic Testing and Pattern 3: Dead Code).

## 1. Public API Exposure (Pattern 1)

To reduce coupling between tests and internal implementation details, several private members were refactored into public APIs or properties.

### Core Refactorings
- **CalibratedExplainer**:
  - Refactored `_discretize` to public `discretize`.
  - Refactored `_is_mondrian` to public `is_mondrian` property.
  - Exposed public properties for orchestrators: `prediction_orchestrator`, `explanation_orchestrator`, `reject_orchestrator`.
  - Exposed execution state: `is_mondrian`, `is_multiclass`.
  - Added public API surface for plugin consumption: `predict_calibrated`, `discretize`, `preload_lime`, `preload_shap`, and `runtime_telemetry`.
  - Fixed `runtime_telemetry` to point to `self.plugin_manager.last_telemetry`.
  - Updated explain methods (`explain_factual`, `explore_alternatives`, `_explain`) to auto-set bins for Mondrian calibration when `bins` is None and `is_mondrian` is True.
- **Explanation / CalibratedExplanation**:
  - Refactored `_get_rules` to public `get_rules` accessor.
  - Refactored `_has_rules` and `_has_conjunctive_rules` to public properties.
  - Refactored `_to_python_number`, `_compute_confidence_level`, `_normalize_threshold_value`, and `_rank_features` to public APIs.
  - Refactored `_get_rules` and `_define_conditions` to public APIs.
  - Resolved recursion in `CalibratedExplanations.get_explainer` and `get_rules`.
  - Updated `FrozenCalibratedExplainer` and `CalibratedExplanations` to use public API surface.
- **WrapCalibratedExplainer**:
  - Added public `auto_encode` and `preprocessor` properties.
- **ExplanationOrchestrator**:
  - Refactored `_resolve_plugin` to public `resolve_plugin`.
  - Refactored `_check_metadata` to public `check_metadata`.
  - Refactored `_build_context` to public `build_context`.
  - Refactored `_interval_registry` to public `interval_registry`.
- **IntervalRegressor**:
  - Standardized to use `_y_cal_hat_storage` consistently.
  - Added compatibility for `crepes` 0.9.0 by handling `_y_cal_hat` vs `y_cal_hat` attribute changes.
- **FrozenCalibratedExplainer**:
  - Added `predict_calibrated` property to ensure API parity with `CalibratedExplainer`.
- **Integrations**:
  - Updated `LimePipeline` integration to use public `is_mondrian` and `predict_calibrated` APIs.
  - Made `_is_mondrian` public in `PredictionHelpers` protocol.

### Module-Specific Refactorings
- **Parallel Execution (`parallel.py`)**:
  - Refactored `_thread_strategy` to public `thread_strategy`.
  - Refactored `_joblib_strategy` to public `joblib_strategy`.
- **Caching (`cache.py`)**:
  - Refactored `_default_size_estimator` to public `default_size_estimator`.
  - Refactored `_cache` in `GlobalCacheManager` to public `cache`.
- **Explain Module (`feature_task.py`)**:
  - Refactored `_feature_task` to public `feature_task`.
- **Plotting (`legacy/plotting.py`)**:
  - Refactored `_compose_save_target` to public `compose_save_target`.
- **CLI (`cli.py`)**:
  - Replaced `_string_tuple` with `coerce_string_tuple` from `core.config_helpers`.
  - Updated `coerce_string_tuple` to be more robust.

## 2. Dead Code & Deprecation Removal (Pattern 3)

Legacy and redundant code was removed to improve maintainability.

- **Removed Deprecated Methods**:
  - `explain_counterfactual` (superseded by `explore_alternatives`).
  - LIME and SHAP integration helpers: `is_lime_enabled`, `is_shap_enabled`, `_preload_lime`, `_preload_shap`, `preload_shap`.
- **Removed Internal Helpers**:
  - `_build_explanation_chain`, `_build_interval_chain` (from `CalibratedExplainer`).
  - `_build_plot_chain` (from `ExplanationOrchestrator`).
  - `_define_conditions` alias from `Explanation`.
  - `_discretize` from `CalibratedExplanations`.
- **Orphaned Tests**:
  - Removed associated orphaned tests in `tests/unit/core/test_calibrated_explainer_lime.py`, `tests/unit/core/test_calibrated_explainer_runtime_helpers.py`, `tests/unit/core/test_wrap_explainer_helpers.py`, and `tests/unit/core/explain/test_explain_orchestrator.py`.

## 3. Infrastructure & Stability Improvements

- **Plugin Architecture Stabilization**:
  - Resolved 38 test failures following the migration to a plugin-based architecture.
  - Standardized test stubs (`DummyExplainer`, `StubExplainer`, `FakeExplainer`, `DummyOriginalExplainer`) to include `PluginManager` and required methods (`infer_explanation_mode`, `predict_calibrated`, `discretize`, `preload_lime`, `preload_shap`).
  - Synchronized `PredictBridgeMonitor` lookup in `ExplanationOrchestrator` to use `identifier or mode`.
  - Fixed attribute access in `_ExecutionExplanationPluginBase` to use `self._explainer`.
  - Resolved recursion error in `test_calibration_helpers.py` by ensuring `CalibratedExplainer` initialization is properly mocked.
  - Updated `DummyLearner` test helper to support configurable `num_classes` and fixed data shape mismatches.
- **Test Helpers & Patches**:
  - Fixed recursive method definitions in test helpers (`rank_features`, `infer_explanation_mode`).
  - Added missing attributes to dummy classes (`has_rules`, `has_conjunctive_rules`, `rules`, `conjunctive_rules`).
  - Updated test patches to use public methods instead of private ones.
- **Allow-List Management**:
  - Added 93 no-expiry allow-listed entries for acceptable test internals in `.github/private_member_allowlist.json`.
  - Updated `.github/private_member_allowlist.json` to remove remediated items.
  - Enforced version-bound expiry (v0.11.0) for temporary waivers.
- **Documentation**:
  - Updated `scripts/anti-pattern-analysis/README.md` and `docs/improvement/anti_pattern_gap_analysis.ipynb` with color-coded visualization of allow-listed usages.
