# CalibratedExplainer Streamlining Plan

**Status:** Phase 1b completed; Phase 2 completed; moved to v0.10.0 development cycle  
**Current metrics:** Reduced from 3947 to 3590 lines (357 lines removed via Phase 1a+1b)
**Next step:** Phase 3 (Plugin Management)
**Goal:** Thin, delegating facade with clear separation of concerns  
**ADR Alignment:** ADR-001 (boundary realignment), ADR-004 (parallel execution), ADR-005 (schema/envelope)

**Implementation Policy** 
- Refactor `CalibratedExplainer` into a thin orchestrator delegating to specialized modules.
- Relocate unrelated helpers and classes to appropriate modules.
- Refactor explanation logic into an `ExplanationOrchestrator` class.
- Refactor functional domains (explain, predict, calibrate, plugins) into separate packages with clear contracts.
- Never keep functionality in calibrated_explainer.py that can logically belong elsewhere.
- Never keep functionality in calibrated_explaominer.py just because tests are referencing it, change the tests
- Tests should focus on behaviour, not implementation details. 
- Always move code incrementally with deprecation notices to maintain backward compatibility over v0.10.0–v0.11.0.

---

## Executive Summary

The `CalibratedExplainer` class has accumulated substantial technical debt:

- **Bloated scope:** Configuration, plugin discovery, interval calibration, explanation orchestration, prediction, calibration, fast explanations, LIME/SHAP integration, plotting, discretization, reject learning—all in one file.
- **Mixed abstraction levels:** Helper functions, private classes, public API methods, and internal implementation details interleaved without clear boundaries.
- **Maintenance burden:** New features must navigate ~4000 lines to find the right injection point.

**Target state:**

- **CalibratedExplainer** becomes a thin, delegating orchestrator (~500–600 lines) with only public API methods and high-level coordination logic.
- **Peripheral code** (helpers, monitors, configuration utilities) relocated to specialized modules.
- **Functional domains** refactored into separate packages (explain, prediction, calibration, plugins) with clear, testable contracts.
- **Lifecycle:** Move functions incrementally over v0.10.0–v0.11.0 to maintain backward compatibility; mark internal APIs `@_internal` and document deprecation.

---

## Phase 0: Identify & Relocate Unrelated Functions

**Rationale:** Functions not directly tied to class behavior or explanation logic belong elsewhere.

### 0a. Spurious Configuration Helpers → `core/config_helpers.py` (new)

These functions handle external configuration (pyproject.toml, environment parsing) not directly tied to orchestration:

| Function | Target Module | Rationale | Scope |
|----------|---------------|-----------|-------|
| `_read_pyproject_section` | `core/config_helpers.py` | TOML parsing utility unrelated to explanations | Standalone utility |
| `_split_csv` | `core/config_helpers.py` | Environment variable parsing | Standalone utility |
| `_coerce_string_tuple` | `core/config_helpers.py` | Type coercion for config values | Standalone utility |

**Action items:**

1. Create `src/calibrated_explanations/core/config_helpers.py`.
2. Move the three functions with docstrings and tests intact.
3. Update imports in `calibrated_explainer.py` to reference the new module.
4. Mark original locations with deprecation notices (v0.10.0–v0.11.0).

**Testing:** Existing unit tests in `tests/unit/core/` remain valid; add imports validation in CI.

---

### 0b. Explanation Feature Tasks → `explain/feature_task.py` (new)

These helpers encode domain-specific perturbation and feature aggregation logic that belongs with the explanation subsystem:

| Function | Target Module | Rationale | Scope |
|----------|---------------|-----------|-------|
| `_assign_weight_scalar` | `explain/feature_task.py` | Scalar weight computation for feature deltas | Explanation computation |
| `_feature_task` | `explain/feature_task.py` | Core per-feature aggregation loop (400+ lines) | Parallel execution kernel |

**Action items:**

1. Create `src/calibrated_explanations/core/explain/feature_task.py`.
2. Move function bodies and type hints.
3. Extract the `FeatureTaskResult` type alias and any shared constants.
4. Update imports in `calibrated_explainer.py` and existing `explain/` submodules.
5. Integrate with `parallel_instance.py` and `parallel_feature.py` task dispatching.

**Testing:** Reuse existing `tests/unit/core/explain/` structure. Add integration tests for task result serialization.

---

### 0c. Prediction Bridge Monitoring → `plugins/predict_monitor.py` (new)

The `_PredictBridgeMonitor` class is a cross-cutting concern for plugin instrumentation; it belongs with the plugin infrastructure:

| Class | Target Module | Rationale | Scope |
|-------|---------------|-----------|-------|
| `_PredictBridgeMonitor` | `plugins/predict_monitor.py` | Plugin instrumentation & telemetry | Plugin framework |

**Action items:**

1. Create `src/calibrated_explanations/plugins/predict_monitor.py`.
2. Move class and docstrings.
3. Update `calibrated_explainer.py` to import from new home.
4. Export via `plugins/__init__.py` for internal use.

**Testing:** Add unit tests in `tests/unit/plugins/test_predict_monitor.py`.

---

## Phase 1: Delegate Explanation Orchestration

**Rationale:** Explanation logic is fragmented across the class. Centralizing it in an `ExplanationOrchestrator` makes room for future explanation plugins and separates concerns.

### 1a. Create `ExplanationOrchestrator` in `explain/orchestrator.py`

This class handles explanation workflow: plugin resolution, context building, and result formatting.

**Responsibilities:**

- Plugin resolution (`_resolve_explanation_plugin`, `_ensure_explanation_plugin`)
- Plugin instantiation (`_instantiate_plugin`, `_check_explanation_runtime_metadata`)
- Context building (`_build_explanation_context`)
- Result telemetry (`_build_instance_telemetry_payload`)
- Mode inference (`_infer_explanation_mode`)
- Plugin invocation (`_invoke_explanation_plugin`)
- Explanation chain building (`_build_explanation_chain`)

**Signature sketch:**

```python
class ExplanationOrchestrator:
    """Orchestrate explanation pipeline execution and plugin coordination."""
    
    def __init__(self, explainer: "CalibratedExplainer", config: ExplanationConfig) -> None:
        """Initialize with back-reference to parent explainer."""
        ...
    
    def resolve_plugin(self, mode: str) -> Tuple[Any, str | None]:
        """Find and instantiate the active explanation plugin."""
        ...
    
    def build_context(self, x: Any, mode: str, ...) -> ExplanationContext:
        """Assemble the context for plugin execution."""
        ...
    
    def invoke(self, x: Any, mode: str, ...) -> CalibratedExplanations:
        """Execute the full explanation pipeline."""
        ...
```

**Methods to move:**

- `_resolve_explanation_plugin`
- `_ensure_explanation_plugin`
- `_build_explanation_chain`
- `_check_explanation_runtime_metadata`
- `_instantiate_plugin`
- `_build_explanation_context`
- `_derive_plot_chain`
- `_build_instance_telemetry_payload`
- `_infer_explanation_mode`
- `_invoke_explanation_plugin`

**Delegating methods in `CalibratedExplainer`:**

```python
def explain_factual(self, x, ...):
    """Explain using factual rules (delegates to orchestrator)."""
    return self._explanation_orchestrator.invoke(x, mode="factual", ...)

def explain_counterfactual(self, x, ...):
    """Explain using counterfactuals (delegates to orchestrator)."""
    return self._explanation_orchestrator.invoke(x, mode="alternative", ...)

def explore_alternatives(self, x, ...):
    """Explore alternative scenarios (delegates to orchestrator)."""
    return self._explanation_orchestrator.invoke(x, mode="alternative", ...)
```

**Action items:**

1. Create `src/calibrated_explanations/core/explain/orchestrator.py`.
2. Extract orchestrator logic into new class.
3. Add `_explanation_orchestrator` field to `CalibratedExplainer.__init__`.
4. Replace moved methods with thin delegating wrappers.
5. Update imports and expose via `explain/__init__.py`.

**Testing:**

- Extract existing explanation method tests into `tests/unit/core/explain/test_orchestrator.py`.
- Add integration tests for plugin resolution fallback chains.

---

### 1b. Create `PredictionOrchestrator` in `prediction/orchestrator.py`

**Responsibilities:**

- Prediction bridge management and monitoring
- Interval calibration oversight
- Difficulty estimation coordination
- Prediction caching and performance tracking

**Methods to move:**

- `_predict_impl` (large prediction logic)
- `_predict` (thin wrapper)
- `_compute_weight_delta`
- `_ensure_interval_runtime_state`
- `_gather_interval_hints`
- `_check_interval_runtime_metadata`
- `_resolve_interval_plugin`
- `_build_interval_context`
- `_build_interval_chain`
- `_obtain_interval_calibrator`
- `_capture_interval_calibrators`

**Delegating methods in `CalibratedExplainer`:**

```python
def predict(self, x, uq_interval=False, calibrated=True, **kwargs):
    """Predict with optional uncertainty quantification."""
    return self._prediction_orchestrator.predict(x, uq_interval=uq_interval, ...)

def predict_proba(self, x, uq_interval=False, calibrated=True, **kwargs):
    """Probabilistic predictions with optional UQ."""
    return self._prediction_orchestrator.predict_proba(x, uq_interval=uq_interval, ...)
```

**Action items:**
1. Create `src/calibrated_explanations/core/prediction/` package.
2. Create `prediction/orchestrator.py` with extracted methods.
3. Add `_prediction_orchestrator` field to `CalibratedExplainer.__init__`.
4. Replace moved methods with thin delegating wrappers.
5. Update imports.

**Testing:** Extract existing prediction tests into `tests/unit/core/prediction/`.

---

## Phase 2: Delegate Fast Explanation Pipeline

**Rationale:** Fast explanations are a distinct explanation mode with specialized logic (perturbation, noise injection, discretization).

### 2a. Create `FastExplanationPipeline` in `external_plugins/fast_explanations/pipeline.py`

Fast explanations use a specialized orchestration path:

**Responsibilities:**

- Fast mode initialization
- Feature perturbation
- Noise injection and scaling
- Parallel feature computation
- Rule extraction and binning

**Methods to move:**

- `explain_fast` (entire method, ~200 lines)
- `_discretize`
- `_preprocess` (discretization prep)
- Rule boundary detection helpers

**Action items:**

1. Extend `external_plugins/fast_explanations/` structure with `pipeline.py`.
2. Extract `explain_fast` and related helpers.
3. Create factory method in `CalibratedExplainer`:

   ```python
   def explain_fast(self, x, ...):
       """Fast explanations (delegates to external plugin)."""
       from ..external_plugins.fast_explanations import FastExplanationPipeline
       pipeline = FastExplanationPipeline(self)
       return pipeline.explain(x, ...)
   ```

4. Keep only the public method stub in `CalibratedExplainer`.

**Testing:**

Move fast explanation tests to `tests/unit/external_plugins/fast_explanations/`.

---

### 2b. Create `IntegrationPipeline` for LIME/SHAP in `external_plugins/integrations/`

**Responsibilities:**

- LIME integration orchestration
- SHAP integration orchestration
- Helper preloading and lazy initialization

**Methods to move:**

- `explain_lime` (entire method)
- `_is_lime_enabled`, `_preload_lime`
- `explain_shap` (if separate)
- `_is_shap_enabled`, `_preload_shap`

**Action items:**

1. Create `external_plugins/integrations/lime_pipeline.py` and `shap_pipeline.py`.
2. Extract methods.
3. Thin delegating stubs in `CalibratedExplainer`:

   ```python
   def explain_lime(self, x, ...):
       """LIME explanations (delegates to external plugin)."""
       from ..external_plugins.integrations import LimePipeline
       return LimePipeline(self).explain(x, ...)
   ```

4. Ensure LIME/SHAP helpers remain accessible for backward compatibility.

**Testing:**

Move integration tests to `tests/unit/external_plugins/integrations/`.

---

## Phase 3: Delegate Plugin Management

**Rationale:** Plugin discovery, resolution, and registry management is a cross-cutting concern that should live with plugin infrastructure.

### 3a. Create `PluginManager` in `plugins/manager.py`

**Responsibilities:**

- Plugin override configuration
- Plugin descriptor lookup
- Plugin resolution with fallback chains
- Plugin instance caching

**Methods to move:**

- `_coerce_plugin_override`
- Registry state initialization (plugin cache fields)

**Instance attributes to extract:**

- `_explanation_plugin_overrides`
- `_interval_plugin_override`
- `_fast_interval_plugin_override`
- `_plot_style_override`
- `_bridge_monitors`
- `_explanation_plugin_instances`
- `_explanation_plugin_identifiers`
- `_explanation_plugin_fallbacks`
- `_plot_plugin_fallbacks`
- `_interval_plugin_hints`
- `_interval_plugin_fallbacks`
- `_interval_plugin_identifiers`
- `_pyproject_explanations`, `_pyproject_intervals`, `_pyproject_plots`

**Action items:**

1. Create `src/calibrated_explanations/plugins/manager.py`.
2. Extract plugin state and resolution logic into `PluginManager` class.
3. Add `_plugin_manager` field to `CalibratedExplainer.__init__`.
4. Replace plugin-related methods with delegating wrappers:

   ```python
   def _coerce_plugin_override(self, override):
       return self._plugin_manager.coerce_override(override)
   ```

5. Update exports via `plugins/__init__.py`.

**Testing:**

Add unit tests in `tests/unit/plugins/test_manager.py`.

---

## Phase 4: Delegate Prediction Functionality

**Rationale:** Prediction is separate from explanation; it should be cleanly partitioned.

### 4a. Refactor Prediction into `prediction/` Subpackage

**Current state:** Prediction logic embedded in `CalibratedExplainer`.

**Target structure:**

```
core/
  prediction/
    __init__.py
    orchestrator.py      # PredictionOrchestrator (moved from Phase 1b)
    interval_registry.py # Interval calibrator management
    cache.py            # Prediction result caching
```

**Methods to move:**

- (From Phase 1b: orchestration methods)
- `_get_sigma_test` → `prediction/interval_registry.py`
- `__constant_sigma` → `prediction/interval_registry.py`
- `__update_interval_learner` → `prediction/interval_registry.py`
- `__initialize_interval_learner` → `prediction/interval_registry.py`
- `__initialize_interval_learner_for_fast_explainer` → `prediction/interval_registry.py`

**Delegating stub in `CalibratedExplainer`:**

```python
@property
def interval_learner(self):
    """Access interval calibrator (delegated)."""
    return self._prediction_orchestrator.interval_learner

@interval_learner.setter
def interval_learner(self, value):
    """Set interval calibrator (delegated)."""
    self._prediction_orchestrator.interval_learner = value
```

**Action items:**

1. Create `src/calibrated_explanations/core/prediction/` package structure.
2. Extract interval management into `interval_registry.py`.
3. Update `calibrated_explainer.py` to use `_prediction_orchestrator`.
4. Ensure backward-compatible property access.

**Testing:**

Extract tests from `tests/unit/core/` into `tests/unit/core/prediction/`.

---

## Phase 5: Refactor Explain Functionality

**Rationale:** The `explain/` subpackage exists but logic is scattered across `calibrated_explainer.py`.

### 5a. Consolidate Explain Logic in `explain/` Subpackage

**Current state:** The `explain/` subpackage exists but logic is scattered across `calibrated_explainer.py`.

**Target structure:**

```
core/explain/
  __init__.py
  orchestrator.py       # ExplanationOrchestrator (moved from Phase 1a)
  feature_task.py       # Feature computation kernels (moved from Phase 0b)
  sequential.py         # Sequential explanation
  parallel_instance.py   # Parallel instance computation
  parallel_feature.py    # Parallel feature computation
  fast.py               # Fast explanation logic (partial, rest in external_plugins)
  _base.py
  _computation.py
  _helpers.py
  _shared.py
```

**Methods to move:**

- (From Phase 1a: orchestration methods)
- (From Phase 0b: feature task functions)
- `_validate_and_prepare_input` → `_helpers.py`
- `_initialize_explanation` → `_computation.py`
- `_explain_predict_step` → `_computation.py`
- `_assign_weight` → `feature_task.py`

**Delegating stubs in `CalibratedExplainer`:**

```python
def explain(self, x, ...):
    """Unified explain interface (delegates to orchestrator)."""
    return self._explanation_orchestrator.invoke(x, ...)

def explain_factual(self, x, ...):
    """Factual explanations (delegates to orchestrator)."""
    return self._explanation_orchestrator.invoke(x, mode="factual", ...)
```

**Action items:**

1. Reorganize `explain/` subpackage structure.
2. Move methods and helper functions.
3. Update internal imports within `explain/`.
4. Ensure backward compatibility for public APIs.

**Testing:**

Consolidate tests under `tests/unit/core/explain/`.

---

## Phase 6: Refactor Calibration Functionality

**Rationale:** Calibration is a distinct concern from explanation; it should have its own module (per ADR-001).

### 6a. Extract Calibration into `calibration/` Subpackage

**Note:** This is a longer-term refactor aligning with ADR-001 gap closure. For v0.10.0, only extract the most independent parts.

**Methods to consider for extraction:**

- `_get_calibration_summaries` → `calibration/summaries.py`
- `_invalidate_calibration_summaries` → `calibration/cache.py`
- Calibration data setters/getters (x_cal, y_cal, append_cal)

**Target state (v0.11.0+):**

```
calibration/
  __init__.py
  summaries.py        # Calibration summary computation & caching
  venn_abers.py       # Moved from core
  interval_learner.py # Interval calibrator abstractions
```

**Action items:**

1. Create `src/calibrated_explanations/calibration/` package (deferred to v0.11.0).
2. Plan extraction of calibration state management in ADR-001 realignment.

**Testing:**

Backlog for v0.11.0 phase.

---

## Phase 7: Refactor Discretization Functionality

**Rationale:** Discretization is orthogonal to explanation; it should be managed independently.

### 7a. Extract Discretization into `discretization/` Subpackage (deferred)

**Current state:** Discretizer instances and setup logic in `CalibratedExplainer`.

**Methods to move:**

- `set_discretizer` → `discretization/manager.py`
- `_discretize` → `discretization/pipeline.py`

**Target state (v0.11.0+):**

```
core/discretization/
  __init__.py
  manager.py        # Discretizer lifecycle
  pipeline.py       # Discretization execution
```

**Action items:**

1. Defer to v0.11.0 phase.
2. Document in ADR-009 (preprocessing pipeline).

---

## Phase 8: Refactor Misc Functionality

**Rationale:** Remaining methods that don't fit primary domains.

### 8a. Create `Utilities` Module for Property/State Management

**Methods to delegate:**
- `__repr__` (stays in `CalibratedExplainer`)
- `set_seed` → `core/utilities.py`
- `reinitialize` → Can stay (lifecycle method, high-level)
- `runtime_telemetry` → `core/telemetry.py`
- Property accessors (feature_names, num_features) → Keep as thin stubs

**Action items:**

1. Create `src/calibrated_explanations/core/utilities.py`.
2. Move seed management and utility functions.
3. Create `src/calibrated_explanations/core/telemetry.py` for runtime observability.

**Testing:**

Add unit tests in `tests/unit/core/`.

---

## Refactored Class Structure

### Target `CalibratedExplainer` (~500–600 lines)

```python
class CalibratedExplainer:
    """Thin orchestrating facade for calibrated explanations and predictions."""
    
    # ===== Lifecycle =====
    def __init__(self, learner, x_cal, y_cal, ...):
        """Initialize with calibration data and configuration."""
        # 1. Learner setup
        # 2. Calibration data setup
        # 3. Feature metadata setup
        # 4. Initialize orchestrators
        #    - _explanation_orchestrator = ExplanationOrchestrator(self, ...)
        #    - _prediction_orchestrator = PredictionOrchestrator(self, ...)
        #    - _plugin_manager = PluginManager(self, ...)
        # 5. Initialize helpers (LIME, SHAP)
        # 6. Mark as fitted
    
    def reinitialize(self, ...):
        """Reinitialize with new learner/calibration data."""
        ...
    
    def __repr__(self):
        """String representation."""
        ...
    
    # ===== Public Prediction API =====
    def predict(self, x, uq_interval=False, calibrated=True, **kwargs):
        """Predict with optional uncertainty quantification."""
        return self._prediction_orchestrator.predict(x, ...)
    
    def predict_proba(self, x, uq_interval=False, calibrated=True, **kwargs):
        """Probabilistic predictions with optional UQ."""
        return self._prediction_orchestrator.predict_proba(x, ...)
    
    def predict_reject(self, x, bins=None, confidence=0.95):
        """Predict with rejection option."""
        ...  # Delegate to rejection module
    
    def predict_calibration(self):
        """Return calibration metrics."""
        ...
    
    def calibrated_confusion_matrix(self):
        """Compute calibrated confusion matrix."""
        ...
    
    # ===== Public Explanation API =====
    def explain_factual(self, x, ...):
        """Explain using factual rules."""
        return self._explanation_orchestrator.invoke(x, mode="factual", ...)
    
    def explain_counterfactual(self, x, ...):
        """Explain using counterfactuals."""
        return self._explanation_orchestrator.invoke(x, mode="alternative", ...)
    
    def explore_alternatives(self, x, ...):
        """Explore alternative scenarios."""
        return self._explanation_orchestrator.invoke(x, mode="alternative", ...)
    
    def explain_fast(self, x, ...):
        """Fast explanations (delegates to external plugin)."""
        from ..external_plugins.fast_explanations import FastExplanationPipeline
        return FastExplanationPipeline(self).explain(x, ...)
    
    def explain_lime(self, x, ...):
        """LIME explanations (delegates to external plugin)."""
        from ..external_plugins.integrations import LimePipeline
        return LimePipeline(self).explain(x, ...)
    
    def explain(self, x, ...):
        """Unified explain interface."""
        return self._explanation_orchestrator.invoke(x, ...)
    
    def __call__(self, x, ...):
        """Shorthand for explain."""
        return self.explain(x, ...)
    
    # ===== Configuration & Setup =====
    def set_seed(self, seed: int):
        """Set random seed."""
        ...
    
    def set_difficulty_estimator(self, estimator, initialize=True):
        """Set difficulty estimator (regression)."""
        ...
    
    def set_discretizer(self, discretizer, ...):
        """Set discretizer for binning."""
        ...
    
    def assign_threshold(self, threshold):
        """Assign classification threshold."""
        ...
    
    def initialize_reject_learner(self, ...):
        """Initialize rejection module."""
        ...
    
    def append_cal(self, x, y):
        """Append calibration data."""
        ...
    
    # ===== State & Introspection =====
    @property
    def x_cal(self):
        """Calibration features."""
        ...
    
    @x_cal.setter
    def x_cal(self, value):
        """Set calibration features."""
        ...
    
    @property
    def y_cal(self):
        """Calibration targets."""
        ...
    
    @y_cal.setter
    def y_cal(self, value):
        """Set calibration targets."""
        ...
    
    @property
    def num_features(self):
        """Number of features."""
        ...
    
    @property
    def feature_names(self):
        """Feature names."""
        ...
    
    @property
    def is_multiclass(self):
        """Is multiclass classification?"""
        ...
    
    @property
    def is_fast(self):
        """Is fast mode enabled?"""
        ...
    
    @property
    def interval_learner(self):
        """Interval calibrator (delegated to prediction orchestrator)."""
        return self._prediction_orchestrator.interval_learner
    
    @interval_learner.setter
    def interval_learner(self, value):
        """Set interval calibrator."""
        self._prediction_orchestrator.interval_learner = value
    
    @property
    def runtime_telemetry(self):
        """Runtime execution telemetry."""
        ...
    
    @property
    def preprocessor_metadata(self):
        """Preprocessor metadata."""
        ...
    
    def set_preprocessor_metadata(self, metadata):
        """Set preprocessor metadata."""
        ...
    
    # ===== Plotting (Thin Delegator) =====
    def plot(self, x, y=None, threshold=None, **kwargs):
        """Plot explanation (delegates to plotting module)."""
        from ..plotting import plot_explanation
        return plot_explanation(self, x, y, threshold, **kwargs)
```

---

## Migration & Backward Compatibility

### Deprecation Timeline

| Version | Action |
|---------|--------|
| **v0.10.0** | Mark internal methods with `@_internal` decorator. Redirect function imports to new modules with warnings. |
| **v0.10.0–v0.11.0** | Accept both old and new calling patterns. Log migration notices. |
| **v0.11.0** | Remove deprecated internal APIs. Archive old locations. |
| **v1.0.0** | Final removal of legacy compatibility shims. |

### Compatibility Layer

Create `src/calibrated_explanations/core/_compat.py` to manage deprecation:

```python
def _deprecated_import(old_name: str, new_module: str, new_name: str | None = None):
    """Issue deprecation warning and redirect import."""
    import warnings
    target_name = new_name or old_name
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v1.0.0. "
        f"Import from {new_module}.{target_name} instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Dynamic import and return
    ...
```

### Public API Preservation

Ensure all public methods remain in `CalibratedExplainer` as thin delegators:

```python
# ✅ This stays public and does not change:
explainer.explain_factual(x)
explainer.predict(x)
explainer.set_discretizer(...)

# ❌ These are internal and may move:
explainer._resolve_explanation_plugin(...)  # → ExplanationOrchestrator
explainer._build_explanation_context(...)   # → ExplanationOrchestrator
explainer._read_pyproject_section(...)      # → config_helpers
```

---

## Refactoring Execution Order

### v0.10.0 Release Window

**Priority 1 (Critical path to unlocking other work):**

1. Phase 0: Extract helper functions (config_helpers, feature_task)
2. Phase 3: Extract plugin management (`PluginManager`)
3. Phase 1a: Extract explanation orchestration (`ExplanationOrchestrator`)

**Priority 2 (Consolidation):**

4. Phase 5: Consolidate explain logic
5. Phase 1b: Extract prediction orchestration

**Priority 3 (Delegation & cleanup):**

6. Phase 2: Delegate fast/LIME/SHAP pipelines
7. Phase 8: Extract utility functions

### v0.10.1–v0.11.0 Release Window

**Deferred (lower priority):**

8. Phase 4: Full prediction subpackage refactoring
9. Phase 6: Calibration package extraction (ADR-001 alignment)
10. Phase 7: Discretization module extraction

---

## Testing Strategy

### Unit Test Reorganization

Current: `tests/unit/core/test_calibrated_explainer.py` (~2000+ lines)  
Target: Split into focused modules:

```
tests/unit/core/
  test_calibrated_explainer.py      # Thin facade tests (~200 lines)
  config_helpers/
    test_pyproject_parsing.py
    test_csv_splitting.py
  explain/
    test_orchestrator.py
    test_feature_task.py
    test_sequential.py
    test_parallel_instance.py
    test_parallel_feature.py
  prediction/
    test_orchestrator.py
    test_interval_registry.py
  plugins/
    test_manager.py
```

### Integration Tests

- `tests/integration/core/test_explain_workflow.py` – Full explanation pipeline
- `tests/integration/core/test_predict_workflow.py` – Full prediction pipeline
- `tests/integration/plugins/test_plugin_resolution.py` – Plugin discovery & resolution

### Regression Tests

- Snapshot tests for old vs. new orchestrator outputs
- Golden fixtures for backward compatibility
- Performance benchmarks (no regression > 5%)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| `CalibratedExplainer` line count | < 600 lines |
| `CalibratedExplainer` method count | < 35 methods |
| `CalibratedExplainer` responsibility domains | 3–4 (lifecycle, prediction, explanation, config) |
| Test coverage (orchestrator classes) | ≥ 85% |
| Backward compatibility | 100% (public API unchanged) |
| Performance overhead | < 2% (thin delegation) |
| Documentation clarity | All internal methods marked `@_internal` with docstrings |

---

## Appendix A: File Checklist

### Files to Create

- [ ] `src/calibrated_explanations/core/config_helpers.py`
- [ ] `src/calibrated_explanations/core/explain/feature_task.py`
- [ ] `src/calibrated_explanations/core/explain/orchestrator.py`
- [ ] `src/calibrated_explanations/core/prediction/orchestrator.py`
- [ ] `src/calibrated_explanations/core/prediction/interval_registry.py`
- [ ] `src/calibrated_explanations/plugins/predict_monitor.py`
- [ ] `src/calibrated_explanations/plugins/manager.py`
- [ ] `src/calibrated_explanations/core/telemetry.py`
- [ ] `src/calibrated_explanations/core/utilities.py`
- [ ] `external_plugins/fast_explanations/pipeline.py`
- [ ] `external_plugins/integrations/lime_pipeline.py`

### Files to Modify

- [ ] `src/calibrated_explanations/core/calibrated_explainer.py` (remove ~3400 lines)
- [ ] `src/calibrated_explanations/core/explain/__init__.py` (export orchestrator)
- [ ] `src/calibrated_explanations/plugins/__init__.py` (export manager, monitor)
- [ ] `src/calibrated_explanations/__init__.py` (update public API)
- [ ] `tests/unit/core/test_calibrated_explainer.py` (split into submodule tests)

### Files to Archive

- [ ] `improvement_docs/cleanup/PHASE_0_COMPLETION.md`
- [ ] `improvement_docs/cleanup/PHASE_1_COMPLETION.md`
- [ ] (one per phase)

---

## References

- [ADR-001: Package & Boundary Layout](../adrs/ADR-001.md)
- [ADR-004: Parallel Execution Framework](../adrs/ADR-004.md)
- [RELEASE_PLAN_v1.md](../RELEASE_PLAN_v1.md) – v0.10.0 runtime boundary realignment
- [Parallel Execution Improvement Plan](../parallel_execution_improvement_plan.md)

---

**Last updated:** 2025-11-13  
**Prepared by:** Copilot  
**Status:** Ready for review & prioritization
