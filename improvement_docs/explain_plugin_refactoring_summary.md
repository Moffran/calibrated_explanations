# Explain Plugin Architecture Refactoring Summary

## Overview

Successfully decomposed the monolithic `CalibratedExplainer.explain` method into a plugin-based architecture, as specified in the user's requirement. This refactoring addresses ADR-004 gaps and separates execution strategies cleanly.

## Changes Implemented

### 1. New Package Structure: `core/explain/`

Created a dedicated package for explain execution plugins:

```
src/calibrated_explanations/core/explain/
├── __init__.py           # Plugin registry and dispatcher
├── _base.py              # BaseExplainPlugin abstract class
├── _shared.py            # ExplainRequest, ExplainResponse, ExplainConfig dataclasses
├── _helpers.py           # Shared utility functions
├── sequential.py         # SequentialExplainPlugin
├── parallel_feature.py   # FeatureParallelExplainPlugin
└── parallel_instance.py  # InstanceParallelExplainPlugin
```

### 2. Plugin Implementations

#### SequentialExplainPlugin (Priority: 10)
- Single-threaded, feature-by-feature processing
- Universal fallback that always supports any request
- 300 lines, faithfully reproducing original sequential logic

#### FeatureParallelExplainPlugin (Priority: 20)
- Distributes feature tasks across executor workers
- Requires executor enabled with `granularity='feature'`
- 310 lines, reusing sequential setup with parallel dispatch

#### InstanceParallelExplainPlugin (Priority: 30)
- Partitions instances into chunks for parallel processing
- Each chunk delegates to SequentialExplainPlugin
- Requires executor enabled with `granularity='instance'`
- 200 lines, includes chunk combination logic

### 3. Shared Infrastructure

#### Data Structures (`_shared.py`)
- **ExplainRequest**: Immutable request context (x, threshold, bins, etc.)
- **ExplainResponse**: Mutable response payload for incremental assembly
- **ExplainConfig**: Configuration context (executor, granularity, feature metadata)

#### Helper Functions (`_helpers.py`)
- `slice_threshold()`, `slice_bins()`: Partition support for instance parallelism
- `merge_ignore_features()`: Combine explainer and request-level ignore sets
- `initialize_explanation()`, `explain_predict_step()`: Delegate to existing helpers
- Thin wrappers to preserve existing helper abstractions

#### Base Plugin (`_base.py`)
- Abstract `supports(request, config)` predicate for plugin selection
- Abstract `execute(request, config, explainer)` method for execution
- `name` and `priority` properties for registry management

### 4. CalibratedExplainer.explain Refactoring

**Before:** 256 lines with nested branching (sequential/instance-parallel/feature-parallel)

**After:** 13 lines delegating to plugin system

```python
def explain(self, x, threshold=None, low_high_percentiles=(5, 95),
            bins=None, features_to_ignore=None, *,
            _use_plugin: bool = True, _skip_instance_parallel: bool = False):
    if _use_plugin:
        # Existing plugin registry path (explanation plugins)
        mode = self._infer_explanation_mode()
        return self._invoke_explanation_plugin(...)

    # NEW: Delegate to explain plugin system
    from .explain import explain as plugin_explain
    return plugin_explain(self, x, threshold, low_high_percentiles,
                          bins, features_to_ignore,
                          _skip_instance_parallel=_skip_instance_parallel)
```

### 5. Plugin Selection Logic

The `select_plugin()` function in `core/explain/__init__.py`:
1. Validates configuration (rejects conflicting granularity)
2. Iterates plugins in priority order (30 → 20 → 10)
3. Selects first plugin where `supports(request, config)` returns `True`
4. Falls back to SequentialExplainPlugin (always supports)

### 6. Behavioral Equivalence Verification

✅ **All tests pass:**
- `test_explain_legacy_equivalence.py`: 5/5 tests pass
  - `test_classification_matches_legacy`
  - `test_regression_matches_legacy`
  - `test_legacy_explain_categorical_paths_and_ignore`
  - `test_legacy_explain_accepts_threshold_tuples_for_regression`
  - `test_legacy_explain_handles_continuous_bins_and_boundaries`

- `test_instance_parallel.py`: 1/1 tests pass
  - Confirms instance-parallel chunking produces identical results

## Benefits

### 1. Separation of Concerns
- Orchestration (plugin selection) separated from execution (plugin logic)
- Each plugin focuses on a single parallelism strategy
- Clear boundaries between sequential/feature/instance approaches

### 2. Maintainability
- Easier to modify individual strategies without affecting others
- Plugin interface enforces consistent contracts
- Shared utilities prevent code duplication

### 3. Testability
- Plugins can be unit-tested independently
- Mock executors can test parallel paths without actual concurrency
- Configuration guards prevent invalid combinations

### 4. Extensibility
- New execution strategies (e.g., distributed, GPU-accelerated) can be added as plugins
- Plugin priority system allows graceful fallback
- ADR-004 compliance enables future Ray/Dask integration

## Alignment with ADRs

### ADR-001: Core Decomposition Boundaries
✅ New `core/explain/` package aligns with modular boundary strategy
✅ Clear separation between calibration and explanation execution

### ADR-004: Parallel Execution Framework
✅ Pluggable executor facade enables strategy swapping
✅ Granularity-based selection (feature vs. instance)
✅ Fallback to serial execution when executor unavailable
✅ Telemetry hooks preserved (via explainer state updates)

## Migration Notes

### For Users
- **No breaking changes**: Public API unchanged
- `_use_plugin=True` still invokes existing explanation plugin registry
- `_use_plugin=False` now uses new explain plugin system (behavioral equivalent)

### For Developers
- Parallel execution strategies now live in `core/explain/` plugins
- To add new execution strategies: implement `BaseExplainPlugin` interface
- Helper methods remain in `core/prediction_helpers.py` and `core/calibrated_explainer.py`

## Performance Characteristics

- **Sequential**: No overhead, matches original implementation exactly
- **Feature-parallel**: Same dispatch cost as before, now isolated in dedicated plugin
- **Instance-parallel**: Chunk combination logic preserved, slightly cleaner code path

## Future Work

From the original task specification, remaining opportunities:

1. **Extract more shared utilities**: `_feature_task`, `_merge_feature_result`, and
   `_get_calibration_summaries` could move into a `core/explain/_computation.py` module
   to reduce explainer class coupling.

2. **Request/response validation**: Add runtime validation for ExplainRequest and
   ExplainResponse to catch configuration errors earlier.

3. **Plugin-specific tests**: While API-level tests pass, dedicated unit tests for
   each plugin (with mocked executors) would improve coverage and catch edge cases.

4. **Configuration object**: Replace scattered executor/granularity checks with a
   unified `ParallelConfig` object per ADR-004 specification.

## Conclusion

This refactoring successfully achieves the stated goals:
- ✅ All explain logic moved into plugins
- ✅ CalibratedExplainer.explain is now a thin delegator
- ✅ Sequential, feature-parallel, and instance-parallel strategies isolated
- ✅ Behavioral parity maintained (all tests pass)
- ✅ ADR-004 compliance for parallel execution framework
- ✅ Foundation for future distributed execution plugins

The code is cleaner, more maintainable, and ready for extension with minimal risk
to existing functionality.
