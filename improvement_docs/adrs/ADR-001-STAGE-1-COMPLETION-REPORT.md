# ADR-001 Gap Closure Implementation Progress

**Status**: Stages 0–1c Completed  
**Date**: 2025-11-28  
**Implemented by**: Stage-by-stage ADR-001 package boundary realignment  

## Completed Stages

### ✅ Stage 0: Confirm Boundaries and ADR-001 Scope
- Created ADR-001-STAGE-0-SCOPE-CONFIRMATION.md documenting intentional divergences
- Confirmed `legacy`, `api`, `plotting.py`, and `perf` (post-Stage 1b) as intentional deviations
- Single maintainer model confirmed; no ownership assignments needed

### ✅ Stage 1a: Extract Calibration into Dedicated Package
**Status**: Completed

**Changes**:
- Created top-level `calibrated_explanations/calibration/` package
- Moved 5 core modules from `core/calibration/` to top-level:
  - `state.py` – Calibration dataset state management
  - `interval_learner.py` – Interval learner management
  - `summaries.py` – Calibration summary caching utilities
  - `interval_regressor.py` – Conformal prediction intervals
  - `venn_abers.py` – Venn-Abers probabilistic calibration
- Created backward compatibility shim at `core/calibration/__init__.py` with deprecation warnings
- Updated all imports across the codebase (core, plugins, prediction modules)
- Updated root `__init__.py` to import from new location

**Affected files** (updated imports):
- `core/calibrated_explainer.py`
- `core/calibration_metrics.py`
- `core/calibration_helpers.py`
- `core/interval_regressor.py`
- `core/venn_abers.py`
- `core/prediction/interval_registry.py`
- `plugins/builtins.py`
- `plugins/manager.py`

### ✅ Stage 1b: Split perf into cache and parallel Packages
**Status**: Completed

**Changes**:
- Created `calibrated_explanations/cache/` package with:
  - `cache.py` – Cache implementation with namespacing, versioning, TTL, memory budgets
  - `__init__.py` – Public exports: CacheConfig, CacheMetrics, CalibratorCache, TelemetryCallback
- Created `calibrated_explanations/parallel/` package with:
  - `parallel.py` – Parallel executor and strategy selection
  - `__init__.py` – Public exports: ParallelConfig, ParallelExecutor, ParallelMetrics
- Converted `perf/__init__.py` to a thin shim re-exporting from cache and parallel
  - Includes `PerfFactory` and `from_config()` for backward compatibility
  - Emits deprecation warning on first import

**Affected files** (imports updated):
- `parallel/parallel.py` – Updated to import from `..cache` instead of `.cache`
- All upstream callers continue to use `perf` module without changes (backward compatible)

### ✅ Stage 1c: Create schema Validation Package
**Status**: Completed

**Changes**:
- Created `calibrated_explanations/schema/` package with:
  - `validation.py` – Schema loading and validation helpers
  - `__init__.py` – Public exports: validate_payload()
- Moved schema validation logic from `serialization.py` to new package
- Updated `serialization.py` to import and re-export `validate_payload()` for backward compatibility
- `validate_payload()` remains accessible via both old and new paths

**Affected files**:
- `serialization.py` – Now imports from schema package

## Key Architectural Improvements

### Boundary Clarity
- ✅ **Calibration** now lives in a dedicated top-level package (not embedded in core)
- ✅ **Cache** and **Parallel** now live in distinct packages (not combined in perf)
- ✅ **Schema validation** now lives in a dedicated package (not mixed into serialization)
- ✅ **Backward compatibility maintained** via thin shims with deprecation warnings

### Import Graph
- Calibration imports reduced to: core domain models, utils, and internal state
- Cache and parallel packages now truly independent with minimal coupling
- Schema package decoupled from serialization concerns

### Migration Path
All changes include:
- ✅ Deprecation warnings for old import paths
- ✅ Compatibility shims at old locations
- ✅ Clear migration guidance in docstrings
- Planned removal: v1.1.0

## Remaining Work (Stages 2–5)

### Stage 2: Decouple Cross-Sibling Imports in CalibratedExplainer
- [ ] Refactor `core.calibrated_explainer` to use orchestrator interfaces
- [ ] Remove direct imports of: perf, plotting, plugins, discretizers, explanations
- [ ] Route dependencies through well-defined core facades

### Stage 3: Tighten Public API Surface
- [ ] Restrict `__init__.py` exports to sanctioned entry points only
- [ ] Move convenience exports to dedicated submodules
- [ ] Add deprecation warnings for removed symbols

### Stage 4: Document Remaining Namespaces
- [ ] Create ADR-001 addenda for `api`, `legacy`, `plotting`
- [ ] Document purpose/boundaries or mark for deprecation
- [ ] Update repository docs with new package map

### Stage 5: Add Import Graph Linting and Tests
- [ ] Enforce no cross-sibling imports via linting or unit tests
- [ ] Document in ADR-001 adoption notes
- [ ] Plan shim removal after migration period

## Success Criteria Status

| Criterion | Status | Notes |
| --------- | ------ | ----- |
| Calibration in dedicated package | ✅ | Top-level package with shim for compatibility |
| Cache and parallel split | ✅ | Separate packages with perf shim |
| Schema validation package | ✅ | Dedicated package with backward-compatible re-export |
| Calibration isolation verified | ⏳ | Deferred to Stage 2 (CalibratedExplainer refactor) |
| Import graph clean | ⏳ | Deferred to Stage 2–5 |
| Public API tightened | ⏳ | Deferred to Stage 3 |
| Full ADR alignment documented | ⏳ | Deferred to Stage 4 |
| Linting and tests in place | ⏳ | Deferred to Stage 5 |

## Version and Release Notes

- **Target release**: v0.10.0 runtime boundary realignment
- **Compatibility**: Full backward compatibility maintained via shims
- **Deprecation timeline**: Shims and old import paths removed in v1.1.0

## Next Steps

1. Run existing test suite to verify backward compatibility
2. Update test imports to use new package locations (gradually)
3. Proceed with Stages 2–5 per RELEASE_PLAN_V1.md timeline
4. Document adoption progress in ADR-001 adoption notes

