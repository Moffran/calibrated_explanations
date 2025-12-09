# ADR-001 Stage 0: Scope Confirmation & Intentional Divergences

**Status**: Decision note for gap closure implementation
**Date**: 2025-11-28
**Context**: Before executing the gap closure plan, confirm ADR-001 scope and document intentional divergences.

## Confirmed ADR-001 Scope

The core decomposition described in ADR-001 defines these top-level internal packages:

- `calibrated_explanations.core`: Shared domain models, strategy interfaces, orchestrators
- `calibrated_explanations.calibration`: Calibration algorithms & conformal predictors (currently embedded under `core/calibration`)
- `calibrated_explanations.explanations`: Explanation strategies
- `calibrated_explanations.cache`: Cache layer (currently under `perf/cache.py`)
- `calibrated_explanations.parallel`: Parallel facade (currently under `perf/parallel.py`)
- `calibrated_explanations.schema`: JSON schema definitions & validation helpers (currently in `serialization.py`)
- `calibrated_explanations.plugins`: Registry & loading
- `calibrated_explanations.viz`: Visualization abstractions
- `calibrated_explanations.utils`: Non-domain helpers

## Intentional Divergences (Approved Deviations from ADR-001)

### 1. `calibrated_explanations.legacy` — Intentional retention

**Rationale**: Contains compatibility shims for deprecated APIs and pre-refactor code paths. Marked for removal in v2.0.0.

**Scope**: Migration helpers, legacy parameter canonicalization, older explanation formats.

**ADR alignment**: Not a violation; classified as a deprecation boundary per ADR-011 (Migration Gates) once adopted.

### 2. `calibrated_explanations.api` — Internal param validation facade

**Rationale**: Provides param canonicalization and validation contract helpers used across the codebase. Acts as a shared contract layer between core and plugin consumers.

**Scope**: Parameter validation, kwargs canonicalization, aliasing resolution.

**ADR alignment**: Should be scoped as a utility sub-layer under core; moving to dedicated package is low-priority and will be deferred to v1.1 if needed.

### 3. `calibrated_explanations.plotting.py` — Deprecated convenience re-export

**Rationale**: Module-level convenience re-export of visualization functions. Currently wraps `viz` to support older import patterns.

**Scope**: Re-exports from `viz` submodule.

**ADR alignment**: Marked for removal in v0.11; all imports should route through `viz` submodule instead.

### 4. `calibrated_explanations.perf` — Temporary shim (post-Stage 1b)

**Rationale**: After Stage 1b splits `cache` and `parallel` into distinct packages, `perf` will remain as a thin shim re-exporting both for backward compatibility.

**Scope**: Re-exports `CalibratorCache`, `ParallelExecutor`, and factory helpers.

**ADR alignment**: Intentional compatibility shim; will be removed after one minor release with deprecation warnings.

### 5. `calibrated_explanations.integrations` — Scope clarified

**Rationale**: Hosts third-party integration helpers (LIME, SHAP, etc.) that augment core explanations.

**Scope**: Integration adapters, helper factories.

**ADR alignment**: Aligns with ADR-001 "utilities" guidance; no cross-talk with core.

## Single Maintainer Model

With one core developer, explicit ownership assignments are not necessary. All package boundaries remain the responsibility of the maintainer to uphold during reviews and refactors.

## Next Steps

This confirmation note is superseded by implementation of Stages 1–5. Any future deviations from this scope require a follow-up decision note and PR discussion.
