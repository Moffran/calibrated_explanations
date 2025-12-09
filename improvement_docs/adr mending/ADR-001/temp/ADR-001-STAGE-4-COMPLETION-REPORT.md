# ADR-001 Stage 4: Documentation of Remaining Namespaces

**Status**: ✅ COMPLETE (2025-11-28)
**Author**: ADR-001 Gap Closure Plan
**Scope**: Document intentional deviations and boundaries for `api`, `legacy`, `plotting`, `perf`, and `integrations` namespaces

## Executive Summary

Stage 4 documents the five top-level namespaces that deviate from or fall outside the core ADR-001 architecture but are retained for specific reasons. This report confirms their intended role, deprecation status, and migration paths—closing **ADR-001 Gap #6** ("Extra top-level namespaces lack ADR coverage").

**Status by Namespace:**
| Namespace | Role | Status | Removal Target |
|-----------|------|--------|-----------------|
| `api` | Parameter validation facade | ✅ Documented | v1.1+ (low priority) |
| `legacy` | Compatibility shims for deprecated APIs | ✅ Documented | v2.0.0 (marked for deprecation) |
| `plotting` | Deprecated convenience re-export module | ✅ Documented | v0.11.0 (deprecation → removal) |
| `perf` | Temporary cache/parallel shim | ✅ Documented | v0.10.1+ (deprecation → removal) |
| `integrations` | Third-party integration helpers | ✅ Documented | N/A (permanent) |

All documented in ADR-001 addenda and architectural guidelines.

---

## Gap #6: "Extra top-level namespaces lack ADR coverage"

**Severity**: 6 (medium) · **Impact**: 3 (architecture drift) × **Scope**: 2 (multiple modules)

**Original Problem**:
- Five top-level packages (`api`, `legacy`, `plotting`, `perf`, `integrations`) exist outside the core ADR-001 boundary definition.
- No documented rationale for their retention or transition strategy.
- Users and maintainers lack clarity on which are permanent vs. deprecated.

**Resolution Approach** (Stage 4):
1. Audit each namespace's current role and dependencies
2. Classify each as permanent, temporary (with deprecation timeline), or for removal
3. Document ADR alignment or justified deviation
4. Create migration paths for deprecated namespaces
5. Update architectural guidelines to clarify boundaries

---

## Documentation: Remaining Namespaces

### 1. `calibrated_explanations.api` — Parameter Validation Facade

**Current Role**: Shared parameter validation and canonicalization layer used across core, plugins, and wrappers.

**Contents**:
- `config.py`: Configuration & parameter store helpers
- `params.py`: Parameter validation, aliasing resolution, kwargs canonicalization
- `quick.py`: Quick-start API helpers

**Classification**: **Intentional; Low-Priority Relocation**

**Rationale**:
- Acts as a shared contract layer preventing circular imports between core and plugin consumers
- Too frequently used across the codebase to deprecate in v0.10
- Deferred relocation to utility sub-layer under `core` marked for v1.1+

**ADR-001 Alignment**:
- Not a violation per se; should be scoped as `calibrated_explanations.core.api` or `calibrated_explanations.utils.api` in long term
- Current separation intentional to break import cycles
- Will be consolidated during major refactor (v1.1 minimum)

**Migration Path**:
- v0.10–v1.0: No action; imports remain stable
- v1.1+: Internal reorganization; public API unchanged if re-exported from `__init__`

**Deprecation Status**: ⏸️ **None** (deferred to v1.1)

---

### 2. `calibrated_explanations.legacy` — Compatibility Shims

**Current Role**: Backward compatibility layer for deprecated APIs and pre-refactor code paths.

**Contents**:
- `__init__.py`: Legacy explanation format helpers
- `plotting.py`: Deprecated plotting convenience functions

**Classification**: **Temporary; Scheduled for v2.0.0 Removal**

**Rationale**:
- Contains migration helpers for users upgrading from pre-v0.9 code
- Marked explicitly for removal in v2.0.0 per CHANGELOG and migration guides
- Appropriate for organizations on extended support; not recommended for new code

**ADR-001 Alignment**:
- Classified as a deprecation boundary per ADR-011 (Migration Gates)
- Explicitly out-of-scope for core architecture but necessary for user migration
- Removal planned well in advance (minimum 1-2 major releases after v1.0.0)

**Migration Path**:
1. **v0.10–v1.0**: Users import from `legacy`; guidance redirects to modern imports
2. **v1.1–v2.0**: Deprecation warnings on `legacy` imports (ADR-011 pattern)
3. **v2.0.0**: Complete removal; users must migrate to direct submodule imports

**Deprecation Status**: ⏳ **Planned** (no warnings yet; will add v1.1+)

**Example Migration**:
```python
# OLD (deprecated, v0.10+)
from calibrated_explanations.legacy import LegacyExplanation
expl = LegacyExplanation(...)

# NEW (recommended)
from calibrated_explanations.explanations import FastExplanation
expl = FastExplanation(...)
```

---

### 3. `calibrated_explanations.plotting` — Deprecated Convenience Re-export

**Current Role**: Module-level convenience re-export wrapping visualization functions; supports older import patterns.

**Contents**:
- Re-exports from `viz` submodule
- Legacy plotting utilities

**Classification**: **Temporary; Scheduled for v0.11.0 Removal**

**Rationale**:
- Wrapper around `viz` for backward compatibility with pre-v0.10 code
- Now redundant after Stage 3 narrowing of public API surface
- Users should import directly from `viz` submodule

**ADR-001 Alignment**:
- Violates ADR-001's clear namespace separation principle
- Redundant wrapper; marked for immediate deprecation per v0.10.0 timeline

**Deprecation Strategy** (Two-Release Window):
- **v0.10.0**: Direct use of `plotting` still works; emits DeprecationWarning (ADR-011 pattern)
- **v0.11.0**: Remove module entirely; users must import from `viz` directly

**Migration Path**:
```python
# OLD (deprecated, v0.10.0)
from calibrated_explanations import plotting
figure = plotting.plot_explanation(expl)  # ⚠️ DeprecationWarning

# NEW (recommended, v0.10.0+)
from calibrated_explanations.viz import plot_explanation
figure = plot_explanation(expl)  # ✅ Clean
```

**Deprecation Status**: ✅ **Ready for v0.10.0 Implementation**

**Deprecation Message Template**:
```
"calibrated_explanations.plotting" module is deprecated and will be removed in v0.11.0.
  ❌ DEPRECATED: from calibrated_explanations import plotting
  ✓ RECOMMENDED: from calibrated_explanations.viz import plot_explanation, ...
See https://calibrated-explanations.readthedocs.io/en/latest/migration/plotting_relocation.html
```

---

### 4. `calibrated_explanations.perf` — Temporary Cache/Parallel Shim

**Current Role**: Thin wrapper re-exporting `cache` and `parallel` submodules for backward compatibility.

**Contents** (Post-Stage 1b):
- Re-exports from `calibrated_explanations.cache` and `calibrated_explanations.parallel`
- Factory helpers for `CalibratorCache`, `ParallelExecutor`, etc.

**Classification**: **Temporary; Scheduled for Removal after v0.10.1**

**Rationale**:
- Introduced in Stage 1b when cache/parallel split from `perf` into dedicated packages
- Provides backward compatibility for code importing from `perf` directly
- Will be removed once users migrate to direct submodule imports

**ADR-001 Alignment**:
- Intentional compatibility shim; temporary per ADR-011 (Migration Gates)
- Will be removed after migration period expires

**Deprecation Strategy** (Two-Release Window):
- **v0.10.1**: Direct use of `perf` still works; emits DeprecationWarning (ADR-011 pattern)
- **v0.11.0**: Remove shim entirely; users must import from `cache`/`parallel` directly

**Migration Path**:
```python
# OLD (deprecated, v0.10.1)
from calibrated_explanations.perf import CalibratorCache, ParallelExecutor  # ⚠️ DeprecationWarning

# NEW (recommended, v0.10.1+)
from calibrated_explanations.cache import CalibratorCache
from calibrated_explanations.parallel import ParallelExecutor  # ✅ Clean
```

**Deprecation Status**: ⏸️ **Deferred** (will implement once Stage 1b completes cache/parallel split)

---

### 5. `calibrated_explanations.integrations` — Third-Party Integration Helpers

**Current Role**: Hosts integration adapters for third-party libraries (LIME, SHAP, etc.) that augment core explanations.

**Contents**:
- Integration adapters and helper factories
- Third-party library bridges

**Classification**: **Permanent; Aligned with ADR-001 Utilities Guidance**

**Rationale**:
- Clearly separates domain-agnostic extensions from core
- No cross-talk with core architecture; can evolve independently
- Aligns with ADR-001 "utilities" guidance for external adapters

**ADR-001 Alignment**:
- ✅ Fully aligned with ADR-001 scope as permanent utility package
- Clear boundary: third-party integrations, not core domain logic
- Can remain indefinitely

**Migration Path**: **None required** (permanent part of architecture)

**Deprecation Status**: ✅ **Not Deprecated** (permanent)

---

## Architecture Diagram: Remaining Namespaces

```
calibrated_explanations/
├── [CORE]
│   ├── core/                  ✅ ADR-001 Core (factories, orchestrators)
│   ├── calibration/           ✅ ADR-001 Extracted (v0.10.0)
│   ├── explanations/          ✅ ADR-001 Domain (strategies)
│   ├── plugins/               ✅ ADR-001 Registry & loading
│   ├── viz/                   ✅ ADR-001 Visualization abstractions
│   ├── cache/                 ✅ ADR-001 Extracted (v0.10.0, Stage 1b)
│   ├── parallel/              ✅ ADR-001 Extracted (v0.10.0, Stage 1b)
│   ├── schema/                ✅ ADR-001 Extracted (v0.10.0, Stage 1c)
│   └── utils/                 ✅ ADR-001 Non-domain helpers
│
├── [DEVIATIONS - DOCUMENTED]
│   ├── api/                   ⏸️  Temporary (relocation to core deferred v1.1+)
│   ├── legacy/                ⏳ Temporary (removal scheduled v2.0.0)
│   ├── plotting.py            ✅ Deprecated (removal scheduled v0.11.0)
│   ├── perf/                  ⏳ Temporary shim (removal scheduled v0.11.0)
│   └── integrations/          ✅ Permanent (aligned with ADR-001 utilities)
```

---

## ADR-001 Gap #6 Closure Checklist

| Item | Status | Details |
|------|--------|---------|
| **`api` namespace audited** | ✅ | Classified as intentional deviation; relocation deferred to v1.1+ |
| **`legacy` namespace audited** | ✅ | Classified as deprecation boundary; removal target v2.0.0 |
| **`plotting.py` audited** | ✅ | Classified as deprecated; removal target v0.11.0 |
| **`perf` shim audited** | ✅ | Classified as temporary; removal target v0.11.0 |
| **`integrations` audited** | ✅ | Classified as permanent; aligned with ADR-001 |
| **Migration paths documented** | ✅ | Examples provided for each deprecated namespace |
| **ADR alignment documented** | ✅ | Each deviation justified in relation to ADR-001 and ADR-011 |
| **Removal timelines specified** | ✅ | v0.11.0 (plotting, perf); v1.1+ (api); v2.0.0 (legacy) |
| **Public API impact assessed** | ✅ | All deviations outside stage 3 sanctioned API (no additional changes needed) |
| **Deprecation warnings planned** | ✅ | Templates provided for plotting (ready), legacy (v1.1+), perf (ready) |

---

## Implementation Summary

**Files Created/Modified**:
- ✅ `improvement_docs/ADR-001-STAGE-4-COMPLETION-REPORT.md` — This document

**Documentation Added to Existing Files**:
- ✅ Updated `RELEASE_PLAN_V1.md` — Stage 4 status and timelines
- ✅ Updated `CHANGELOG.md` — Stage 4 completion entry
- ✅ Updated `ADR-001-gap-analysis.md` — Gap #6 marked complete

**Deprecation Ready Items**:
- `plotting.py` — Ready for v0.10.0 deprecation implementation
- `perf` shim — Ready for v0.10.1 deprecation implementation (post-Stage 1b)

**Future Work** (not in v0.10.0 scope):
- v1.1+: Deprecate `api` imports with ADR-011 warnings
- v1.1+: Consolidate `api` into core utilities
- v2.0.0: Remove `legacy` namespace entirely

---

## ADR-001 Overall Completion Status

| Gap | Stage | Status | Notes |
|-----|-------|--------|-------|
| #1: Calibration embedded | Stage 1a | ✅ Complete | Extracted to top-level package with shim |
| #2: Cross-sibling imports | Stage 2 | ✅ Complete | Refactored to lazy/TYPE_CHECKING imports |
| #3: Cache/parallel not split | Stage 1b | ✅ Complete | Split into dedicated packages with perf shim |
| #4: Schema validation missing | Stage 1c | ✅ Complete | Created schema validation package |
| #5: Public API overly broad | Stage 3 | ✅ Complete | Deprecated 13 unsanctioned, reserved 3 sanctioned |
| #6: Extra namespaces undocumented | **Stage 4** | **✅ Complete** | **All documented with rationale & timelines** |

---

## Sign-Off

**Stage 4 Status**: ✅ **COMPLETE**

All remaining namespaces are now documented with:
- ✅ Clear classification (permanent, temporary, or intentionally deferred)
- ✅ Documented rationale and ADR alignment
- ✅ Migration paths for deprecated items
- ✅ Removal timelines and deprecation strategies
- ✅ Public API impact assessment

**ADR-001 Gap #6 is CLOSED.**

ADR-001 gap closure is now progressing toward **Stages 5–6**:
- **Stage 5**: Add import graph linting and enforcement tests (deferred to v0.11+)
- **Stage 6**: Final adoption review and v1.0.0 readiness assessment

See `improvement_docs/RELEASE_PLAN_V1.md` for continuation plan.
