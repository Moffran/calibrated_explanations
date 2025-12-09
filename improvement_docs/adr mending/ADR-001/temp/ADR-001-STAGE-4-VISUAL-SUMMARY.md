# ADR-001 Stage 4: Visual Summary & Architecture Impact

**Prepared for**: ADR-001 Gap Closure Plan (Stages 0–6)
**Date**: 2025-11-28
**Status**: ✅ COMPLETE

---

## Quick Reference: Namespace Classification

```
calibrated_explanations/
│
├─ [CORE ADR-001 PACKAGES] ✅ Primary Architecture
│  ├─ core/                          (factories, orchestrators)
│  ├─ calibration/                   (calibration algorithms)
│  ├─ explanations/                  (explanation strategies)
│  ├─ plugins/                       (registry & loading)
│  ├─ viz/                           (visualization abstractions)
│  ├─ cache/                         (caching layer)
│  ├─ parallel/                      (parallel execution)
│  ├─ schema/                        (JSON schema validation)
│  └─ utils/                         (non-domain utilities)
│
└─ [EXTRA NAMESPACES - NOW DOCUMENTED] 📋 Stage 4 Closure
   ├─ api/                           ⏸️  INTENTIONAL DEVIATION (relocation deferred v1.1+)
   ├─ legacy/                        ⏳ DEPRECATION SCHEDULED (removal v2.0.0)
   ├─ plotting.py                    ❌ DEPRECATED (removal v0.11.0)
   ├─ perf/                          ⏳ TEMPORARY SHIM (removal v0.11.0)
   └─ integrations/                  ✅ PERMANENT (ADR-001 aligned)
```

---

## Stage 4 Achievements Checklist

| Task | Status | Evidence |
|------|--------|----------|
| Audit all 5 remaining namespaces | ✅ | `api`, `legacy`, `plotting`, `perf`, `integrations` reviewed |
| Classify each namespace | ✅ | Categorized as permanent / temporary / intentional deviation |
| Document rationale for each | ✅ | ADR-001 alignment or justified deviation for all 5 |
| Create migration paths | ✅ | Examples provided for deprecated namespaces |
| Specify removal timelines | ✅ | v0.11.0 (plotting, perf); v1.1+ (api); v2.0.0 (legacy) |
| Update architectural docs | ✅ | `RELEASE_PLAN_V1.md`, `CHANGELOG.md` updated |
| ADR-001 Gap #6 closed | ✅ | All namespaces now documented |

---

## Namespace Status Summary

### 1. `api/` — Parameter Validation Facade

```
Classification: ⏸️  INTENTIONAL DEVIATION
Timeline: v1.1+ (relocation deferred)
Impact: Internal; no user-facing deprecation
Status: DOCUMENTED ✅
```

**Why retained?**
- Shared contract layer breaking circular imports between core and plugins
- Frequent usage across codebase prevents immediate deprecation
- Future relocation to `core.utils.api` or `utils.api` will be transparent to users

**Migration path:** Internal reorganization only; no changes required from users

---

### 2. `legacy/` — Compatibility Shims

```
Classification: ⏳ DEPRECATION SCHEDULED
Timeline: v2.0.0 (removal)
Impact: Users on old APIs; migration path needed
Status: DOCUMENTED ✅
```

**Why retained?**
- Migration helpers for users upgrading from pre-v0.9 code
- Provides compatibility bridge for extended support users
- Removal coordinated well in advance (v1.0 → v2.0 window)

**Migration timeline:**
```
v0.10–v1.0   → Users import from legacy; no warnings yet
v1.1–v1.x    → DeprecationWarning added (ADR-011)
v2.0.0       → Complete removal; users must migrate
```

**Example:**
```python
# v0.10-v1.0 (no warnings)
from calibrated_explanations.legacy import LegacyExplanation

# v1.1+ (DeprecationWarning)
from calibrated_explanations.legacy import LegacyExplanation  # ⚠️ Warning
# ✓ RECOMMENDED: from calibrated_explanations.explanations import FastExplanation

# v2.0.0+ (removed)
from calibrated_explanations.legacy import LegacyExplanation  # ❌ ImportError
```

---

### 3. `plotting.py` — Deprecated Convenience Re-export

```
Classification: ❌ DEPRECATED
Timeline: v0.10.0 (warnings) → v0.11.0 (removal)
Impact: Users still using module-level import
Status: READY FOR IMPLEMENTATION ✅
```

**Why deprecated?**
- Redundant wrapper around `viz` submodule
- Violates ADR-001 namespace clarity principle
- Users should import directly from `viz`

**Deprecation timeline:**
```
v0.10.0      → DeprecationWarning on import
v0.11.0      → Module removed entirely
```

**Deprecation message:**
```
"calibrated_explanations.plotting" module is deprecated and will be removed in v0.11.0.
  ❌ DEPRECATED: from calibrated_explanations import plotting
  ✓ RECOMMENDED: from calibrated_explanations.viz import plot_explanation, ...
See https://calibrated-explanations.readthedocs.io/en/latest/migration/plotting_relocation.html
```

**Example:**
```python
# v0.10.0 (deprecated, shows warning)
from calibrated_explanations import plotting
figure = plotting.plot_explanation(expl)  # ⚠️ DeprecationWarning

# v0.10.0+ (recommended)
from calibrated_explanations.viz import plot_explanation
figure = plot_explanation(expl)  # ✅ Clean

# v0.11.0+ (old import fails)
from calibrated_explanations import plotting  # ❌ ImportError
```

---

### 4. `perf/` — Temporary Cache/Parallel Shim

```
Classification: ⏳ TEMPORARY SHIM
Timeline: v0.10.1 (warnings) → v0.11.0 (removal)
Impact: Users importing from perf directly
Status: READY FOR IMPLEMENTATION (post-Stage 1b) ✅
```

**Why temporary?**
- Created as compatibility wrapper after Stage 1b cache/parallel split
- Users should import directly from `cache` and `parallel`
- Will be removed once migration period expires

**Deprecation timeline:**
```
v0.10.1      → DeprecationWarning on import
v0.11.0      → Shim removed entirely
```

**Deprecation message:**
```
"calibrated_explanations.perf" is deprecated and will be removed in v0.11.0.
  ❌ DEPRECATED: from calibrated_explanations.perf import CalibratorCache, ParallelExecutor
  ✓ RECOMMENDED:
     from calibrated_explanations.cache import CalibratorCache
     from calibrated_explanations.parallel import ParallelExecutor
See https://calibrated-explanations.readthedocs.io/en/latest/migration/perf_relocation.html
```

**Example:**
```python
# v0.10.1 (deprecated, shows warning)
from calibrated_explanations.perf import CalibratorCache, ParallelExecutor  # ⚠️ DeprecationWarning

# v0.10.1+ (recommended)
from calibrated_explanations.cache import CalibratorCache
from calibrated_explanations.parallel import ParallelExecutor  # ✅ Clean

# v0.11.0+ (old import fails)
from calibrated_explanations.perf import CalibratorCache  # ❌ ImportError
```

---

### 5. `integrations/` — Third-Party Integration Helpers

```
Classification: ✅ PERMANENT
Timeline: Indefinite (no removal planned)
Impact: None (permanent architecture component)
Status: DOCUMENTED ✅
```

**Why permanent?**
- Clearly separates third-party extensions from core domain logic
- No cross-talk with core; can evolve independently
- Aligns with ADR-001 utilities guidance

**Migration path:** None required (permanent)

---

## ADR-001 Completion Snapshot

| Stage | Gap | Status | Details |
|-------|-----|--------|---------|
| 0 | Scope & divergences | ✅ Complete | Confirmed boundaries; documented deviations |
| 1a | Calibration extraction | ✅ Complete | Moved to top-level package with shim |
| 1b | Cache/parallel split | ✅ Complete | Split into dedicated packages; perf shim created |
| 1c | Schema validation | ✅ Complete | New package created; helpers extracted |
| 2 | Cross-sibling imports | ✅ Complete | CalibratedExplainer refactored (14 imports → lazy/TYPE_CHECKING) |
| 3 | Public API narrowing | ✅ Complete | Deprecated 13 unsanctioned; reserved 3 sanctioned |
| **4** | **Namespace docs** | **✅ Complete** | **All 5 remaining namespaces documented** |
| 5 | Import graph linting | ⏸️ Deferred | Deferred to v0.11+; design phase ready |
| 6 | v1.0.0 readiness | ⏸️ Pending | Final adoption review after stages 5+ |

---

## Impact Assessment

### User-Facing Changes (v0.10.0 Release)

**Deprecation Warnings Added:**
- None new in v0.10.0 for Stage 4 (Stage 3 added 13 warnings for public API)

**Ready for Implementation in v0.10.1:**
- `plotting` module → DeprecationWarning template ready
- `perf` shim → DeprecationWarning template ready

**Future Warnings (v1.1+):**
- `legacy` imports → ADR-011 DeprecationWarning pattern

**No changes required for users on:**
- `api` (internal relocation transparent)
- `integrations` (permanent; no changes)

### Migration Burden

| Group | Burden | Timeline |
|-------|--------|----------|
| Users on sanctioned API | ✅ None | No changes needed |
| Users on `plotting` imports | 🟡 Low | Single find/replace; 6-month grace period |
| Users on `perf` imports | 🟡 Low | Single find/replace; 6-month grace period |
| Users on `legacy` imports | 🟡 Low–Medium | Multiple replacements; 2+ year grace period |
| Users on `api` imports | ✅ None | Transparent internal reorganization |

---

## Architectural Clarity Improvement

### Before Stage 4:
```
❓ Is `plotting` a permanent facade or deprecated?
❓ Why does `perf` exist if cache/parallel are separate?
❓ Will `api` remain in top-level forever?
❓ When will `legacy` be removed?
❓ Is `integrations` considered core architecture?
```

### After Stage 4:
```
✅ `plotting` is deprecated (removal v0.11.0)
✅ `perf` is a temporary shim (removal v0.11.0)
✅ `api` is deferred relocation (v1.1+)
✅ `legacy` is scheduled removal (v2.0.0)
✅ `integrations` is permanent; ADR-001 aligned
```

---

## Next Steps: Stages 5–6

### Stage 5: Import Graph Linting (v0.11+ scope)
- [ ] Add static analysis to detect violations of documented boundaries
- [ ] Create linting rule enforcement in CI
- [ ] Document import validation strategy

### Stage 6: v1.0.0 Readiness (pre-release)
- [ ] Final adoption review of all stages
- [ ] Confirm all ADR-001 gaps are closed
- [ ] Validate deprecation timelines and migration paths
- [ ] Sign-off on ADR-001 implementation

---

## References

- **Stage 4 Completion Report**: `improvement_docs/ADR-001-STAGE-4-COMPLETION-REPORT.md`
- **Release Plan**: `improvement_docs/RELEASE_PLAN_V1.md`
- **ADR-001**: `improvement_docs/adrs/ADR-001.md`
- **ADR-011 (Deprecation)**: `improvement_docs/adrs/ADR-011.md`
- **Gap Analysis**: `improvement_docs/ADR-gap-analysis.md`

---

## Sign-Off

✅ **Stage 4 COMPLETE** (2025-11-28)

All remaining namespaces are now documented with clear classification, rationale, and migration paths. ADR-001 Gap #6 is closed.

**ADR-001 Gap Closure Progress**: **Stages 0–4 COMPLETE** (5 of 6 major stages)
