# ADR-001 Stage 5 Gap Analysis: Executive Decision Brief

**Date**: 2025-11-30
**Status**: Action Required
**Impact**: v0.10.0 Release Readiness + Long-Term Architecture
**Audience**: Maintainers & Steering Committee

---

## THE SITUATION

The ADR-001 Stage 5 linting implementation successfully detects import graph violations, but revealed **153 cross-sibling imports** that don't match the boundary rules currently in `scripts/check_import_graph.py`.

The Stage 5 completion report claimed "no boundary violations," but this was based on an **incomplete allowlist** in the linter configuration. The violations are real and systematic:

- **57 exception imports** (core.exceptions used everywhere)
- **25 orchestrator imports** (core.calibrated_explainer coordinates packages)
- **15 interface imports** (strategies, feature tasks, etc.)
- **Plus 56 others** (plugins, visualization, caching)

**The Core Problem**: Either the violations are **intentional design** (and should be allowed) or they're **architectural debt** (and should be refactored).

---

## WHAT THE VIOLATIONS MEAN

### They're NOT bugs—they're architectural choices that need codification

**Example 1: Exception Taxonomy** (57 violations)
- Every package that raises exceptions imports from `core.exceptions` (ADR-002 unified taxonomy).
- Centralizing exceptions is *correct* and *intentional*.
- **Question**: Should this be an allowed cross-sibling import, or should each package re-export from a facade?

**Example 2: Orchestrator Hub** (25 violations in `calibrated_explainer.py`)
- The explainer coordinates calibration, explanation, caching, plugins—it *needs* to import from all siblings.
- This is the intended architecture (per ADR-001).
- **Question**: Is this acceptable as-is, or should all sibling→explainer calls go through a coordinator layer?

**Example 3: Calibration State Checks** (6 violations)
- Calibration modules check if the explainer is fitted/calibrated (type hints + runtime checks).
- State is owned by explainer, but calibrators need visibility.
- **Question**: Is this acceptable (current), or should state be exposed via contracts/facade?

---

## THE OPTIONS AT A GLANCE

| **Option** | **Action** | **Timeline** | **Effort** | **Outcome** | **Verdict** |
|:---|:---|:---|:---|:---|:---|
| **A: Allow-List** | Document violations as intentional in linter | NOW (v0.10.0) | 2h | Linter passes, violations codified | ✅ **FAST** |
| **B: Contracts Layer** | Create `core/contracts.py` with re-exports + protocols | v0.10.1 | 10h | Cleaner boundaries, explicit contracts | ✅ **RECOMMENDED** |
| **C: Lazy Imports** | Use TYPE_CHECKING blocks + runtime imports | v0.10.1 | 12h | Breaks cycles, less code churn | ⚠️ **ALTERNATIVE** |
| **D: Coordinator** | Create cross-package coordinator/facade layer | v0.11.0+ | 50h | Cleanest long-term, enables package split | 🟢 **FUTURE** |

---

## CRITICAL DECISION POINTS

### Decision 1: Are Exceptions a "Shared Domain Contract"?

**Context**: 57 violations, all importing `core.exceptions` for ADR-002 exception raising.

**Question**: Should every package be able to import `core.exceptions`, or should they import from a facade?

**Options**:
- ✅ **YES (Option A)**: Allow `(*,core.exceptions)` as a sanctioned shared contract. Minimal refactoring, justified by ADR-002 (unified exception taxonomy).
- 🟡 **MAYBE (Option B)**: Create `core/contracts.py` that re-exports exceptions. Cleaner boundaries, but adds abstraction layer.
- ❌ **NO (none)**: Duplicate exception types in each package. Not recommended—defeats ADR-002 purpose.

**Recommendation**: **YES (Option A in short term, B in medium term)**
- v0.10.0: Allow `(*,core.exceptions)` with documentation.
- v0.10.1: Move to `core/contracts.py` re-export to clarify intent.

---

### Decision 2: Should `core.calibrated_explainer` Be the Orchestrator Hub?

**Context**: 25 violations from explainer importing calibration, plugins, cache, parallel, explanations.

**Question**: Is it OK for the orchestrator to have spokes to all siblings, or should all communication route through a coordinator?

**Options**:
- ✅ **HUB PATTERN (Option A)**: Keep explainer as hub. Simpler now, documented as intentional.
- 🟡 **COORDINATOR PATTERN (Option D)**: Create `core/coordinator.py` that mediates all calls. More complex, better for future modularization.

**Recommendation**: **HUB PATTERN now (v0.10.0), revisit in v0.11.0 if packaging goals require separate distributions.**
- v0.10.0–v0.10.1: Allow direct spokes (documented).
- v0.11.0: If we're planning multi-distribution packaging, switch to coordinator.

---

### Decision 3: Should Siblings Import Core Domain Interfaces?

**Context**: 12 violations for imports like `core.explain.feature_task`, `core.prediction.orchestrator`.

**Question**: Should siblings import these as-is, or via a contracts layer?

**Options**:
- ✅ **DIRECT IMPORTS (Option A short term)**: Siblings directly import interfaces. Works now, but couples to core internals.
- 🟡 **CONTRACTS FACADE (Option B)**: `core/contracts.py` exports protocols. Cleaner boundary, but adds layer.

**Recommendation**: **Move to contracts in v0.10.1 (Option B)**
- v0.10.0: Allow direct imports (temporary).
- v0.10.1: Export from `core/contracts.py` for clarity.

---

### Decision 4: What About Plugin Coordination?

**Context**: 10 violations for plugins importing core.*, explanations.

**Question**: Should plugins import internals, or wait for ADR-006 plugin interface (v0.10.2)?

**Options**:
- ✅ **ALLOW TEMPORARILY (Option A)**: Allow plugin→core imports until ADR-006 plugin interface ready.
- ⚠️ **STRICT BOUNDARY NOW (Custom)**: Refactor to defer all plugin imports. Adds work now.

**Recommendation**: **Allow temporarily (Option A), enforce cleaner boundaries in v0.10.2 (ADR-006)**
- v0.10.0–v0.10.1: Document as temporary.
- v0.10.2: Introduce proper plugin interface, migrate imports.

---

## RECOMMENDED APPROACH: HYBRID (Option A + B)

### Phase 1: Immediate (v0.10.0, ~2 hours)

**Objective**: Unblock v0.10.0 release and enable linter in CI.

**Action**:
1. Update `scripts/check_import_graph.py` to allow:
   - `(*,core.exceptions)` – shared exception taxonomy (ADR-002)
   - `(calibration,core.calibrated_explainer)` – explainer is orchestrator
   - `(calibration,core.explain.feature_task)` – interface imports
   - `(explanations,core)` – core domain models
   - `(plugins,core)` – plugin coordination (temporary until ADR-006)
   - `(cache,core)` – cache state sync
   - `(parallel,core)` – parallel coordination
   - `(viz,core)` + `(viz,explanations)` – visualization adapters
   - `(api,core)` – parameter validation

2. Document rationale in `improvement_docs/ADR-001-EXCEPTIONS-AND-CONTRACTS.md`

3. Wire linter into CI (`/.github/workflows/lint.yml`)

**Exit Criteria**:
- `python scripts/check_import_graph.py` → passes with 0 violations
- Linter integrated into CI (blocks PRs with violations)
- Release notes document: "ADR-001 Stage 5: Import boundaries codified and enforced in CI"

---

### Phase 2: Polish (v0.10.1, ~10–12 hours)

**Objective**: Establish formal domain contracts layer; deepen boundaries.

**Action**:
1. Create `src/calibrated_explanations/core/contracts.py`:
   ```python
   """Domain contracts: shared interfaces for all packages."""

   # Re-export exception taxonomy (single import point)
   from calibrated_explanations.core.exceptions import (
       CalibratedExplanationError,
       ValidationError,
       ConfigurationError,
       NotFittedError,
   )

   # Export protocols for siblings
   from abc import ABC, abstractmethod
   from typing import Protocol

   class ExplanationStrategy(ABC):
       """Interface for explanation algorithms."""

   class CalibrationStrategy(ABC):
       """Interface for calibrators."""

   class CalibratedExplainerState(Protocol):
       """Protocol for explainer state (used by calibrators, cache)."""
       @property
       def is_fitted(self) -> bool: ...
       @property
       def is_calibrated(self) -> bool: ...
   ```

2. Update imports in ~15 files:
   - `calibration/*.py` → import from `core.contracts` instead of `core.exceptions` + `core.calibrated_explainer`
   - `plugins/*.py` → same
   - `cache/*.py` → same
   - `viz/*.py` → same

3. Update linter allowlist:
   - Change `(*,core.exceptions)` to `(*,core.contracts)`
   - Add rationale comment

4. Add regression tests

**Exit Criteria**:
- `python scripts/check_import_graph.py` → passes (fewer violations due to contracts consolidation)
- `core/contracts.py` is the single import point for exceptions
- All siblings import from contracts, not core internals
- Contracts module documented and tested

---

### Phase 3: Long-Term Assessment (v0.11.0+, Optional)

**Objective**: Evaluate if Option D (coordinator pattern) is necessary.

**Trigger**: If product goals include splitting packages into separate distributions.

**Action** (if triggered):
- Design `core/coordinator.py` as a message/service bus
- Migrate all cross-sibling calls through coordinator
- Update linter allowlist to only allow `(*,core.coordinator)`

**Exit Criteria** (if pursued):
- All cross-package calls routed through coordinator
- Packages can theoretically be split into separate distributions
- Documentation updated

**If NOT triggered**: Maintain Phase 2 (contracts layer) indefinitely. It's sufficient and clean.

---

## RISK ANALYSIS

### Risk: We Allow Too Many Cross-Sibling Imports and Architecture Degrades

**Mitigation**:
- Document each allowed import with rationale + deprecation timeline.
- Review in v0.11.0 post-release.
- Phase 2 (contracts) tightens boundaries without code churn.

### Risk: Phase 2 (Contracts) Is Too Much Work and Gets Skipped

**Mitigation**:
- Contracts layer is relatively low-effort (10h) and high-value.
- Schedule it early in v0.10.1 to avoid scope creep.
- Refactoring is mechanical (find-replace imports across ~15 files).

### Risk: Phase 3 (Coordinator) Gets Postponed Forever

**Mitigation**:
- Only pursue if actual packaging/distribution goals exist.
- v0.11.0 is a natural reassessment point.
- If not needed, Phase 2 (contracts) is the final target state.

### Risk: Linter Becomes Maintenance Burden

**Mitigation**:
- AST-based linter is fairly stable.
- Allowlist rules should be stable after v0.10.1 (only add new packages/rules rarely).
- Keep implementation simple (~350 LOC, no dependencies).

---

## EFFORT SUMMARY

| Phase | Timeline | Hours | LOC Change | Risk | Value |
|:---|:---|:---:|:---:|:---:|:---:|
| **Phase 1** (Option A) | v0.10.0 NOW | 2h | ~50 lines | 🟢 LOW | 🔴 UNBLOCKS v0.10.0 |
| **Phase 2** (Option B) | v0.10.1 | 10–12h | ~200 lines + 800 imports | 🟡 MED | 🟢 ARCHITECTURE |
| **Phase 3** (Option D) | v0.11.0+ | 40–50h | ~500 lines + 1000+ imports | 🔴 HIGH | 🟢 FUTURE AGILITY |

---

## DECISION MATRIX FOR MAINTAINERS

### Question 1: Do we ship v0.10.0 with allow-listed violations?
- **YES** → Approve Phase 1 (Option A). Unblocks release in 2h.
- **NO** → Halt v0.10.0. Refactor now (risks delay).

**Recommendation**: **YES**. The violations represent *intentional* architecture (orchestrator hub, shared exceptions, etc.). Document and move forward.

---

### Question 2: Do we clean up boundaries in v0.10.1 (Option B)?
- **YES** → Schedule Phase 2. Improves architecture, 10–12h effort.
- **NO** → Remain with Phase 1 indefinitely (acceptable, but less clean).

**Recommendation**: **YES**. Phase 2 is relatively low-effort and solidifies boundaries for long-term maintainability.

---

### Question 3: Do we pursue Option D (Coordinator) in v0.11.0?
- **YES (if planning multi-distribution packaging)** → Reserve v0.11.0 capacity.
- **NO (single distribution)** → Skip. Phase 2 is final target state.

**Recommendation**: **Defer decision to v0.11.0 planning**. Reassess based on actual packaging/distribution goals at that time.

---

## RECOMMENDED VOTES

| Item | Recommendation | Confidence | Next Step |
|:---|:---|:---:|:---|
| **Approve Phase 1 (v0.10.0)** | ✅ YES | 95% | Begin in next 2h |
| **Schedule Phase 2 (v0.10.1)** | ✅ YES | 85% | Assign after v0.10.0 release |
| **Reserve Phase 3 (v0.10.1+)** | 🟡 DEFER | 70% | Revisit in v0.11.0 planning |

---

## WHAT HAPPENS IF WE DON'T ACT

### If Phase 1 Skipped:
- v0.10.0 release **blocked**.
- Linter still reports 153 violations in CI.
- No boundary enforcement.
- **Impact**: HIGH (release delay)

### If Phase 2 Skipped:
- Boundaries less explicit.
- Siblings still tightly coupled.
- Harder to refactor later.
- **Impact**: MEDIUM (tech debt accumulates, but not release-blocking)

### If Phase 3 Skipped:
- No coordinator layer.
- Packages remain tightly coupled.
- Future multi-distribution split becomes harder.
- **Impact**: MEDIUM (optional, only matters if distribution split is goal)

---

## NEXT IMMEDIATE ACTIONS

1. **Maintainers Review & Vote** (Today/Tomorrow):
   - Approve Phase 1 (Option A)?
   - Approve Phase 2 follow-up (Option B)?
   - Defer Phase 3 decision?

2. **If Phase 1 Approved** (Immediate):
   - Assign developer to update linter (2h task).
   - Run tests, verify CI integration.
   - Update CHANGELOG.md and release notes.

3. **If Phase 2 Approved** (Schedule post-v0.10.0):
   - Design `core/contracts.py` (1h).
   - Assign refactoring work (~8h coding, 2h testing).
   - Merge before v0.10.1 release.

---

## SUPPORTING DOCUMENTS

For detailed analysis, see:
1. **`improvement_docs/ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md`** (full 4-option analysis + rationale)
2. **`improvement_docs/ADR-001-VIOLATIONS-VISUAL-SUMMARY.md`** (visual architecture maps + heatmaps)
3. **`violations.txt`** (raw linter output, generated by `python scripts/check_import_graph.py`)

---

**Status**: Awaiting maintainer decision on Phases 1, 2, and 3.

**Prepared by**: ADR-001 Stage 5 Gap Analysis
**Date**: 2025-11-30
**Contact**: See CONTRIBUTING.md for maintainer contact info
