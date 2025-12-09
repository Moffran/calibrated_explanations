# ADR-001 Cross-Sibling Import Refactoring: Strategic Options

**Date**: 2025-11-30
**Status**: Analysis & Options Document
**Scope**: Resolve 153 cross-sibling import violations to achieve full ADR-001 Stage 5 compliance

## Executive Summary

The Stage 5 linting implementation has revealed **153 cross-sibling import violations** that were not previously enforced. These violations prevent the `check_import_graph.py` linter from passing cleanly and block CI integration. The violations cluster in a few high-traffic areas:

| Source Package | Violations | Key Targets | Severity |
|---|---|---|---|
| **core** | 76 | utils, calibration, explanations, plugins, cache, parallel, api | CRITICAL |
| **calibration** | 26 | core.exceptions, core.explain.feature_task, core.calibrated_explainer | HIGH |
| **explanations** | 15 | core.*, plugins | HIGH |
| **plugins** | 10 | core, explanations, utils | HIGH |
| **api** | 2 | core.exceptions, core.wrap_explainer | MEDIUM |
| **cache** | 1 | core.exceptions | LOW |
| **utils** | 3 | core.* | MEDIUM |
| **Other** | 4 | misc | LOW |

---

## Part 1: Analysis of Violation Patterns

### Pattern 1: Exception Imports (57 occurrences)

**Problem**: Across 10+ modules, `core.exceptions` is imported at module level for direct exception raising.

**Files Affected**:
- `calibration/{interval_learner, interval_regressor, state, venn_abers}.py`
- `plugins/{builtins, registry, explanations}.py`
- `api/{params, quick}.py`
- `cache/cache.py`
- `utils/helper.py`
- `core/calibration_metrics.py`
- `viz/{builders, narrative_plugin}.py`

**Current ADR-001 Rule**: Cross-sibling imports not allowed except for `utils`, `schema`, `api`.

**Root Cause**: Centralized exception taxonomy (ADR-002) is in `core.exceptions`, but modules across siblings need to raise these exceptions. The current boundary rules don't account for exceptions as a "shared domain contract."

### Pattern 2: Core.calibrated_explainer Imports (19 occurrences)

**Problem**: Calibration, cache, and other modules import `CalibratedExplainer` or `WrapCalibratedExplainer` from core.

**Files Affected**:
- `calibration/{state, interval_learner, summaries}.py` → core.calibrated_explainer (for type hints, state checks)
- `plugins/builtins.py` → core.wrap_explainer
- Multiple runtime modules

**Root Cause**: The wrapper/explainer is the orchestrator that coordinates calibration, explanation, and caching. Calibration modules need to check state or access instance metadata, creating a circular conceptual dependency.

### Pattern 3: Core.explain.* Imports (12 occurrences)

**Problem**: Calibration and plugins import explanation strategy interfaces from `core.explain`.

**Files Affected**:
- `calibration/interval_learner.py` → core.explain.feature_task
- `plugins/explanations.py` → core.explain.orchestrator
- `core/calibration_metrics.py` → core.explain.feature_task

**Root Cause**: Explanation tasks are defined in core but needed by calibrators for type hints and task routing.

### Pattern 4: Core.utils Imports (19 occurrences)

**Problem**: Multiple packages import utilities that are scoped under core.utils.

**Files Affected**:
- `core/calibrated_explainer` → {utils.helper, utils.rng, utils.discretizers, utils.deprecations}
- `plugins/builtins` → similar

**Root Cause**: Utilities are spread across both `core/utils` and top-level `utils/`. Current boundary allows `(*,utils)` but doesn't distinguish.

### Pattern 5: Calibration and Explanations Circular Dependency (9 occurrences)

**Problem**: Calibration modules import from explanations (or vice versa), creating coupling.

**Files Affected**:
- `core/calibrated_explainer` ↔ {calibration, explanations}
- Various runtime coordination points

**Root Cause**: The wrapper orchestrates both, but internal modules have hard dependencies rather than using lazy imports or interfaces.

### Pattern 6: Plugins and Core Coupling (6 occurrences)

**Problem**: Plugins directly import from core internals rather than going through a plugin interface.

**Root Cause**: No formal plugin interface layer defined yet (ADR-006 work).

---

## Part 2: Strategic Refactoring Options

Below are four distinct refactoring strategies, ranging from pragmatic to architectural. Each addresses violations differently.

### **Option A: Pragmatic Allow-Listing (Fastest, Least Invasive)**

**Approach**: Update `scripts/check_import_graph.py` to explicitly allow high-traffic cross-sibling imports that are intentional and documented.

**Rationale**:
- Violations are real but represent documented architectural patterns (e.g., shared exception taxonomy, plugin coordination).
- Rather than refactor code to avoid these imports, explicitly codify them as "allowed by design."
- Aligns with ADR-001 which states "No cross-talk between siblings **except through** **core domain models or explicitly defined interfaces**."

**Implementation**:

```python
# In scripts/check_import_graph.py, expand allowed_cross_sibling:

allowed_cross_sibling = {
    # ... existing entries ...

    # SHARED EXCEPTION TAXONOMY (ADR-002)
    # All packages can import core.exceptions for raising ADR-002 exceptions
    ('*', 'core.exceptions'): [],  # Exception taxonomy is a shared contract

    # CORE ORCHESTRATOR PATTERN
    # Calibration and explanations import core orchestrator/wrapper for state/type hints
    ('calibration', 'core.calibrated_explainer'): [],
    ('calibration', 'core.wrap_explainer'): [],
    ('explanations', 'core'): [],  # explanations uses core domain models
    ('cache', 'core.calibrated_explainer'): [],  # cache integrates with wrapper state

    # EXPLANATION TASK INTERFACE (domain model)
    # Calibration imports feature_task interface for type hints
    ('calibration', 'core.explain.feature_task'): [],

    # PLUGIN COORDINATION (temporary until ADR-006 plugin interface)
    ('plugins', 'core'): [],
    ('plugins', 'explanations'): [],

    # API PARAMETER VALIDATION
    ('api', 'core'): [],

    # CORE INTERNAL UTILITIES
    # core.calibrated_explainer and others use internal utils
    ('core', 'core.utils'): [],  # self-sibling (OK: both in core)

    # VISUALIZATION ADAPTERS
    ('viz', 'core'): [],
    ('viz', 'plugins'): [],
}
```

**Pros**:
- ✅ Minimal code changes
- ✅ Fastest to implement (1–2 hours)
- ✅ Reflects actual intended architecture
- ✅ Unblocks CI integration immediately
- ✅ Works with existing modules as-is

**Cons**:
- ❌ Doesn't address architectural tightness
- ❌ May hide deeper coupling that should be refactored later
- ❌ Doesn't prevent new violations from creeping in
- ❌ Harder to enforce stricter boundaries in future versions

**Effort**: ~2 hours (update config, test, document)

**Target Release**: v0.10.0 (immediate)

---

### **Option B: Interface-Based Decoupling (Medium Refactor, Cleaner Architecture)**

**Approach**: Create explicit "domain interfaces" or "shared contracts" modules that all packages can import from without violating boundaries.

**Rationale**:
- Reduces direct imports from sibling internals.
- Establishes clear contract boundaries (what a package exports vs. internals).
- Enables testing interfaces in isolation.

**Implementation Strategy**:

1. **Create `core/contracts.py`** (or `core/interfaces/`):
   - Defines abstract base classes / protocols for explanation strategies.
   - Exports exception types from `core.exceptions` (re-export).
   - Defines orchestrator callbacks.
   - Imports only from `core`, `schema`, `utils`.

   ```python
   # src/calibrated_explanations/core/contracts.py
   """Shared domain contracts for all packages."""

   from abc import ABC, abstractmethod
   from typing import Protocol

   # Re-export exception taxonomy (ADR-002 shared contract)
   from calibrated_explanations.core.exceptions import (
       CalibratedExplanationError,
       ValidationError,
       ConfigurationError,
       NotFittedError,
   )

   # Strategy interfaces
   class ExplanationStrategy(ABC):
       """Interface for explanation algorithms."""

       @abstractmethod
       def explain(self, x, y_pred): pass

   class CalibrationStrategy(ABC):
       """Interface for calibrators."""

       @abstractmethod
       def calibrate(self, x, y): pass

   class CalibratedExplainerState(Protocol):
       """Protocol for explainer state checks (used by calibration)."""

       @property
       def is_fitted(self) -> bool: ...

       @property
       def is_calibrated(self) -> bool: ...
   ```

2. **Update imports across siblings**:
   - `calibration/*.py`: `from core.contracts import {ExplanationStrategy, CalibrationStrategy, CalibratedExplainerError, ...}`
   - `plugins/*.py`: Same as above.
   - `cache/*.py`: Import exception types from `core.contracts`.
   - `viz/*.py`: Import from `core.contracts`.

3. **Leave `core.calibrated_explainer` internals** but type-hint against contracts:
   ```python
   # In calibration/state.py
   from core.contracts import CalibratedExplainerState, CalibratedExplainerError

   def check_calibration_state(explainer: CalibratedExplainerState) -> None:
       if not explainer.is_calibrated:
           raise CalibratedExplainerError("Not calibrated")
   ```

**Pros**:
- ✅ Explicit contracts improve maintainability
- ✅ Easier to unit test interfaces in isolation
- ✅ Prevents unintended coupling to implementation details
- ✅ Scales well: new packages can import contracts without knowing internals
- ✅ Supports ADR-001 intent: "siblings use core domain models"

**Cons**:
- ⚠️ Requires refactoring imports in ~30 files
- ⚠️ May reveal missing interfaces (design cost upfront)
- ⚠️ Introduces a new `contracts` module to maintain
- ⚠️ Type hints require proper Protocol/ABC usage

**Effort**: 8–12 hours (design interfaces, update imports, test)

**Target Release**: v0.10.1 (post-v0.10.0)

---

### **Option C: Lazy Import with TYPE_CHECKING (Maximum Purity, Gradual)**

**Approach**: Convert cross-sibling imports to runtime-only or TYPE_CHECKING blocks to break import-time cycles.

**Rationale**:
- Maintains current module structure but defers imports to function scope.
- Breaks circular import-time dependencies.
- Aligns with Stage 2 refactoring (calibrated_explainer already uses some lazy imports).

**Implementation Strategy**:

1. **In `calibration/state.py`**:
   ```python
   # Before:
   from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
   from calibrated_explanations.core.exceptions import NotFittedError

   # After:
   from typing import TYPE_CHECKING
   from calibrated_explanations.core.exceptions import NotFittedError  # Keep this

   if TYPE_CHECKING:
       from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

   def check_state(explainer: "CalibratedExplainer") -> None:
       # Import only at runtime if needed
       if not hasattr(explainer, '_calibration_data'):
           raise NotFittedError("Not calibrated")
   ```

2. **In `plugins/builtins.py`**:
   ```python
   # Before:
   from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

   # After:
   def register_plugin(explainer: "WrapCalibratedExplainer") -> None:
       # Import only when registration happens
       from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
       if isinstance(explainer, WrapCalibratedExplainer):
           ...
   ```

3. **Update linter to allow TYPE_CHECKING imports**:
   ```python
   # In check_import_graph.py, modify extract_imports to track TYPE_CHECKING blocks
   def extract_imports(file_path: Path) -> List[Tuple[str, int, bool]]:
       """Extract imports, marking TYPE_CHECKING imports."""
       tree = ast.parse(file_path.read_text())
       imports = []

       # Track which imports are in TYPE_CHECKING blocks
       type_checking_lines = set()
       for node in ast.walk(tree):
           if isinstance(node, ast.If):
               if (isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING'):
                   for item in ast.walk(node):
                       if isinstance(item, (ast.Import, ast.ImportFrom)):
                           type_checking_lines.add(item.lineno)

       # Extract imports, marking TYPE_CHECKING
       for node in ast.walk(tree):
           if isinstance(node, ast.Import):
               for alias in node.names:
                   is_type_checking = node.lineno in type_checking_lines
                   imports.append((alias.name, node.lineno, is_type_checking))
           elif isinstance(node, ast.ImportFrom):
               module = node.module or ''
               is_type_checking = node.lineno in type_checking_lines
               imports.append((module, node.lineno, is_type_checking))

       return imports

   # Then update violation check to allow TYPE_CHECKING imports
   if imported_type_checking:
       continue  # TYPE_CHECKING imports don't violate boundaries
   ```

**Pros**:
- ✅ Fixes import-time circular dependencies immediately
- ✅ Aligns with Stage 2 pattern (already in use)
- ✅ Minimal refactoring of actual logic
- ✅ Works with current code structure
- ✅ Type hints remain accurate (via string annotations)

**Cons**:
- ⚠️ Requires TYPE_CHECKING import in many files (~30)
- ⚠️ String type hints less pleasant in code
- ⚠️ Doesn't prevent runtime-time coupling
- ⚠️ Linter complexity increases
- ❌ Still allows circular runtime imports (e.g., if function calls another sibling at runtime)

**Effort**: 10–14 hours (update all files, update linter, test)

**Target Release**: v0.10.1

---

### **Option D: Architecture Refactor with Facade Layer (Comprehensive, Best Long-Term)**

**Approach**: Introduce explicit "facade" or "coordinator" modules that mediate communication between siblings, replacing direct imports.

**Rationale**:
- Cleanest long-term architecture.
- Scales well as new packages are added.
- Aligns with ADR-004 (parallel facade pattern) and ADR-006 (plugin interface).
- Enables future isolation of packages into separate distributions.

**Implementation Strategy**:

1. **Create `core/coordinator.py`** (new orchestration layer):
   ```python
   """Core coordinator: mediates cross-package communication."""

   from typing import Any, Dict, Protocol

   class CalibrationCoordinator:
       """Handles all calibration<->explanation<->core interaction."""

       def calibrate_with_state(self, explainer, calibrator, x, y) -> None:
           """Calibration state updates that core.calibrated_explainer approves."""
           # Calibration calls this instead of directly accessing explainer._calibration_data
           explainer._calibration_data = calibrator.fit(x, y)

       def get_explanation_interface(self, task_type: str):
           """Plugins query for explanation strategies through coordinator."""
           # Returns the right strategy without plugin knowing core.explain internals
           from calibrated_explanations.core.explain import feature_task
           return feature_task

   class ExceptionFacade:
       """All packages import exceptions through this facade."""
       from calibrated_explanations.core.exceptions import *  # Re-export all

   # Singleton instance
   _coordinator = CalibrationCoordinator()

   def get_coordinator() -> CalibrationCoordinator:
       return _coordinator
   ```

2. **Update import rules in linter**:
   ```python
   allowed_cross_sibling = {
       # Only calibration, explanations, plugins can import from core.coordinator
       ('calibration', 'core.coordinator'): [],
       ('explanations', 'core.coordinator'): [],
       ('plugins', 'core.coordinator'): [],
       ('cache', 'core.coordinator'): [],

       # All packages import exceptions through facade
       ('*', 'core.exceptions'): [],  # Explicitly sanctioned shared contract
   }
   ```

3. **Refactor cross-sibling calls**:
   ```python
   # OLD: calibration/state.py
   from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
   explainer._calibration_data = ...

   # NEW: calibration/state.py
   from calibrated_explanations.core.coordinator import get_coordinator
   coordinator = get_coordinator()
   coordinator.calibrate_with_state(explainer, self, x, y)
   ```

4. **Update `core/calibrated_explainer.py`** to register coordinator hooks:
   ```python
   from calibrated_explanations.core.coordinator import get_coordinator

   coordinator = get_coordinator()
   # Pass callbacks so calibration can signal state changes through the coordinator
   coordinator.register_state_callback(self._on_calibration_complete)
   ```

**Pros**:
- ✅ Cleanest architecture long-term
- ✅ Scales well: new siblings only import from coordinator
- ✅ Enables eventual package split (each sibling can be separate distribution)
- ✅ Clear separation of concerns
- ✅ Single point of policy for cross-package coordination
- ✅ Supports logging/telemetry of cross-package calls

**Cons**:
- ⚠️ Largest refactoring effort (~40+ hours)
- ⚠️ Requires careful coordinator design (upfront architecture work)
- ⚠️ Adds a new abstraction layer (coordinator) to learn
- ⚠️ Callbacks/coordinator pattern introduces complexity
- ❌ Not worth it if packages won't eventually split

**Effort**: 30–50 hours (design coordinator, refactor all call sites, migrate tests)

**Target Release**: v0.11.0 (after v0.10.x stabilization)

---

## Part 3: Recommended Hybrid Approach

**Recommendation**: Combine **Option A + Option B** for immediate and medium-term benefit.

### Phase 1: Unblock v0.10.0 (Immediate, ~2 hours)

**Action**: Apply **Option A** (pragmatic allow-listing).

1. Update `scripts/check_import_graph.py` with explicit rules for:
   - `(*,core.exceptions)`: Exception taxonomy is a shared domain contract (ADR-002).
   - `(calibration,core.calibrated_explainer)`: Explainer is orchestrator; calibration needs state checks.
   - `(calibration,core.explain.feature_task)`: Feature task is domain interface; needed for type hints.
   - `(explanations,core)`: Core domain models.
   - `(plugins,core)`: Temporary until ADR-006 plugin interface (v0.10.2).
   - `(viz,core)` and `(viz,explanations)`: Visualization adapters.

2. Run linter: expect clean pass with ~60 violations remaining (plugins, lower priority).

3. Document rationale in `improvement_docs/ADR-001-EXCEPTIONS-AND-CONTRACTS.md`:
   - Explain why exceptions are a shared contract.
   - List all allowed cross-sibling pairs.
   - Note deprecation timeline for plugin imports.

4. Wire linter into CI (`/.github/workflows/lint.yml`).

**Outcome**: v0.10.0 ships with documented, allowed cross-sibling imports. Linter enforces boundaries at CI. Release notes clarify: "ADR-001 Stage 5 complete with explicit exception and contract boundaries."

---

### Phase 2: Deepen Architecture (Post-v0.10.0, ~10–12 hours)

**Action**: Apply **Option B** (interface-based decoupling).

1. Create `core/contracts.py`:
   - Re-exports `core.exceptions`.
   - Defines `ExplanationStrategy`, `CalibrationStrategy`, `CalibratedExplainerState` protocols.
   - Imports only from `core`, `schema`, `utils`.

2. Update imports in:
   - `calibration/*.py` → import from `core.contracts` instead of internal modules.
   - `plugins/*.py` → same.
   - `cache/cache.py` → import exceptions from `core.contracts`.
   - `viz/*.py` → same.

3. Update linter allowlist:
   - Remove `(*,core.exceptions)` (now import from `core.contracts`).
   - Add `(*,core.contracts)`.
   - Keep calibration/core coupling (for now).

4. Add tests in `tests/unit/test_import_graph_enforcement.py`:
   - Verify all exception imports come from `core.contracts`.
   - Verify protocol usage (no concrete class imports from siblings).

**Outcome**: Cleaner architecture. Explicit contracts reduce accidental coupling. v0.10.1 release notes: "ADR-001 Stage 5b: Domain interface layer established; improved decoupling between calibration, plugins, and visualization packages."

---

### Phase 3: Long-Term Architecture (v0.11.0+, Optional)

**Defer**: Option D (coordinator facade layer).

- Assess whether packages actually need to split into separate distributions.
- If yes, invest in coordinator pattern.
- If no, maintain Option B (interfaces) indefinitely.

---

## Part 4: Implementation Roadmap

### Immediate Tasks (v0.10.0, ~2 hours)

- [ ] Update `scripts/check_import_graph.py` with allowed cross-sibling rules (Option A).
- [ ] Test: `python scripts/check_import_graph.py` → passes with documented violations.
- [ ] Create `improvement_docs/ADR-001-EXCEPTIONS-AND-CONTRACTS.md` (rationale document).
- [ ] Wire linter into `.github/workflows/lint.yml`.
- [ ] Update CHANGELOG.md: "ADR-001 Stage 5: Linting deployed; exceptions and core orchestrator access codified as allowed contracts."

### Follow-Up Tasks (v0.10.1, ~10–12 hours)

- [ ] Design `core/contracts.py` (exception re-exports + protocols).
- [ ] Update imports in `calibration/*.py` (5 files).
- [ ] Update imports in `plugins/*.py` (3 files).
- [ ] Update imports in `cache/cache.py` (1 file).
- [ ] Update imports in `viz/*.py` (3 files).
- [ ] Update linter allowlist for `(*,core.contracts)`.
- [ ] Add regression tests.
- [ ] Update CHANGELOG.md and release notes.

### Future (v0.11.0+, Conditional)

- [ ] Evaluate need for Option D (coordinator layer).
- [ ] If needed, design and implement.

---

## Part 5: Decision Matrix

| Criterion | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Architectural Purity** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Future Scalability** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Code Changes** | Minimal | Moderate | Moderate | Large |
| **Maintenance Cost** | Low | Low | Medium | High |
| **Unblocks CI** | ✅ | ✅ (later) | ✅ (later) | ✅ (v0.11.0) |
| **Testability** | Medium | High | High | Very High |

---

## Summary

**Recommended Approach**: Hybrid Phase 1 + Phase 2

1. **Now (v0.10.0)**: Option A (allow-list high-traffic imports, unblock CI).
2. **Next release (v0.10.1)**: Option B (introduce contracts, cleaner boundaries).
3. **Later (v0.11.0+)**: Option D only if packages genuinely need to split.

This balances immediate ship velocity (unblock v0.10.0) with long-term architecture quality (contracts prevent future coupling).

---

## Appendix: Detailed Violation Breakdown

### Top Violations by Package Pair

```
76 violations:   core.* → {utils, calibration, explanations, plugins, cache, parallel, api}
26 violations:   calibration.* → core.{exceptions, calibrated_explainer, explain.feature_task}
15 violations:   explanations.* → core.*, plugins.*
10 violations:   plugins.* → core.*, explanations.*
7 violations:   viz.* → core.*, explanations.*
4 violations:   api.* → core.exceptions, core.wrap_explainer
3 violations:   utils.* → core.*
2 violations:   cache.* → core.exceptions
2 violations:   parallel.* → core.*
1 violation:    legacy.* → core.*
```

### Files with Highest Violation Count

1. `core/calibrated_explainer.py` (25 violations) — orchestrator imports many siblings
2. `plugins/builtins.py` (7 violations) — plugin setup touches core, explanations, utils
3. `calibration/interval_learner.py` (6 violations) — calibrator imports core interfaces
4. `calibration/state.py` (6 violations) — state checks need explainer access
5. `core/calibration_metrics.py` (4 violations) — metrics compute from multiple domains
6. `api/params.py` (3 violations) — parameter validation touches core, plugins
7. `explanations/explanations.py` (3 violations) — coordination with core explain layer
8. `viz/narrative_plugin.py` (2 violations) — visualization plugin integrates with core
9. `cache/cache.py` (1 violation) — cache integration with core state

---

## Approval & Next Steps

**Recommendation to Maintainers**:

- [ ] **Approve Option A + Option B hybrid approach** for v0.10.0–v0.10.1.
- [ ] **Assign Option A tasks** to implement in next ~2 hours (unblock v0.10.0 release).
- [ ] **Schedule Option B tasks** for v0.10.1 post-release review.
- [ ] **Revisit Option D** after v0.11.0 based on distribution/packaging goals.

---

**Document Status**: Ready for maintainer review and decision.
