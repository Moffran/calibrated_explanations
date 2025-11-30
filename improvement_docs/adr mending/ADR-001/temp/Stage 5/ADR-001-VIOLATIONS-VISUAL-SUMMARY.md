# ADR-001 Cross-Sibling Violations: Visual Summary

**Generated**: 2025-11-30  
**Total Violations**: 153  
**Analysis Date**: Stage 5 Linting Assessment

---

## Quick Reference: Violation Heatmap

```
SOURCE PACKAGE    → DESTINATION PACKAGE    VIOLATIONS  STATUS
═══════════════════════════════════════════════════════════════════

CRITICAL ZONES (must address)
─────────────────────────────
core              → utils                 19          ⚠️  Core internal utils
core              → calibration           6           ⚠️  Orchestrator dependency
core              → explanations          6           ⚠️  Orchestrator dependency
core              → plugins               4           ⚠️  Plugin coordination
core              → cache                 3           ⚠️  Cache state sync
core              → parallel              2           ⚠️  Parallel coordination
core              → api                   2           ⚠️  Parameter routing

calibration       → core.exceptions       8           🔴 Shared contract
calibration       → core.calibrated_expl  6           🔴 State checks
calibration       → core.explain.*        4           🔴 Type hints (interfaces)

explanations      → core.*                8           🔴 Domain models
explanations      → plugins               4           🔴 Plugin loading

plugins           → core                  4           🔴 Strategy access
plugins           → explanations          3           🔴 Explanation loading
plugins           → core.exceptions       2           🔴 Exception raising

HIGH PRIORITY (secondary)
─────────────────────────
api               → core.exceptions       2           🟡 Exception raising
api               → core.wrap_explainer   1           🟡 Wrapper access
viz               → core                  4           🟡 Visualization adapters
viz               → explanations          2           🟡 Explanation access
utils             → core                  3           🟡 Internal utility use
cache             → core.exceptions       1           🟡 Exception handling

LOWER PRIORITY
──────────────
parallel          → core                  2           🟢 Minor
legacy            → core                  1           🟢 Deprecated path
```

---

## Architecture Map: Current State vs. Desired State

### Current State (with violations)

```
┌─────────────────────────────────────────────────────────────┐
│                   calibrated_explanations                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              core (ORCHESTRATOR)                      │   │
│  ├──────────────────────────────────┬────────────────────┤   │
│  │ calibrated_explainer             │ Other core modules │   │
│  │ (imports from siblings ×25 times)│                    │   │
│  │ ⬇️ ⬇️ ⬇️ ⬇️ ⬇️ ⬇️ ⬇️ ⬇️ ⬇️   |                    │   │
│  └──────────────────────────────────┴────────────────────┘   │
│          │              │           │        │      │         │
│          ├─────X────────┤           ├──X─────┤      │         │
│          ▼              ▼           ▼        ▼      ▼         │
│  ┌────────────┐  ┌────────────┐  ┌──────┐ ┌───┐ ┌─────────┐ │
│  │calibration │  │explanations│  │cache │ │viz│ │plugins  │ │
│  │  ×26 viol  │  │ ×15 viol   │  │×1viol│ │×7 │ │ ×10viol │ │
│  └────────────┘  └────────────┘  └──────┘ └───┘ └─────────┘ │
│         │              │            │       │          │      │
│         └──────────────┬────────────┘       │          │      │
│                  (circular)                 └──────────┘      │
│                                                                │
│  ┌────────┐  ┌──────────┐  ┌──────────────────────────────┐  │
│  │ utils  │  │   api    │  │   core.exceptions (shared)   │  │
│  │ ×3viol │  │ ×2 viol  │  │   (×57 imports from others)  │  │
│  └────────┘  └──────────┘  └──────────────────────────────┘  │
│                                                                │
└─────────────────────────────────────────────────────────────┘

Legend: X = violation, → = intended import
```

### Desired State (Option B: Interface Layer)

```
┌─────────────────────────────────────────────────────────────┐
│                   calibrated_explanations                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              core (DOMAIN & ORCHESTRATOR)            │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  core/contracts.py                                  │   │
│  │  - ExceptionFacade (re-exports core.exceptions)    │   │
│  │  - ExplanationStrategy (interface)                 │   │
│  │  - CalibrationStrategy (interface)                 │   │
│  │  - CalibratedExplainerState (protocol)             │   │
│  │                                                     │   │
│  │  Available to all siblings (no boundary violation) │   │
│  └─────────────────────────────────────────────────────┘   │
│          ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️ ⬆️          │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌──────┐ ┌───┐ ┌─────────┐ │
│  │calibration │  │explanations│  │cache │ │viz│ │plugins  │ │
│  │  (clean)   │  │  (clean)   │  │clean │ │cli│ │ (clean) │ │
│  └────────────┘  └────────────┘  └──────┘ └───┘ └─────────┘ │
│                                                                │
│  ┌────────┐  ┌──────────┐                                    │
│  │ utils  │  │   api    │                                    │
│  │ (clean)│  │ (clean)  │                                    │
│  └────────┘  └──────────┘                                    │
│                                                                │
└─────────────────────────────────────────────────────────────┘

Legend: ⬆️ = clean, allowed import path
```

---

## Decision Framework: Quick Comparison

| **Option** | **Violations Fixed** | **Effort** | **Purity** | **Scalability** | **When** |
|:---|:---:|:---:|:---:|:---:|:---|
| **A: Allow-list** | ✅ All (documented) | 2h | ⭐⭐ | ⭐⭐ | v0.10.0 NOW |
| **B: Contracts** | ✅ All (clean) | 10h | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | v0.10.1 NEXT |
| **C: Lazy imports** | ✅ All (defer) | 12h | ⭐⭐⭐ | ⭐⭐⭐ | v0.10.1 ALT |
| **D: Coordinator** | ✅ All (refactored) | 50h | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | v0.11.0 MAYBE |

---

## Top 10 Problem Files

```
RANK  FILE                                 VIOLATIONS  PATTERN
────────────────────────────────────────────────────────────────
 1.   core/calibrated_explainer.py                25  Orchestrator hub
 2.   plugins/builtins.py                         7   Plugin coordination
 3.   calibration/interval_learner.py             6   State checking
 4.   calibration/state.py                        6   State management
 5.   core/calibration_metrics.py                 4   Cross-domain metrics
 6.   api/params.py                               3   Parameter validation
 7.   explanations/explanations.py                3   Explanation dispatch
 8.   core/explain/orchestrator.py                3   Core dispatcher
 9.   viz/narrative_plugin.py                     2   Visualization integration
10.   cache/cache.py                              1   Cache state sync
```

---

## Violation Patterns: Root Causes

### Pattern 1️⃣: Exception Taxonomy (57 occurrences)
- **Root Cause**: Centralized ADR-002 exceptions in `core.exceptions`
- **Solution**: Option A (allow) or Option B (re-export via contracts)
- **Severity**: 🔴 HIGH (exception raising is foundational)

### Pattern 2️⃣: Orchestrator Coupling (25 occurrences)
- **Root Cause**: `CalibratedExplainer` coordinates all packages
- **Solution**: Option A (allow) → eventually Option D (coordinator)
- **Severity**: 🟡 HIGH (but intentional design)

### Pattern 3️⃣: Domain Interfaces (12 occurrences)
- **Root Cause**: Siblings need feature_task and strategy interfaces
- **Solution**: Option B (contracts exports interfaces)
- **Severity**: 🟡 MEDIUM (fixable via contracts layer)

### Pattern 4️⃣: Plugin Coordination (10 occurrences)
- **Root Cause**: Plugins load explanations and strategies at runtime
- **Solution**: Option A (temp) → Option D (plugin interface v0.10.2)
- **Severity**: 🟡 MEDIUM (temporary until ADR-006)

### Pattern 5️⃣: Visualization Adapters (7 occurrences)
- **Root Cause**: Viz converts core models to plot specs
- **Solution**: Option A (allow) or Option B (contracts)
- **Severity**: 🟢 LOW (adapters are expected to cross boundaries)

### Pattern 6️⃣: Internal Utilities (19 occurrences)
- **Root Cause**: `core.utils` used by `core.calibrated_explainer`
- **Solution**: Move utilities to `core/utils/` or allow internally
- **Severity**: 🟢 LOW (acceptable within core)

---

## Implementation Timeline (Hybrid Approach)

```
NOW              v0.10.0          v0.10.1          v0.11.0+
│                  │                 │                 │
├─ Analysis ✅     ├─ Option A      ├─ Option B      ├─ Option D?
│ (done)           │ (2h effort)    │ (10h effort)   │ (50h effort)
│                  │                │                │
│                  ├─ Linter pass  ├─ Contracts     ├─ Coordinator
│                  ├─ CI ready     ├─ Clean imports ├─ Factory pattern
│                  ├─ v0.10.0 GA   ├─ v0.10.1 GA   └─ v0.11.0+ GA
│                  │ (unblock)     │ (polish)
└──────────────────┴───────────────┴────────────────────────────
  
  Recommended: Hybrid Phase 1 + Phase 2
  Later: Option D only if need to split packages
```

---

## Quick Start: What to Do Next

### If You Approve Option A (Immediate)

1. **Edit** `scripts/check_import_graph.py`
   - Update `allowed_cross_sibling` dict (add ~8 rules)
   - Keep all existing violations documented

2. **Test**
   ```bash
   python scripts/check_import_graph.py
   # Should pass with 0 violations (or documented ones only)
   ```

3. **Document**
   - Create `improvement_docs/ADR-001-EXCEPTIONS-AND-CONTRACTS.md`
   - Explain each allow-list rule

4. **CI Integration**
   - Wire into `.github/workflows/lint.yml`
   - Add check to PR template

### If You Approve Option A + Option B (Recommended)

**Same as above, PLUS:**

1. **Post-v0.10.0**: Create `core/contracts.py` (1–2 hours design)
2. **Update imports** in siblings (8–10 hours coding + testing)
3. **Re-run linter**: Should have fewer violations (cleaner boundaries)

---

## Glossary

- **Cross-sibling import**: Package A (e.g., `calibration`) imports from Package B (e.g., `core`), where both are at the same level in hierarchy.
- **Domain contract**: Shared interface/type that multiple packages depend on (e.g., exceptions, protocols).
- **Facade pattern**: Intermediate module that re-exports from multiple places to provide a single import point.
- **TYPE_CHECKING import**: Import wrapped in `if TYPE_CHECKING:` block; used for type hints, not runtime.
- **Coordinator pattern**: Central mediator module that handles cross-package communication (like a message bus).

---

## Questions & Feedback

For detailed rationale and trade-offs, see the companion document:
**`improvement_docs/ADR-001-CROSS-SIBLING-REFACTORING-OPTIONS.md`**

---

**Status**: Ready for maintainer review and decision (Option A, B, C, D, or hybrid).
