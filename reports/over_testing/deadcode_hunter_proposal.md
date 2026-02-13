# DEADCODE-HUNTER PROPOSAL: Dead/Non-Contributing Source Code Analysis

## 1. Dead Private Methods

The private method analysis found **349 library definitions**. Results:

| Pattern | Count | Description |
|---------|-------|-------------|
| Consistent (Internal Only) | 348 | Properly used within src/ -- **no action needed** |
| Pattern 2 (Local Test Helper) | 2 | Test-only helpers in test files -- normal |
| **Pattern 3 (Completely Dead)** | **1** | Defined but never called anywhere |

### Single Dead Method
- **`_missing_`** in [explanations/reject.py:41](src/calibrated_explanations/explanations/reject.py#L41)
- This is a `classmethod` on `RejectPolicy` enum handling deprecated policy names
- 0 src usages, 0 test usages
- **However**: `_missing_` is a special Python enum method called automatically when an unknown value is passed to the enum constructor. It is NOT dead code -- it's invoked by Python's enum metaclass, invisible to static analysis.
- **Verdict: DO NOT REMOVE** -- false positive from static analysis

### Summary: No dead private methods to remove. The codebase is clean.

---

## 2. Large Uncovered Code Blocks (>= 50 lines)

Analysis of `gaps.csv` identified **100+ blocks >= 50 lines**. Here are the top 10:

| File | Lines | Size | Classification |
|------|-------|------|----------------|
| `core/explain/feature_task.py` | 119-631 | 513 | **Untested production code** -- core explain algorithm, called from orchestrator |
| `plotting.py` | 1029-1486 | 458 | **Untested production code** -- plot rendering functions, reachable from public API |
| `plugins/builtins.py` | 450-856 | 407 | **Untested production code** -- builtin plugin implementations, loaded via registry |
| `core/explain/orchestrator.py` | 325-723 | 399 | **Untested production code** -- explain orchestration, core path |
| `core/explain/_legacy_explain.py` | 44-367 | 324 | **Conditionally reachable** -- legacy path, imported from orchestrator.py and sequential.py |
| `core/narrative_generator.py` | 525-833 | 309 | **Untested production code** -- narrative generation, actively used |
| `plugins/builtins.py` | 1223-1500 | 278 | **Untested production code** -- more builtin plugin code |
| `core/explain/_computation.py` | 360-611 | 252 | **Untested production code** -- core computation, called from feature_task |
| `core/explain/_feature_filter.py` | 165-413 | 249 | **Untested production code** -- feature filtering logic |
| `core/narrative_generator.py` | 199-443 | 245 | **Untested production code** -- more narrative generation |

### Key Finding: Almost ALL large gaps are UNTESTED PRODUCTION CODE, not dead code.

These modules are all reachable from the public API through:
- `CalibratedExplainer.explain_factual()` -> orchestrator -> feature_task -> _computation
- `CalibratedExplainer.explain_alternative()` -> orchestrator -> feature_task
- `plotting.plot()` -> plotting functions
- Plugin registry -> builtins

**These should NOT be removed. They need tests.**

---

## 3. Lazy Import Analysis

`__init__.py` uses `__getattr__` for lazy loading of:
- **Sanctioned**: `CalibratedExplainer`, `WrapCalibratedExplainer`, `transform_to_numeric`
- **Deprecated** (with warnings): `AlternativeExplanation`, `FactualExplanation`, `FastExplanation`, `AlternativeExplanations`, `CalibratedExplanations`, discretizers, `IntervalRegressor`, `VennAbers`, `plotting`, `viz`

This means all deprecated symbols are still reachable through `from calibrated_explanations import X`. Code supporting these deprecated paths must be kept until v0.11.0 (per ADR-011).

Specifically:
- `plotting.py` (458 uncovered lines) -- reachable via deprecated `from calibrated_explanations import plotting`
- `legacy/plotting.py` -- reachable indirectly through plotting.py
- All explanation classes -- reachable through deprecated imports
- `_legacy_explain.py` -- reachable through orchestrator fallback chain

**No "hidden dead code" found through lazy imports.**

---

## 4. Code Serving Only Test Infrastructure

### testing/ module
- `testing/__init__.py` -- package init, minimal
- `testing/parity_compare.py` (77 uncovered lines at 31-107) -- parity testing utility
- **Verdict**: This module exists explicitly for test infrastructure. It's legitimate test tooling, not dead production code. However, parity_compare.py's 77 uncovered lines suggest the parity tests may not be running in the current suite.

### test_force_mark_lines_for_coverage
- Already identified by pruner -- 991 lines of coverage padding
- **This is the single biggest instance of "src code (artificially) serving test infrastructure"**

### Debug/introspection code
No dedicated debug modules found. The `logging.py` module follows STD-005 standard and is legitimate.

---

## 5. Estimated Coverage Impact of Proposed Changes

| Action | Coverage Impact |
|--------|----------------|
| Remove `_missing_` (reject.py) | **N/A -- don't remove** (false positive) |
| Remove testing/parity_compare.py uncovered code | 0 lines (already uncovered) |
| Remove coverage-padding test (pruner's domain) | -991 artificial lines |

### Total removable dead src code: **NONE identified**

The codebase is remarkably clean. All 349 private methods are properly used internally. The large uncovered blocks are production code that needs tests, not dead code.

---

## 6. What Cannot Be Determined Without Deeper Investigation

1. **Dynamic plugin loading paths**: Plugins loaded via entry points may exercise code paths invisible to static analysis. Need to test with all plugins enabled.
2. **Python version-conditional code**: Some `try/except ImportError` blocks may only execute on specific Python versions (3.10 vs 3.13). Need CI matrix coverage aggregation.
3. **Optional dependency paths**: Code guarded by `if HAS_MATPLOTLIB`, `if HAS_SHAP` etc. may appear dead without those dependencies installed.
4. **Notebook/evaluation usage**: Files in `evaluation/` and `notebooks/` may exercise src code not covered by the test suite. These weren't analyzed for dead code cross-referencing.

---

## 7. Recommendation

**The deadcode-hunter has little to report.** The source code is well-maintained with:
- 0 truly dead private methods
- 0 dead modules
- Active production code in all large uncovered blocks

The real opportunity is not removing dead code but **adding tests for the ~5,000+ uncovered production code lines** in core explanation, plotting, and plugin modules. This is the process-architect's domain, not the deadcode-hunter's.
