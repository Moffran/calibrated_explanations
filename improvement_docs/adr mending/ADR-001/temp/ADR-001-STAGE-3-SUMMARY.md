# ADR-001 Stage 3 Analysis Summary: Key Recommendations

**Date:** 2025-11-28
**Prepared for:** ADR-001 Gap Closure Plan
**Status:** ✅ Ready for v0.10.0 Implementation

---

## 🎯 Quick Summary

ADR-001 Gap #5 identifies that `calibrated_explanations.__init__` exports are **too broad**. This analysis recommends a two-release deprecation window:

- **v0.10.0:** Emit structured warnings for 13 unsanctioned symbols
- **v0.11.0:** Remove unsanctioned symbols; lock to 3-symbol sanctioned API

---

## 📋 Sanctioned vs. Unsanctioned API

### ✅ SANCTIONED (Keep in Top Level)

| Symbol | Path | Rationale |
| --- | --- | --- |
| `CalibratedExplainer` | `core.calibrated_explainer` | Core factory; primary entry point |
| `WrapCalibratedExplainer` | `core.wrap_explainer` | Wrapper factory; primary entry point |
| `transform_to_numeric` | `utils.helper` | High-level utility; public contract |

**Status:** No changes required; these remain stable.

### ❌ UNSANCTIONED (Move to Submodules)

#### Category A: Explanation Classes (5 symbols)

```
AlternativeExplanation       → from calibrated_explanations.explanations.explanation import AlternativeExplanation
FactualExplanation          → from calibrated_explanations.explanations.explanation import FactualExplanation
FastExplanation             → from calibrated_explanations.explanations.explanation import FastExplanation
AlternativeExplanations     → from calibrated_explanations.explanations import AlternativeExplanations
CalibratedExplanations      → from calibrated_explanations.explanations import CalibratedExplanations
```

**Rationale:** Domain dataclasses returned by methods, not entry points. Similar to pandas DataFrame—users don't import it at top level.

#### Category B: Discretizers (4 symbols)

```
BinaryEntropyDiscretizer    → from calibrated_explanations.utils.discretizers import BinaryEntropyDiscretizer
BinaryRegressorDiscretizer  → from calibrated_explanations.utils.discretizers import BinaryRegressorDiscretizer
EntropyDiscretizer          → from calibrated_explanations.utils.discretizers import EntropyDiscretizer
RegressorDiscretizer        → from calibrated_explanations.utils.discretizers import RegressorDiscretizer
```

**Rationale:** Internal utilities for preprocessing; not meant for direct user interaction in most workflows.

#### Category C: Calibrators (2 symbols)

```
IntervalRegressor           → from calibrated_explanations.calibration import IntervalRegressor
VennAbers                   → from calibrated_explanations.calibration import VennAbers
```

**Rationale:** Low-level domain components; advanced users only. Belong in calibration subpackage (ADR-001 Stage 1a).

**⚠️ BUG:** Current `__getattr__` uses `from ..calibration.interval_regressor` (incorrect relative path). Should be `from .calibration.interval_regressor`.

#### Category D: Visualization Namespace (1 symbol)

```
viz                         → from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter
```

**Rationale:** Entire namespace marked as "experimental"; imports heavy dependencies. Users should import specific items.

---

## 📈 Deprecation Timeline

### v0.10.0 (Current, "Deprecation Phase")

**What happens:**
- Users accessing unsanctioned symbols see structured deprecation warnings
- All code continues to work (no breaking changes)
- Sanctioned symbols remain unchanged (no warnings)
- Migration guide published

**Implementation effort:** 4–6 hours

**Checklist:**
- [ ] Create `utils/deprecation.py` helper
- [ ] Wrap all 13 unsanctioned symbols with warnings in `__getattr__`
- [ ] Fix calibration import bug
- [ ] Add 14 unit tests for warnings
- [ ] Update CHANGELOG
- [ ] Publish migration guide in docs

### v0.11.0 ("Removal Phase")

**What happens:**
- Unsanctioned symbols are removed from `__getattr__`
- Accessing them raises `AttributeError`
- Users must use submodule imports

**Breaking change:** Yes, but with 1 full release (v0.10.x patch cycle) as migration window.

---

## 🧪 Test Strategy

### v0.10.0: Deprecation Warning Tests

**File:** `tests/unit/test_package_init.py`

```python
# Should emit warnings
def test_should_emit_deprecation_for_alternative_explanation(monkeypatch):
    monkeypatch.delitem(ce.__dict__, "AlternativeExplanation", raising=False)
    with pytest.warns(DeprecationWarning, match="...v0.11.0..."):
        _ = ce.AlternativeExplanation

# Should NOT emit warnings
def test_should_not_warn_for_calibrated_explainer(monkeypatch):
    monkeypatch.delitem(ce.__dict__, "CalibratedExplainer", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = ce.CalibratedExplainer  # Should not raise
```

**Test count:** 14 new tests
- 13 × "should emit warning" (one per unsanctioned symbol)
- 3 × "should not warn" (one per sanctioned symbol)

### v0.11.0: Removal Tests

Replace deprecation tests with:

```python
# Should raise AttributeError
def test_should_raise_on_alternative_explanation():
    with pytest.raises(AttributeError, match="has no attribute 'AlternativeExplanation'"):
        ce.AlternativeExplanation
```

---

## 📝 User Migration Examples

### Migration Example 1: Explanation Classes

```python
# ❌ v0.10.0 (deprecated, emits warning)
# ✗ v0.11.0+ (raises AttributeError)
from calibrated_explanations import CalibratedExplanations

# ✅ v0.11.0+ (correct)
from calibrated_explanations.explanations import CalibratedExplanations
```

### Migration Example 2: Calibrators

```python
# ❌ v0.10.0 (deprecated)
# ✗ v0.11.0+ (raises AttributeError)
from calibrated_explanations import IntervalRegressor

# ✅ v0.11.0+ (correct)
from calibrated_explanations.calibration import IntervalRegressor
```

### Migration Example 3: Discretizers

```python
# ❌ v0.10.0 (deprecated)
# ✗ v0.11.0+ (raises AttributeError)
from calibrated_explanations import EntropyDiscretizer

# ✅ v0.11.0+ (correct)
from calibrated_explanations.utils.discretizers import EntropyDiscretizer
```

### Migration Example 4: Visualization

```python
# ❌ v0.10.0 (deprecated)
# ✗ v0.11.0+ (raises AttributeError)
from calibrated_explanations import viz
viz.plots.plot_factual(...)

# ✅ v0.11.0+ (correct)
from calibrated_explanations.viz import plots
plots.plot_factual(...)
```

---

## 🚀 Implementation Steps (v0.10.0)

1. **Create deprecation helper** (30 min)
   - File: `src/calibrated_explanations/utils/deprecation.py`
   - Function: `deprecate_public_api_symbol()`

2. **Fix calibration import bug** (10 min)
   - File: `src/calibrated_explanations/__init__.py`
   - Change: `from ..calibration` → `from .calibration`

3. **Update `__getattr__` with warnings** (1.5 hours)
   - Wrap all 13 unsanctioned symbols with deprecation calls

4. **Add unit tests** (1.5 hours)
   - 14 new tests in `tests/unit/test_package_init.py`

5. **Update CHANGELOG** (20 min)
   - Document deprecations with before/after examples

6. **Create migration guide** (45 min)
   - File: `docs/migration/api_surface_narrowing.md`
   - Include all 4 migration examples

7. **Update docs** (30 min)
   - New architecture doc: `docs/architecture/public_api.md`
   - Update README if needed

8. **Test & validate** (30 min)
   - Run full test suite
   - Verify coverage ≥88%

**Total effort:** 4–6 hours (1 iteration)

---

## ✅ Success Criteria

### v0.10.0
- ✅ All 13 unsanctioned symbols emit structured `DeprecationWarning`
- ✅ All 3 sanctioned symbols do NOT emit warnings
- ✅ Tests pass with ≥88% coverage
- ✅ Migration guide published
- ✅ CHANGELOG documents changes

### v0.11.0
- ✅ Unsanctioned symbols removed from `__getattr__`
- ✅ Accessing them raises `AttributeError`
- ✅ Internal code updated to use submodule imports
- ✅ All docs/examples use new import paths
- ✅ Breaking change clearly documented

---

## 📊 Impact Analysis

| Audience | Impact | Effort | Timeline |
| --- | --- | --- | --- |
| End users using deprecated imports | 🟡 Moderate | Low (just change imports) | Full v0.10.x cycle (2-4 mo) |
| End users using sanctioned API | ✅ None | None | N/A |
| Internal code/tests | 🟡 Moderate | Medium (grep + update) | Complete before v0.11.0 |
| Documentation | 🟡 Moderate | Medium (review all examples) | Complete before v0.11.0 |
| CI/CD | ✅ Low | Low (add deprecation test) | v0.10.0 |

**Risk level:** 🟢 LOW – Deprecation window is long, warnings are clear, migration is straightforward.

---

## 🔗 ADR Alignment

| ADR | Gap | Status | Action |
| --- | --- | --- | --- |
| ADR-001 | Gap #5: Public API overly broad (severity 6) | ✅ Addressed | Narrows surface to 3 symbols |
| ADR-001 | Gap #6: Extra namespaces lack coverage | ✅ Addressed | `viz` documented as submodule-only |
| ADR-011 | Deprecation policy | ✅ Implemented | Two-release window, structured warnings |
| ADR-001 | Overall completion | 📈 Stage 3 | 60% complete (Stages 0–2 done) |

---

## 📚 Deliverables

### Generated Documents (This Analysis)

1. **`ADR-001-STAGE-3-PUBLIC-API-NARROWING.md`** (Primary analysis)
   - Complete deprecation strategy
   - Migration guide examples
   - Test code samples
   - Success criteria

2. **`ADR-001-STAGE-3-IMPLEMENTATION-ROADMAP.md`** (Implementation guide)
   - Step-by-step instructions
   - Code snippets ready to copy/paste
   - Validation checklist
   - Commit message template

3. **`ADR-001-STAGE-3-SUMMARY.md`** (This document)
   - Executive summary
   - Quick reference tables
   - Timeline and effort estimates

### Recommended Next Actions

1. **Approve Stage 3 plan** (async review)
2. **Create feature branch** `feat/adr-001-stage-3-api-narrowing`
3. **Implement changes** following roadmap steps 1–8
4. **Create PR** with reference to this analysis
5. **Merge after Stage 2 verification** and review approval

---

## 📞 Questions & Discussion Points

- **Q: Why not remove in v0.10.0?**
  - A: ADR-011 requires two-release deprecation window. This gives users time to migrate and collect field evidence.

- **Q: Will this break existing workflows?**
  - A: Not in v0.10.0 (warnings only). In v0.11.0, yes—but migration is trivial (one-line import changes).

- **Q: What about internal tests?**
  - A: Scan codebase for deprecated imports; update before v0.11.0 release.

- **Q: Should we provide an automated migration script?**
  - A: Possible future enhancement; for now, clear documentation + examples are sufficient.

---

## 🔄 Relationship to Other Stages

```
ADR-001 Stage 0 ✅ → Stage 1a ✅ → Stage 1b ✅ → Stage 1c ✅ → Stage 2 ✅ → Stage 3 ⏳
                    (calibration)  (perf split)  (schema)      (decoupling)   (api narrow)
                                                                               ↑ YOU ARE HERE
```

**After Stage 3:**
- Package boundaries fully realigned (Stages 0–2) ✅
- Public API narrowed to sanctioned facade (Stage 3) ✅
- Remaining gaps (ADR-002+) addressed in parallel streams

---

## 📎 Related Files

**Read first:**
- `improvement_docs/ADR-gap-analysis.md` (Line 50–52: Gap #5)
- `improvement_docs/RELEASE_PLAN_v1.md` (Line 20–22: Stage 3 tracking)
- `src/calibrated_explanations/__init__.py` (Current public API)

**For implementation:**
- `ADR-001-STAGE-3-IMPLEMENTATION-ROADMAP.md` (Copy-paste code)
- `.github/tests-guidance.md` (Test policy)

**For context:**
- `improvement_docs/ADR-001-STAGE-2-COMPLETION-REPORT.md` (Recent Stage 2)
- `improvement_docs/adrs/ADR-001-STAGE-0-SCOPE-CONFIRMATION.md` (Original boundaries)

---

## 🎬 Next Steps

1. **Review this analysis** – confirm scope and timeline
2. **Follow implementation roadmap** – 8 step-by-step stages
3. **Run validation checklist** before committing
4. **Publish PR** with clear migration messaging
5. **Plan v0.11.0 removal** once v0.10.0 ships

---

**Analysis prepared by:** Copilot (v0.10.0 dev analysis)
**Status:** ✅ Ready for team review and implementation
