# ADR-001 Stage 3: Visual Gap Closure & Architecture Summary

**Created:** 2025-11-28
**Stage:** 3 of 5
**Overall Progress:** 60% (Stages 0–2 complete; 3-5 in flight)

---

## 🗺️ Current State vs. Target State

### Current State (v0.10.0 dev)

```
calibrated_explanations/
├── __init__.py
│   ├── __all__ = ["CalibratedExplainer", "WrapCalibratedExplainer", "transform_to_numeric"]
│   │
│   └── __getattr__(name) → lazy imports
│       ├── ✅ CalibratedExplainer         (sanctioned)
│       ├── ✅ WrapCalibratedExplainer     (sanctioned)
│       ├── ✅ transform_to_numeric        (sanctioned)
│       │
│       ├── ❌ AlternativeExplanation      (unsanctioned)
│       ├── ❌ FactualExplanation          (unsanctioned)
│       ├── ❌ FastExplanation             (unsanctioned)
│       ├── ❌ AlternativeExplanations     (unsanctioned)
│       ├── ❌ CalibratedExplanations      (unsanctioned)
│       │
│       ├── ❌ BinaryEntropyDiscretizer    (unsanctioned)
│       ├── ❌ BinaryRegressorDiscretizer  (unsanctioned)
│       ├── ❌ EntropyDiscretizer          (unsanctioned)
│       ├── ❌ RegressorDiscretizer        (unsanctioned)
│       │
│       ├── ❌ IntervalRegressor           (unsanctioned, buggy path)
│       ├── ❌ VennAbers                   (unsanctioned, buggy path)
│       │
│       └── ❌ viz (entire namespace)      (unsanctioned)
│
└── PROBLEM: 13 unsanctioned symbols pollute public API
              Users confused about which imports are "official"
              ADR-001 Gap #5 (severity 6)
```

### Target State (v0.11.0+)

```
calibrated_explanations/
├── __init__.py
│   ├── __all__ = ["CalibratedExplainer", "WrapCalibratedExplainer", "transform_to_numeric"]
│   │
│   └── __getattr__(name) → lazy imports
│       ├── ✅ CalibratedExplainer         (sanctioned → top-level)
│       ├── ✅ WrapCalibratedExplainer     (sanctioned → top-level)
│       ├── ✅ transform_to_numeric        (sanctioned → top-level)
│       │
│       └── ❌ AttributeError for all others
│
├── core/
│   ├── calibrated_explainer.py
│   └── wrap_explainer.py
│
├── explanations/
│   ├── __init__.py → AlternativeExplanations, CalibratedExplanations
│   ├── explanation.py → AlternativeExplanation, FactualExplanation, FastExplanation
│
├── utils/
│   ├── discretizers.py → EntropyDiscretizer, RegressorDiscretizer, ...
│   └── helper.py → transform_to_numeric
│
├── calibration/
│   ├── interval_regressor.py → IntervalRegressor
│   ├── venn_abers.py → VennAbers
│
├── viz/
│   ├── __init__.py → (users import from submodule)
│   ├── plots.py
│   └── plotspec.py
│
└── SOLUTION: Clear separation of concerns
              Users know exactly where to find each symbol
              ADR-001 Gap #5 RESOLVED ✅
```

---

## 📊 Migration Flow

```
User Code (Current v0.10.0-dev)
         ↓
    deprecation warning emitted
         ↓
    "See migration guide"
         ↓
User reads docs/migration/api_surface_narrowing.md
         ↓
User updates imports:
  ❌ from calibrated_explanations import X
  ✅ from calibrated_explanations.submodule import X
         ↓
User runs tests ✅
         ↓
v0.11.0 released
         ↓
User can upgrade (old imports now raise AttributeError if not updated)
```

---

## 🔄 Version Timeline

```
v0.10.0 (Current)
├─ ✅ Deprecation warnings active
├─ ✅ All unsanctioned symbols work (with warnings)
├─ ✅ Migration guide published
├─ ✅ Users have ~2-4 months to update
└─ Full patch cycle: v0.10.0, v0.10.1, v0.10.2, v0.10.3, ...

        ↓ (full minor release cycle ~2-4 months)

v0.11.0 (Target)
├─ ❌ Unsanctioned symbols removed
├─ ✅ Only sanctioned symbols in __getattr__
├─ ✅ All internal code updated
├─ ⛔ Breaking change (AttributeError if users didn't migrate)
└─ Clear messaging: "See migration guide from v0.10.0"

        ↓

v1.0.0 (Future)
├─ ✅ Stable public API
├─ ✅ ADR-001 fully implemented
└─ Sanctioned API locked
```

---

## 🎯 Symbol Classification Matrix

```
                      ┌─────────────────────────────────────────────┐
                      │ SANCTIONED (Top-Level Only)                 │
                      ├─────────────────────────────────────────────┤
                      │ ✅ CalibratedExplainer                      │
                      │ ✅ WrapCalibratedExplainer                  │
                      │ ✅ transform_to_numeric                     │
                      └─────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │ UNSANCTIONED (Submodule Imports Only)                               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Category            │ Symbols                │ Submodule Path       │
    ├─────────────────────┼────────────────────────┼──────────────────────┤
    │ Explanation Classes │ • AlternativeExplan   │ .explanations         │
    │                     │ • FactualExplan       │ .explanation          │
    │                     │ • FastExplan          │ .explanation          │
    │                     │ • AlternativeExplans  │ .explanations         │
    │                     │ • CalibratedExplans   │ .explanations         │
    ├─────────────────────┼────────────────────────┼──────────────────────┤
    │ Discretizers        │ • EntropyDiscretizer  │ .utils.discretizers   │
    │                     │ • RegressorDiscr      │ .utils.discretizers   │
    │                     │ • BinaryEntropyDisc   │ .utils.discretizers   │
    │                     │ • BinaryRegressorDisc │ .utils.discretizers   │
    ├─────────────────────┼────────────────────────┼──────────────────────┤
    │ Calibrators         │ • IntervalRegressor   │ .calibration          │
    │                     │ • VennAbers           │ .calibration          │
    ├─────────────────────┼────────────────────────┼──────────────────────┤
    │ Visualization       │ • viz (entire ns)     │ .viz (import items)   │
    └─────────────────────┴────────────────────────┴──────────────────────┘
```

---

## 🚀 Implementation Timeline

```
ITERATION 1: Preparation & Testing
├─ Mon-Wed: Create deprecation helper + update __getattr__
├─ Thu: Add unit tests (14 new tests)
└─ Fri: Verify no regressions (coverage ≥88%)

ITERATION 2: Documentation & Release
├─ Mon-Tue: Create migration guide + architecture docs
├─ Wed: Update CHANGELOG + README
├─ Thu: Code review + address feedback
└─ Fri: Merge to main + release v0.10.0-rc1

AFTER v0.10.0 RELEASE:
├─ Full patch cycle: v0.10.1, v0.10.2, v0.10.3, ...
├─ Users have ~2-4 months to migrate
├─ Monitor deprecation warnings in telemetry
└─ Gather feedback on migration difficulty

WHEN READY FOR v0.11.0:
├─ Remove all unsanctioned branches from __getattr__
├─ Update tests: "deprecation" → "AttributeError"
├─ Scan & update internal code
├─ Release v0.11.0 with clear breaking change messaging
└─ Archive deprecation.py (no longer needed)
```

---

## 📈 Effort Breakdown

```
┌────────────────────────────────────────────┐
│ v0.10.0 Implementation (5 hours total)     │
├────────────────────────────────────────────┤
│ Create deprecation helper       30 min  ████
│ Fix calibration import bug      10 min  ██
│ Update __getattr__ (13x)        90 min  ██████████████
│ Add unit tests (14x)            90 min  ██████████████
│ CHANGELOG + docs                60 min  █████████
│ Full test suite validation      30 min  ████
├────────────────────────────────────────────┤
│ TOTAL EFFORT: ~5 hours (1 iteration)       │
└────────────────────────────────────────────┘

Complexity: 🟡 Medium
  - Straightforward deprecation pattern
  - No architectural changes
  - Clear test cases
  - Well-documented (this analysis)

Risk: 🟢 Low
  - No breaking changes in v0.10.0
  - Users have full release cycle to migrate
  - Migration is trivial (import path changes only)
  - Rollback simple (revert deprecation warnings)
```

---

## 🧪 Test Coverage Summary

```
Current Tests (test_package_init.py):
├─ test_interval_regressor_lazy_import ✅
├─ test_venn_abers_lazy_import ✅
└─ test_unknown_attribute_raises ✅

NEW Tests (v0.10.0):
├─ TestDeprecatedPublicApiSymbols (13 tests)
│  ├─ test_should_emit_deprecation_for_alternative_explanation ✅
│  ├─ test_should_emit_deprecation_for_factual_explanation ✅
│  ├─ test_should_emit_deprecation_for_fast_explanation ✅
│  ├─ test_should_emit_deprecation_for_alternative_explanations ✅
│  ├─ test_should_emit_deprecation_for_calibrated_explanations ✅
│  ├─ test_should_emit_deprecation_for_entropy_discretizer ✅
│  ├─ test_should_emit_deprecation_for_regressor_discretizer ✅
│  ├─ test_should_emit_deprecation_for_binary_entropy_discretizer ✅
│  ├─ test_should_emit_deprecation_for_binary_regressor_discretizer ✅
│  ├─ test_should_emit_deprecation_for_interval_regressor ✅
│  ├─ test_should_emit_deprecation_for_venn_abers ✅
│  └─ test_should_emit_deprecation_for_viz_namespace ✅
│
└─ TestSanctionedSymbolsNoWarnings (3 tests)
   ├─ test_should_not_warn_for_calibrated_explainer ✅
   ├─ test_should_not_warn_for_wrap_calibrated_explainer ✅
   └─ test_should_not_warn_for_transform_to_numeric ✅

Total New Tests: 16
Total Test Lines: ~200
Coverage: ≥95% (deprecation module)
```

---

## 📚 Document Artifacts Generated

This analysis produced 4 comprehensive documents:

```
improvement_docs/adrs/
├── ADR-001-STAGE-3-PUBLIC-API-NARROWING.md (Primary Analysis)
│   ├─ Complete strategy
│   ├─ Migration examples (all 4 categories)
│   ├─ Test code samples
│   └─ Success criteria (v0.10.0 & v0.11.0)
│
├── ADR-001-STAGE-3-IMPLEMENTATION-ROADMAP.md (How-To)
│   ├─ 8 step-by-step stages
│   ├─ Copy-paste code templates
│   ├─ Validation checklist
│   └─ Commit message template
│
├── ADR-001-STAGE-3-SUMMARY.md (Executive Summary)
│   ├─ Quick overview
│   ├─ Impact analysis
│   ├─ ADR alignment table
│   └─ Next steps
│
└── ADR-001-STAGE-3-QUICK-REFERENCE.md (Cheat Sheet)
    ├─ Symbol disposition table
    ├─ 5-minute implementation template
    ├─ Common mistakes to avoid
    └─ Decision trees
```

---

## 🎓 Knowledge Transfer

### For Developers Implementing Stage 3

1. **Start with:** `ADR-001-STAGE-3-QUICK-REFERENCE.md` (5 min read)
2. **Then read:** `ADR-001-STAGE-3-IMPLEMENTATION-ROADMAP.md` (20 min)
3. **Implement:** Follow 8 step-by-step stages (4–6 hours)
4. **Reference:** Use code templates from roadmap
5. **Validate:** Run checklist before committing

### For Reviewers

1. **Context:** Read `ADR-001-STAGE-3-SUMMARY.md` (10 min)
2. **Details:** Review `ADR-001-STAGE-3-PUBLIC-API-NARROWING.md` (20 min)
3. **Check PR:** Verify against "Success Criteria" section
4. **Approve if:** All 14 tests passing + docs published + no warnings for sanctioned symbols

### For Maintainers (v0.11.0 Planning)

1. **Tracking:** Reference this Stage 3 analysis as scope
2. **Timeline:** Plan removal for v0.11.0 (after full v0.10.x cycle)
3. **Communication:** Link v0.11.0 release notes to v0.10.0 migration guide
4. **Internal:** Create grep-based scan to find all deprecated imports before removal

---

## ✨ Key Highlights

| Aspect | Highlights |
| --- | --- |
| **Simplicity** | Only 3 sanctioned symbols remain; clear submodule paths for all others |
| **User Experience** | Clear, actionable deprecation warnings point to specific migration path |
| **Timeline** | Full release cycle (v0.10.x) for users to migrate; no surprise breakage |
| **Testing** | 14 new unit tests validate both warning emission and no-warning cases |
| **Documentation** | Migration guide with real examples for all 4 symbol categories |
| **ADR Alignment** | Directly addresses ADR-001 Gap #5 (severity 6); implements ADR-011 deprecation |
| **Risk** | 🟢 Low – deprecation pattern is proven; migration is trivial |
| **Effort** | ~5 hours implementation + ongoing maintenance (minimal) |

---

## 🔗 ADR-001 Stages Overview

```
Stage 0 (Dec 2024) ✅
└─ Confirm boundaries and scope

Stage 1a-1c (Jan-Feb 2025) ✅
├─ Calibration extracted to top-level package
├─ Cache/parallel split into perf namespace
└─ Schema validation package created

Stage 2 (Feb-Mar 2025) ✅
└─ Decouple cross-sibling imports in CalibratedExplainer
   (14 module-level imports → lazy/TYPE_CHECKING)

Stage 3 (NOW) ⏳
└─ Narrow public API surface (13 unsanctioned → submodule-only)
   v0.10.0 deprecation warnings
   v0.11.0 removal
   ← YOU ARE HERE

Stage 4-5 (Future)
├─ Documentation of remaining namespaces
└─ Final ADR-001 sign-off
```

---

## 📞 FAQ Quick Answers

**Q: Why remove in two releases?**
A: ADR-011 requires two-release deprecation window. Gives users ~2-4 months to migrate.

**Q: Will users' code break?**
A: Not in v0.10.0. In v0.11.0, only if they didn't update imports (simple fix: change import path).

**Q: Are sanctioned symbols safe?**
A: Yes. Zero changes to CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric.

**Q: What about internal tests?**
A: Scan before v0.11.0; update any tests using unsanctioned imports to use submodule paths.

**Q: Should we auto-migrate user code?**
A: Not in scope. Clear deprecation warnings + migration guide are sufficient.

---

**Status:** ✅ ANALYSIS COMPLETE – Ready for implementation review and approval
