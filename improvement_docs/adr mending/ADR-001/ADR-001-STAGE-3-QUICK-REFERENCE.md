# ADR-001 Stage 3: Quick Reference Card

**Print this or bookmark for implementation reference.**

---

## 🎯 One-Liner

Narrow `calibrated_explanations.__init__` from 16 exported symbols to 3 sanctioned ones, using v0.10.0 deprecation warnings + v0.11.0 removal.

---

## 📊 Symbol Disposition

| Symbol | Category | v0.10.0 | v0.11.0 | New Import Path |
| --- | --- | --- | --- | --- |
| **CalibratedExplainer** | ✅ Sanctioned | Keep | Keep | Top-level only |
| **WrapCalibratedExplainer** | ✅ Sanctioned | Keep | Keep | Top-level only |
| **transform_to_numeric** | ✅ Sanctioned | Keep | Keep | Top-level only |
| **AlternativeExplanation** | ❌ Unsanctioned | Warn | Remove | `.explanations.explanation` |
| **FactualExplanation** | ❌ Unsanctioned | Warn | Remove | `.explanations.explanation` |
| **FastExplanation** | ❌ Unsanctioned | Warn | Remove | `.explanations.explanation` |
| **AlternativeExplanations** | ❌ Unsanctioned | Warn | Remove | `.explanations` |
| **CalibratedExplanations** | ❌ Unsanctioned | Warn | Remove | `.explanations` |
| **BinaryEntropyDiscretizer** | ❌ Unsanctioned | Warn | Remove | `.utils.discretizers` |
| **BinaryRegressorDiscretizer** | ❌ Unsanctioned | Warn | Remove | `.utils.discretizers` |
| **EntropyDiscretizer** | ❌ Unsanctioned | Warn | Remove | `.utils.discretizers` |
| **RegressorDiscretizer** | ❌ Unsanctioned | Warn | Remove | `.utils.discretizers` |
| **IntervalRegressor** | ❌ Unsanctioned | Warn | Remove | `.calibration` |
| **VennAbers** | ❌ Unsanctioned | Warn | Remove | `.calibration` |
| **viz** | ❌ Unsanctioned | Warn | Remove | `.viz` (submodule items) |

---

## 🔧 Implementation Checklist

### v0.10.0

- [ ] Create `src/calibrated_explanations/utils/deprecation.py`
  - Function: `deprecate_public_api_symbol(symbol_name, current_import, recommended_import, removal_version, extra_context)`
  
- [ ] Fix bug in `src/calibrated_explanations/__init__.py`
  - Line ~87: `from ..calibration.interval_regressor` → `from .calibration.interval_regressor`
  
- [ ] Update `__getattr__` for each unsanctioned symbol
  - Wrap with: `deprecate_public_api_symbol(...)`
  - Pattern: Call helper before returning module/value
  
- [ ] Add tests to `tests/unit/test_package_init.py`
  - 13 × `test_should_emit_deprecation_for_*`
  - 3 × `test_should_not_warn_for_*`
  
- [ ] Update `CHANGELOG.md`
  - Section: "Deprecations"
  - List all 13 symbols with before/after imports
  
- [ ] Create `docs/migration/api_surface_narrowing.md`
  - Include all 4 migration examples (explanations, discretizers, calibrators, viz)
  
- [ ] Create `docs/architecture/public_api.md`
  - Document sanctioned symbols and submodule import paths
  
- [ ] Run tests
  ```bash
  pytest tests/unit/test_package_init.py -v
  pytest tests/unit/ --cov=src/calibrated_explanations --cov-fail-under=88
  ```

### v0.11.0

- [ ] Remove all unsanctioned branches from `__getattr__`
- [ ] Keep only sanctioned symbols
- [ ] Update `__all__` to: `["CalibratedExplainer", "WrapCalibratedExplainer", "transform_to_numeric"]`
- [ ] Update tests: replace "deprecation" tests with "raises AttributeError" tests
- [ ] Audit internal code: `grep -r "from calibrated_explanations import"` and update

---

## 🚀 5-Minute Implementation Template

### File 1: Create deprecation helper

```python
# src/calibrated_explanations/utils/deprecation.py
import warnings
from typing import Optional

def deprecate_public_api_symbol(
    symbol_name: str,
    current_import: str,
    recommended_import: str,
    removal_version: str = "v0.11.0",
    extra_context: Optional[str] = None,
) -> None:
    """Emit structured deprecation warning."""
    message = (
        f"\n{symbol_name!r} imported from top level is deprecated and will be removed in {removal_version}.\n"
        f"  ❌ DEPRECATED: {current_import}\n"
        f"  ✓  RECOMMENDED: {recommended_import}\n"
    )
    if extra_context:
        message += f"\n  Details: {extra_context}\n"
    warnings.warn(message.rstrip(), category=DeprecationWarning, stacklevel=3)
```

### File 2: Update `__getattr__` (example for one symbol)

```python
# src/calibrated_explanations/__init__.py

if name == "CalibratedExplanations":
    from .utils.deprecation import deprecate_public_api_symbol
    deprecate_public_api_symbol(
        "CalibratedExplanations",
        "from calibrated_explanations import CalibratedExplanations",
        "from calibrated_explanations.explanations import CalibratedExplanations",
        extra_context="Explanation dataclasses are domain objects; import from the submodule.",
    )
    module = importlib.import_module(f"{__name__}.explanations.explanations")
    value = getattr(module, "CalibratedExplanations")
    globals()[name] = value
    return value
```

### File 3: Add tests

```python
# tests/unit/test_package_init.py

def test_should_emit_deprecation_for_calibrated_explanations(monkeypatch):
    import warnings
    import calibrated_explanations as ce
    monkeypatch.delitem(ce.__dict__, "CalibratedExplanations", raising=False)
    with pytest.warns(DeprecationWarning, match="CalibratedExplanations.*v0.11.0"):
        _ = ce.CalibratedExplanations

def test_should_not_warn_for_calibrated_explainer(monkeypatch):
    import warnings
    import calibrated_explanations as ce
    monkeypatch.delitem(ce.__dict__, "CalibratedExplainer", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = ce.CalibratedExplainer  # Should not raise
```

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Impact | Fix |
| --- | --- | --- |
| Forget to call deprecation helper | Warnings not emitted; users don't know | Check every unsanctioned branch |
| Emit warnings for sanctioned symbols | Breaks backward compat; confuses users | Verify sanctioned symbols unchanged |
| Wrong import path in warning | Users get wrong migration path | Test each migration example |
| Relative import path bug (..calibration) | Import fails at runtime | Use `.calibration` not `..calibration` |
| Forget to test that warnings emit | Tests pass but warnings don't work | Run: `pytest -W error::DeprecationWarning` |
| Update CHANGELOG but not docs | Users see warning but can't find guide | Check both CHANGELOG and docs/ |

---

## 🧪 Test Commands

```bash
# Run deprecation tests only
pytest tests/unit/test_package_init.py::TestDeprecatedPublicApiSymbols -v

# Run "no warnings" tests
pytest tests/unit/test_package_init.py::TestSanctionedSymbolsNoWarnings -v

# Run all init tests
pytest tests/unit/test_package_init.py -v

# Run with coverage
pytest tests/unit/test_package_init.py --cov=src/calibrated_explanations.utils.deprecation --cov-fail-under=88

# Full suite check
pytest tests/unit/ --cov=src/calibrated_explanations --cov-fail-under=88
```

---

## 📝 Migration Path Summary

**Before (deprecated):**
```python
from calibrated_explanations import CalibratedExplanations, EntropyDiscretizer, IntervalRegressor, viz
```

**After (v0.11.0 required):**
```python
from calibrated_explanations import CalibratedExplainer  # Only sanctioned
from calibrated_explanations.explanations import CalibratedExplanations
from calibrated_explanations.utils.discretizers import EntropyDiscretizer
from calibrated_explanations.calibration import IntervalRegressor
from calibrated_explanations.viz import plots
```

---

## 🔗 Key Files

| File | Purpose | Action |
| --- | --- | --- |
| `src/calibrated_explanations/__init__.py` | Main API surface | Wrap unsanctioned symbols with warnings |
| `src/calibrated_explanations/utils/deprecation.py` | Deprecation helper | CREATE NEW |
| `tests/unit/test_package_init.py` | Init tests | Add 14 deprecation tests |
| `CHANGELOG.md` | Release notes | Add deprecation section |
| `docs/migration/api_surface_narrowing.md` | User guide | CREATE NEW |
| `docs/architecture/public_api.md` | Architecture doc | CREATE NEW |

---

## 📞 Decision Tree

**Q: Should I deprecate symbol X?**

```
Is X one of: CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric?
  ├─ YES → Keep it in top level, NO warnings
  └─ NO  → Deprecate with warning, move to submodule
```

**Q: What import path should I recommend?**

```
What is the symbol's location?
  ├─ Explanation class → .explanations or .explanations.explanation
  ├─ Discretizer       → .utils.discretizers
  ├─ Calibrator        → .calibration
  └─ Viz               → .viz
```

**Q: When should I remove it?**

```
→ v0.11.0 (full minor release after v0.10.0)
→ Wait full v0.10.x patch cycle before removal
```

---

## ⏱️ Time Estimates

| Task | Time | Notes |
| --- | --- | --- |
| Deprecation helper (File 1) | 30 min | Copy-paste template, adjust message |
| Fix calibration bug | 10 min | One-line change |
| Update `__getattr__` | 1.5 hr | 13 symbols × 7 min each |
| Add tests (File 3) | 1.5 hr | 14 tests × 6 min each |
| CHANGELOG + docs | 1 hr | Use template from roadmap |
| Test suite run | 30 min | Full pytest cycle |
| **Total** | **~5 hours** | One sprint per team |

---

## ✅ Before Committing

Run this checklist:

```bash
# Test deprecation warnings emit correctly
pytest tests/unit/test_package_init.py::TestDeprecatedPublicApiSymbols -v

# Test sanctioned symbols do NOT warn
pytest tests/unit/test_package_init.py::TestSanctionedSymbolsNoWarnings -v

# Full test suite with coverage
pytest tests/unit/ --cov=src/calibrated_explanations --cov-fail-under=88

# Verify no regressions in integration tests
pytest tests/integration/ -v

# Check style/lint
pylint src/calibrated_explanations/utils/deprecation.py
mypy src/calibrated_explanations/__init__.py
```

All green? → Commit and create PR.

---

## 🎓 Learning Resources

**For context:**
- `improvement_docs/ADR-gap-analysis.md` – Gap #5 (severity 6)
- `improvement_docs/RELEASE_PLAN_v1.md` – Overall roadmap
- `improvement_docs/adrs/ADR-001.md` – Full architecture decision

**For implementation:**
- `ADR-001-STAGE-3-PUBLIC-API-NARROWING.md` – Complete analysis
- `ADR-001-STAGE-3-IMPLEMENTATION-ROADMAP.md` – Detailed steps

**For testing:**
- `.github/tests-guidance.md` – Test policy
- `tests/unit/test_package_init.py` – Existing test structure

---

**Last updated:** 2025-11-28  
**Status:** ✅ Ready for implementation

