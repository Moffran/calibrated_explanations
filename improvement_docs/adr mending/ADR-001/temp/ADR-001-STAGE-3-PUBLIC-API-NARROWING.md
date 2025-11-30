# ADR-001 Stage 3: Public API Surface Narrowing & Deprecation Strategy

**Date Authored:** 2025-11-28  
**Status:** ✅ RECOMMENDED FOR IMPLEMENTATION  
**Target Release Cycle:** v0.10.0 → v0.11.0 (Stage 3 deprecation gate closure)  
**Linked ADRs:** ADR-001 (Package Layout), ADR-011 (Deprecation Policy)  
**Severity:** 6 (medium) – Violation impact 3 × Code scope 2  

---

## Executive Summary

ADR-001 Gap #5 identifies that the public API surface exposed via `calibrated_explanations.__init__` is **overly broad**. Currently, `__getattr__` lazily exports 13+ symbols across four categories—core factories (sanctioned), explanation classes (not sanctioned), discretizers (not sanctioned), visualization namespace (not sanctioned), and calibrators (not sanctioned).

This Stage 3 analysis proposes a **two-release deprecation window** that:
1. **v0.10.0 (current dev)** – Emit structured deprecation warnings for unsanctioned symbols
2. **v0.11.0** – Remove unsanctioned symbols from `__getattr__` and lock to sanctioned-only API
3. Provides clear migration paths for users relying on top-level imports

---

## Current Public API Surface (via `__getattr__`)

### 1. **Sanctioned Symbols (ADR-001 Aligned)**

| Symbol | Current Status | Notes |
| --- | --- | --- |
| `CalibratedExplainer` | ✅ Public | Core factory; users import via `from calibrated_explanations import CalibratedExplainer` |
| `WrapCalibratedExplainer` | ✅ Public | Wrapper factory; users import via `from calibrated_explanations import WrapCalibratedExplainer` |
| `transform_to_numeric` | ✅ Public | High-level utility; users import via `from calibrated_explanations import transform_to_numeric` |

**Status:** These three symbols form the **sanctioned façade** per ADR-001. No changes required; these remain in `__all__` and `__getattr__`.

---

### 2. **Unsanctioned Symbols Requiring Deprecation**

#### Category A: Explanation Classes
| Symbol | Current Path | Import Pattern | Action |
| --- | --- | --- | --- |
| `AlternativeExplanation` | `explanations.explanation` | Lazy `__getattr__` | Deprecate + move to submodule |
| `FactualExplanation` | `explanations.explanation` | Lazy `__getattr__` | Deprecate + move to submodule |
| `FastExplanation` | `explanations.explanation` | Lazy `__getattr__` | Deprecate + move to submodule |
| `AlternativeExplanations` | `explanations.explanations` | Lazy `__getattr__` | Deprecate + move to submodule |
| `CalibratedExplanations` | `explanations.explanations` | Lazy `__getattr__` | Deprecate + move to submodule |

**Rationale:** These are domain dataclasses returned by explainer methods, not entry points. Users should import them from the `explanations` submodule: `from calibrated_explanations.explanations import CalibratedExplanations`.

#### Category B: Discretizers
| Symbol | Current Path | Import Pattern | Action |
| --- | --- | --- | --- |
| `BinaryEntropyDiscretizer` | `utils.discretizers` | Lazy `__getattr__` | Deprecate + move to submodule |
| `BinaryRegressorDiscretizer` | `utils.discretizers` | Lazy `__getattr__` | Deprecate + move to submodule |
| `EntropyDiscretizer` | `utils.discretizers` | Lazy `__getattr__` | Deprecate + move to submodule |
| `RegressorDiscretizer` | `utils.discretizers` | Lazy `__getattr__` | Deprecate + move to submodule |

**Rationale:** Discretizers are utilities for internal preprocessing. Users should import from `calibrated_explanations.utils.discretizers`: `from calibrated_explanations.utils.discretizers import EntropyDiscretizer`.

#### Category C: Calibrators
| Symbol | Current Path | Import Pattern | Action |
| --- | --- | --- | --- |
| `IntervalRegressor` | `calibration.interval_regressor` | Lazy `__getattr__` (path bug: `..calibration`) | Deprecate + move to submodule |
| `VennAbers` | `calibration.venn_abers` | Lazy `__getattr__` (path bug: `..calibration`) | Deprecate + move to submodule |

**Rationale:** Calibrators are low-level components. Users should import from `calibrated_explanations.calibration`: `from calibrated_explanations.calibration import IntervalRegressor, VennAbers`.

**⚠️ BUG NOTE:** Current `__getattr__` uses `from ..calibration.interval_regressor` which is incorrect relative path (goes up from `__init__.py` in `src/calibrated_explanations/`). Should be `from .calibration.interval_regressor`. This is a separate bug fix needed before Stage 3 deprecation.

#### Category D: Visualization Namespace
| Symbol | Current Path | Import Pattern | Action |
| --- | --- | --- | --- |
| `viz` | `viz` (entire module) | Lazy `__getattr__` | Deprecate + move to submodule |

**Rationale:** The visualization module is large, imports heavy dependencies (matplotlib), and is explicitly marked as "experimental" in its own `__init__.py`. Users should import explicitly: `from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter`.

---

## Recommended Deprecation Strategy

### Phase 1: v0.10.0 – Emit Structured Warnings

**Objective:** Warn users without breaking their code; collect field usage data.

#### 1.1 Implement Central Deprecation Helper

**File:** `src/calibrated_explanations/utils/deprecation.py` (new file)

```python
"""Central deprecation helper for ADR-011 migration gates."""

import warnings
from typing import Optional


def deprecate_public_api_symbol(
    symbol_name: str,
    current_import: str,
    recommended_import: str,
    removal_version: str = "v0.11.0",
    extra_context: Optional[str] = None,
):
    """Emit structured deprecation warning for top-level API symbols.
    
    Args:
        symbol_name: Name of the symbol being accessed (e.g., "CalibratedExplanations")
        current_import: Current (deprecated) import path 
                        (e.g., "from calibrated_explanations import CalibratedExplanations")
        recommended_import: Recommended new import path
                           (e.g., "from calibrated_explanations.explanations import CalibratedExplanations")
        removal_version: Version in which the symbol will be removed from __init__.py
        extra_context: Optional additional migration guidance
    """
    message = (
        f"\n{symbol_name!r} imported from top level is deprecated and will be removed in {removal_version}.\n"
        f"  Current:     {current_import}\n"
        f"  Recommended: {recommended_import}\n"
    )
    
    if extra_context:
        message += f"  Context: {extra_context}\n"
    
    warnings.warn(
        message.rstrip(),
        category=DeprecationWarning,
        stacklevel=3,  # Adjust to point to user code
    )
```

#### 1.2 Update `__getattr__` to Emit Warnings

**File:** `src/calibrated_explanations/__init__.py`

Wrap each unsanctioned symbol with a deprecation warning:

```python
def __getattr__(name: str):
    """Lazy import for public symbols (some deprecated per ADR-001 Stage 3)."""
    
    # ... [existing sanctioned imports] ...
    
    if name == "viz":
        from calibrated_explanations.utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            "viz",
            "from calibrated_explanations import viz",
            "from calibrated_explanations.viz import ...",
            extra_context="The viz namespace is now a submodule. Import specific classes/functions from it directly.",
        )
        module = importlib.import_module(f"{__name__}.viz")
        globals()[name] = module
        return module
    
    # Explanation classes
    if name in {"AlternativeExplanation", "FactualExplanation", "FastExplanation"}:
        from calibrated_explanations.utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            name,
            f"from calibrated_explanations import {name}",
            f"from calibrated_explanations.explanations.explanation import {name}",
            extra_context="Explanation domain classes should be imported from the explanations submodule.",
        )
        module = importlib.import_module(f"{__name__}.explanations.explanation")
        value = getattr(module, name)
        globals()[name] = value
        return value
    
    # ... [repeat pattern for all unsanctioned symbols] ...
```

### Phase 2: v0.11.0 – Remove Unsanctioned Exports

**Objective:** Lock API to sanctioned-only surface; require explicit submodule imports.

#### 2.1 Remove Unsanctioned Symbols from `__getattr__`

**File:** `src/calibrated_explanations/__init__.py`

Remove all `if name in {unsanctioned}` branches. Keep only:

```python
def __getattr__(name: str):
    """Lazy import for sanctioned public symbols (ADR-001 Stage 3)."""
    if name == "CalibratedExplainer":
        from .core.calibrated_explainer import CalibratedExplainer
        globals()[name] = CalibratedExplainer
        return CalibratedExplainer
    if name == "WrapCalibratedExplainer":
        from .core.wrap_explainer import WrapCalibratedExplainer
        globals()[name] = WrapCalibratedExplainer
        return WrapCalibratedExplainer
    if name == "transform_to_numeric":
        module = importlib.import_module(f"{__name__}.utils.helper")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

#### 2.2 Update `__all__`

```python
__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "transform_to_numeric",
]
```

---

## Migration Guide for Users

### Migration Path 1: Explanation Classes

**Before (v0.10.0 & deprecated in v0.11.0):**
```python
from calibrated_explanations import CalibratedExplanations, FactualExplanation
```

**After (v0.11.0 required):**
```python
from calibrated_explanations.explanations import CalibratedExplanations, FactualExplanation
```

### Migration Path 2: Discretizers

**Before (deprecated in v0.10.0, removed in v0.11.0):**
```python
from calibrated_explanations import EntropyDiscretizer, RegressorDiscretizer
```

**After (v0.11.0 required):**
```python
from calibrated_explanations.utils.discretizers import EntropyDiscretizer, RegressorDiscretizer
```

### Migration Path 3: Calibrators

**Before (deprecated in v0.10.0, removed in v0.11.0):**
```python
from calibrated_explanations import IntervalRegressor, VennAbers
```

**After (v0.11.0 required):**
```python
from calibrated_explanations.calibration import IntervalRegressor, VennAbers
```

### Migration Path 4: Visualization

**Before (deprecated in v0.10.0, removed in v0.11.0):**
```python
from calibrated_explanations import viz
viz.plots.plot_factual(...)
```

**After (v0.11.0 required):**
```python
from calibrated_explanations.viz import plots
plots.plot_factual(...)
```

---

## Implementation Checklist

### v0.10.0 (Deprecation Phase)

- [ ] **Create deprecation helper**
  - File: `src/calibrated_explanations/utils/deprecation.py`
  - Function: `deprecate_public_api_symbol()`
  
- [ ] **Update `__init__.py` with warnings**
  - Wrap all unsanctioned symbols with deprecation calls
  - Test warnings are emitted correctly
  
- [ ] **Fix calibration import bug**
  - Change `from ..calibration.interval_regressor` → `from .calibration.interval_regressor`
  
- [ ] **Add tests for deprecation warnings**
  - Test each unsanctioned symbol emits the correct warning
  - Verify sanctioned symbols do NOT emit warnings
  - See "Test Changes" section below
  
- [ ] **Update CHANGELOG**
  - Document that explanation, discretizer, calibrator, and viz imports are deprecated
  - Point users to migration guide
  
- [ ] **Documentation**
  - Add migration guide to docs
  - Update architecture docs to reflect sanctioned API

### v0.11.0 (Removal Phase)

- [ ] **Remove unsanctioned symbols from `__getattr__`**
  - Delete all deprecation branches
  - Update `__all__` to sanctioned-only list
  
- [ ] **Update tests**
  - Remove tests that verify unsanctioned imports work
  - Update integration tests to use new import paths
  
- [ ] **Update CHANGELOG**
  - Document breaking change (removal of top-level exports)
  - Reference v0.10.0 migration guide

---

## Test Changes Needed

### Phase 1: v0.10.0 – Deprecation Tests

**File:** `tests/unit/test_package_init.py` (extend existing)

```python
"""Tests for public API deprecation strategy (ADR-001 Stage 3)."""

import pytest
import warnings
import calibrated_explanations as ce


class TestDeprecatedExplanationImports:
    """Verify explanation classes emit deprecation warnings from top level."""
    
    def test_should_emit_deprecation_when_accessing_alternative_explanation(self, monkeypatch):
        # Arrange
        monkeypatch.delitem(ce.__dict__, "AlternativeExplanation", raising=False)
        
        # Act & Assert
        with pytest.warns(DeprecationWarning, match="AlternativeExplanation.*deprecated"):
            alternative_explanation = ce.AlternativeExplanation
        
        # Verify actual import still works
        from calibrated_explanations.explanations.explanation import AlternativeExplanation
        assert alternative_explanation is AlternativeExplanation
    
    def test_should_emit_deprecation_when_accessing_factual_explanation(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "FactualExplanation", raising=False)
        
        with pytest.warns(DeprecationWarning, match="FactualExplanation.*deprecated"):
            factual_explanation = ce.FactualExplanation
        
        from calibrated_explanations.explanations.explanation import FactualExplanation
        assert factual_explanation is FactualExplanation
    
    def test_should_emit_deprecation_when_accessing_calibrated_explanations(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "CalibratedExplanations", raising=False)
        
        with pytest.warns(DeprecationWarning, match="CalibratedExplanations.*deprecated"):
            explanations = ce.CalibratedExplanations
        
        from calibrated_explanations.explanations.explanations import CalibratedExplanations
        assert explanations is CalibratedExplanations


class TestDeprecatedDiscretizerImports:
    """Verify discretizers emit deprecation warnings from top level."""
    
    def test_should_emit_deprecation_when_accessing_entropy_discretizer(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "EntropyDiscretizer", raising=False)
        
        with pytest.warns(DeprecationWarning, match="EntropyDiscretizer.*deprecated"):
            discretizer = ce.EntropyDiscretizer
        
        from calibrated_explanations.utils.discretizers import EntropyDiscretizer
        assert discretizer is EntropyDiscretizer
    
    def test_should_emit_deprecation_when_accessing_regressor_discretizer(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "RegressorDiscretizer", raising=False)
        
        with pytest.warns(DeprecationWarning, match="RegressorDiscretizer.*deprecated"):
            discretizer = ce.RegressorDiscretizer
        
        from calibrated_explanations.utils.discretizers import RegressorDiscretizer
        assert discretizer is RegressorDiscretizer


class TestDeprecatedCalibratorImports:
    """Verify calibrators emit deprecation warnings from top level."""
    
    def test_should_emit_deprecation_when_accessing_interval_regressor(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "IntervalRegressor", raising=False)
        
        with pytest.warns(DeprecationWarning, match="IntervalRegressor.*deprecated"):
            regressor = ce.IntervalRegressor
        
        from calibrated_explanations.calibration import IntervalRegressor
        assert regressor is IntervalRegressor
    
    def test_should_emit_deprecation_when_accessing_venn_abers(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "VennAbers", raising=False)
        
        with pytest.warns(DeprecationWarning, match="VennAbers.*deprecated"):
            venn = ce.VennAbers
        
        from calibrated_explanations.calibration import VennAbers
        assert venn is VennAbers


class TestDeprecatedVizImports:
    """Verify viz namespace emits deprecation warning from top level."""
    
    def test_should_emit_deprecation_when_accessing_viz_namespace(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "viz", raising=False)
        
        with pytest.warns(DeprecationWarning, match="viz.*deprecated"):
            viz = ce.viz
        
        from calibrated_explanations import viz as viz_direct
        assert viz is viz_direct


class TestSanctionedImportsNoWarnings:
    """Verify sanctioned symbols do NOT emit deprecation warnings."""
    
    def test_should_not_emit_deprecation_for_calibrated_explainer(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "CalibratedExplainer", raising=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should not raise
            explainer_cls = ce.CalibratedExplainer
        
        from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
        assert explainer_cls is CalibratedExplainer
    
    def test_should_not_emit_deprecation_for_wrap_calibrated_explainer(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "WrapCalibratedExplainer", raising=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should not raise
            wrapper_cls = ce.WrapCalibratedExplainer
        
        from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
        assert wrapper_cls is WrapCalibratedExplainer
    
    def test_should_not_emit_deprecation_for_transform_to_numeric(self, monkeypatch):
        monkeypatch.delitem(ce.__dict__, "transform_to_numeric", raising=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should not raise
            transform_fn = ce.transform_to_numeric
        
        from calibrated_explanations.utils.helper import transform_to_numeric
        assert transform_fn is transform_to_numeric
```

### Phase 2: v0.11.0 – Removal Tests

**File:** `tests/unit/test_package_init.py` (replace deprecation tests)

```python
"""Tests for sanctioned-only public API (ADR-001 Stage 3, v0.11.0)."""

import pytest
import calibrated_explanations as ce


class TestSanctionedSymbolsOnly:
    """Verify only sanctioned symbols are accessible from top level."""
    
    def test_should_access_calibrated_explainer(self):
        from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
        assert ce.CalibratedExplainer is CalibratedExplainer
    
    def test_should_access_wrap_calibrated_explainer(self):
        from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
        assert ce.WrapCalibratedExplainer is WrapCalibratedExplainer
    
    def test_should_access_transform_to_numeric(self):
        from calibrated_explanations.utils.helper import transform_to_numeric
        assert ce.transform_to_numeric is transform_to_numeric


class TestUnsanctionedSymbolsRaiseAttributeError:
    """Verify unsanctioned symbols raise AttributeError."""
    
    def test_should_raise_on_alternative_explanation(self):
        with pytest.raises(AttributeError, match="has no attribute 'AlternativeExplanation'"):
            ce.AlternativeExplanation
    
    def test_should_raise_on_entropy_discretizer(self):
        with pytest.raises(AttributeError, match="has no attribute 'EntropyDiscretizer'"):
            ce.EntropyDiscretizer
    
    def test_should_raise_on_interval_regressor(self):
        with pytest.raises(AttributeError, match="has no attribute 'IntervalRegressor'"):
            ce.IntervalRegressor
    
    def test_should_raise_on_venn_abers(self):
        with pytest.raises(AttributeError, match="has no attribute 'VennAbers'"):
            ce.VennAbers
    
    def test_should_raise_on_viz_namespace(self):
        with pytest.raises(AttributeError, match="has no attribute 'viz'"):
            ce.viz
```

### Integration Tests Update

**File:** `tests/integration/test_*.py` (various)

Scan for any integration tests that use deprecated top-level imports and update them:

```python
# OLD (deprecated)
from calibrated_explanations import (
    CalibratedExplainer, 
    CalibratedExplanations,  # ← Deprecated
    EntropyDiscretizer,      # ← Deprecated
)

# NEW (v0.11.0)
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.explanations import CalibratedExplanations
from calibrated_explanations.utils.discretizers import EntropyDiscretizer
```

---

## Breaking Changes & User Impact

### Categories of Affected Users

1. **Users importing explanation classes from top level**
   - Impact: Moderate (easy migration, documented)
   - Count estimate: Small (specialized users examining internal types)

2. **Users importing discretizers from top level**
   - Impact: Low (discretizers are internal utilities)
   - Count estimate: Very small

3. **Users importing calibrators from top level**
   - Impact: Moderate (calibrators used in advanced workflows)
   - Count estimate: Small to moderate

4. **Users accessing viz namespace from top level**
   - Impact: Moderate (viz is experimental and documented as such)
   - Count estimate: Small to moderate (most use submodule imports already)

### Mitigation Strategy

1. **Clear messaging in v0.10.0 release notes** pointing to migration guide
2. **Deprecation warnings** with actionable import paths
3. **Extended deprecation window** (full minor version cycle = 2-4 months)
4. **Migration guide in docs** with copy-paste examples
5. **CI/CD warning verification** to catch internal code needing updates early

---

## ADR-001 Alignment

| Gap | Status | Impact |
| --- | --- | --- |
| Public API surface overly broad | ✅ Addressed | v0.10.0 deprecates; v0.11.0 removes unsanctioned exports |
| Extra top-level namespaces lack ADR coverage | ✅ Addressed | `viz` moved to submodule-only; docs clarified |
| Overall ADR-001 completion | 📈 Stage 3 Ready | Stages 0–2 complete; Stage 3 closes API surface gap |

**ADR-011 Integration:** This stage implements the first concrete deprecation gate per ADR-011 (two-release window, structured warnings, migration guidance).

---

## Rollout Risks & Mitigations

| Risk | Severity | Mitigation |
| --- | --- | --- |
| External users relying on deprecated imports fail to update | High | Early communication, long deprecation window, clear docs |
| Internal code not updated before v0.11.0 release | High | Automated grep-based checks in CI; lint rule added |
| Documentation examples still show old imports | Medium | Audit all docs and notebooks before v0.11.0 RC |
| Tests fail in v0.11.0 if migration incomplete | Medium | Automated test import migration script |

---

## Documentation & Communication

### 1. CHANGELOG Entry (v0.10.0)

```markdown
### Deprecations

- **Top-level explanation class imports** (e.g., `from calibrated_explanations import CalibratedExplanations`) 
  are deprecated and will be removed in v0.11.0. 
  Import from `calibrated_explanations.explanations` instead.
  
- **Top-level discretizer imports** (e.g., `from calibrated_explanations import EntropyDiscretizer`) 
  are deprecated and will be removed in v0.11.0. 
  Import from `calibrated_explanations.utils.discretizers` instead.
  
- **Top-level calibrator imports** (e.g., `from calibrated_explanations import IntervalRegressor`) 
  are deprecated and will be removed in v0.11.0. 
  Import from `calibrated_explanations.calibration` instead.
  
- **Top-level `viz` namespace import** (e.g., `from calibrated_explanations import viz`) 
  is deprecated and will be removed in v0.11.0. 
  Import specific items from `calibrated_explanations.viz` instead.

See [migration guide](./docs/migration/api_surface_narrowing.md) for detailed examples.
```

### 2. Migration Guide (new file)

**File:** `docs/migration/api_surface_narrowing.md`

(See migration examples in "Migration Guide for Users" section above)

### 3. Architecture Documentation Update

**File:** `docs/architecture/public_api.md` (new file)

Document the sanctioned façade, explain rationale, link to submodule docs.

---

## Success Criteria

✅ **v0.10.0:**
- Deprecation helper implemented
- All unsanctioned imports emit structured warnings
- Tests verify warnings are emitted correctly
- CHANGELOG documents changes
- Migration guide published in docs

✅ **v0.11.0:**
- Unsanctioned symbols removed from `__getattr__`
- `__all__` updated to sanctioned-only list
- Internal code updated to use submodule imports
- All tests passing with new API
- Breaking change clearly documented in release notes

---

## References

- **ADR-001:** Package and Boundary Layout – `improvement_docs/adrs/ADR-001.md`
- **ADR-001 Stage 2:** Decouple Cross-Sibling Imports – `improvement_docs/ADR-001-STAGE-2-COMPLETION-REPORT.md`
- **ADR Gap #5:** Public API Surface Overly Broad (severity 6) – `improvement_docs/ADR-gap-analysis.md:L50-L52`
- **ADR-011:** Deprecation Policy – `improvement_docs/RELEASE_PLAN_v1.md:L104-L120`
- **Current `__init__.py`:** `src/calibrated_explanations/__init__.py`
- **Test guidance:** `.github/tests-guidance.md`

