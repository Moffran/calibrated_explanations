# ADR-001 Stage 3 Implementation Roadmap

**Status:** Ready for v0.10.0 Implementation  
**Estimated Effort:** 4–6 hours (3-4 iterations)  
**Prerequisites:** ADR-001 Stages 1–2 complete ✅

---

## Quick Reference: What Gets Changed

### Current State (v0.10.0 dev)
```python
# calibrated_explanations/__init__.py
__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer", 
    "transform_to_numeric",
]

def __getattr__(name):
    # 13+ symbols via lazy imports, including:
    # - Sanctioned: CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric ✓
    # - Unsanctioned: viz, explanations (5x), discretizers (4x), calibrators (2x) ✗
```

### Target State (v0.11.0)
```python
# calibrated_explanations/__init__.py
__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer", 
    "transform_to_numeric",
]

def __getattr__(name):
    # Only sanctioned symbols; everything else raises AttributeError
```

---

## Step-by-Step Implementation (v0.10.0)

### STEP 1: Create Deprecation Helper (30 min)

**File:** `src/calibrated_explanations/utils/deprecation.py` (NEW)

```python
"""Central deprecation helper for ADR-011 migration gates.

This module provides structured deprecation warnings that are:
- Consistent across all deprecation sites
- Testable and mockable
- Tagged with removal version and alternative imports
- Suitable for CI/CD enforcement
"""

import warnings
from typing import Optional


def deprecate_public_api_symbol(
    symbol_name: str,
    current_import: str,
    recommended_import: str,
    removal_version: str = "v0.11.0",
    extra_context: Optional[str] = None,
) -> None:
    """Emit structured deprecation warning for top-level API symbols.
    
    This function centralizes deprecation messaging for unsanctioned exports
    from calibrated_explanations.__init__, following ADR-001 Stage 3 and 
    ADR-011 deprecation policy.
    
    Args:
        symbol_name: Name of the symbol being accessed (e.g., "CalibratedExplanations")
        current_import: Current (deprecated) import path 
                        (e.g., "from calibrated_explanations import CalibratedExplanations")
        recommended_import: Recommended new import path
                           (e.g., "from calibrated_explanations.explanations import CalibratedExplanations")
        removal_version: Version in which the symbol will be removed from __init__.py (default: v0.11.0)
        extra_context: Optional additional migration guidance or explanation
        
    Examples:
        >>> deprecate_public_api_symbol(
        ...     "CalibratedExplanations",
        ...     "from calibrated_explanations import CalibratedExplanations",
        ...     "from calibrated_explanations.explanations import CalibratedExplanations",
        ...     extra_context="Explanation dataclasses are domain objects; import from the submodule.",
        ... )
    """
    message = (
        f"\n{symbol_name!r} imported from top level is deprecated and will be removed in {removal_version}.\n"
        f"  ❌ DEPRECATED: {current_import}\n"
        f"  ✓  RECOMMENDED: {recommended_import}\n"
    )
    
    if extra_context:
        message += f"\n  Details: {extra_context}\n"
    
    message += f"\nSee https://calibrated-explanations.readthedocs.io/en/latest/migration/api_surface_narrowing.html for migration guide.\n"
    
    warnings.warn(
        message.rstrip(),
        category=DeprecationWarning,
        stacklevel=3,
    )
```

---

### STEP 2: Fix Calibration Import Bug (10 min)

**File:** `src/calibrated_explanations/__init__.py`

**Current (BUGGY):**
```python
    if name == "IntervalRegressor":
        from ..calibration.interval_regressor import IntervalRegressor
```

**Fixed:**
```python
    if name == "IntervalRegressor":
        from .calibration.interval_regressor import IntervalRegressor
```

**Rationale:** The relative import `..calibration` goes up one level from the `src/calibrated_explanations/` directory, which is incorrect. Should be `.calibration` to reference the sibling package.

---

### STEP 3: Update `__getattr__` with Deprecation Warnings (1.5 hours)

**File:** `src/calibrated_explanations/__init__.py`

Replace each unsanctioned symbol block with deprecation wrapper:

#### 3a. Add import at top of file:

```python
import importlib
import logging as _logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports; used only for type hints
    pass
```

#### 3b. Replace viz block:

```python
    if name == "viz":
        from .utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            "viz",
            "from calibrated_explanations import viz",
            "from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter",
            extra_context="The viz namespace is now a submodule. Import specific classes/functions from it directly.",
        )
        module = importlib.import_module(f"{__name__}.viz")
        globals()[name] = module
        return module
```

#### 3c. Replace explanation classes block:

```python
    if name in {
        "AlternativeExplanation",
        "FactualExplanation",
        "FastExplanation",
    }:
        from .utils.deprecation import deprecate_public_api_symbol
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
```

#### 3d. Replace alternative/calibrated explanations block:

```python
    if name in {
        "AlternativeExplanations",
        "CalibratedExplanations",
    }:
        from .utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            name,
            f"from calibrated_explanations import {name}",
            f"from calibrated_explanations.explanations import {name}",
            extra_context="Explanation collections should be imported from the explanations submodule.",
        )
        module = importlib.import_module(f"{__name__}.explanations.explanations")
        value = getattr(module, name)
        globals()[name] = value
        return value
```

#### 3e. Replace discretizers block:

```python
    if name in {
        "BinaryEntropyDiscretizer",
        "BinaryRegressorDiscretizer",
        "EntropyDiscretizer",
        "RegressorDiscretizer",
    }:
        from .utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            name,
            f"from calibrated_explanations import {name}",
            f"from calibrated_explanations.utils.discretizers import {name}",
            extra_context="Discretizers are internal utilities; import from the discretizers submodule.",
        )
        module = importlib.import_module(f"{__name__}.utils.discretizers")
        value = getattr(module, name)
        globals()[name] = value
        return value
```

#### 3f. Replace calibrators block (also fix the relative import):

```python
    if name == "IntervalRegressor":
        from .utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            "IntervalRegressor",
            "from calibrated_explanations import IntervalRegressor",
            "from calibrated_explanations.calibration import IntervalRegressor",
            extra_context="Calibrators are domain components; import from the calibration submodule.",
        )
        from .calibration.interval_regressor import IntervalRegressor

        globals()[name] = IntervalRegressor
        return IntervalRegressor
    
    if name == "VennAbers":
        from .utils.deprecation import deprecate_public_api_symbol
        deprecate_public_api_symbol(
            "VennAbers",
            "from calibrated_explanations import VennAbers",
            "from calibrated_explanations.calibration import VennAbers",
            extra_context="Calibrators are domain components; import from the calibration submodule.",
        )
        from .calibration.venn_abers import VennAbers

        globals()[name] = VennAbers
        return VennAbers
```

**Keep sanctioned symbols unchanged:**

```python
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
```

---

### STEP 4: Add Unit Tests for Deprecation Warnings (1.5 hours)

**File:** `tests/unit/test_package_init.py` (extend existing)

Add test class to verify deprecation warnings:

```python
"""Tests for public API deprecation (ADR-001 Stage 3)."""

import pytest
import warnings
import calibrated_explanations as ce


class TestDeprecatedPublicApiSymbols:
    """Verify that unsanctioned symbols emit DeprecationWarning when accessed from top level."""
    
    # Explanation classes
    
    def test_should_emit_deprecation_for_alternative_explanation(self, monkeypatch):
        """Should warn when accessing AlternativeExplanation from top level."""
        monkeypatch.delitem(ce.__dict__, "AlternativeExplanation", raising=False)
        
        with pytest.warns(DeprecationWarning, match="AlternativeExplanation.*deprecated.*v0.11.0"):
            _ = ce.AlternativeExplanation
    
    def test_should_emit_deprecation_for_factual_explanation(self, monkeypatch):
        """Should warn when accessing FactualExplanation from top level."""
        monkeypatch.delitem(ce.__dict__, "FactualExplanation", raising=False)
        
        with pytest.warns(DeprecationWarning, match="FactualExplanation.*deprecated.*v0.11.0"):
            _ = ce.FactualExplanation
    
    def test_should_emit_deprecation_for_fast_explanation(self, monkeypatch):
        """Should warn when accessing FastExplanation from top level."""
        monkeypatch.delitem(ce.__dict__, "FastExplanation", raising=False)
        
        with pytest.warns(DeprecationWarning, match="FastExplanation.*deprecated.*v0.11.0"):
            _ = ce.FastExplanation
    
    def test_should_emit_deprecation_for_alternative_explanations(self, monkeypatch):
        """Should warn when accessing AlternativeExplanations from top level."""
        monkeypatch.delitem(ce.__dict__, "AlternativeExplanations", raising=False)
        
        with pytest.warns(DeprecationWarning, match="AlternativeExplanations.*deprecated.*v0.11.0"):
            _ = ce.AlternativeExplanations
    
    def test_should_emit_deprecation_for_calibrated_explanations(self, monkeypatch):
        """Should warn when accessing CalibratedExplanations from top level."""
        monkeypatch.delitem(ce.__dict__, "CalibratedExplanations", raising=False)
        
        with pytest.warns(DeprecationWarning, match="CalibratedExplanations.*deprecated.*v0.11.0"):
            _ = ce.CalibratedExplanations
    
    # Discretizers
    
    def test_should_emit_deprecation_for_entropy_discretizer(self, monkeypatch):
        """Should warn when accessing EntropyDiscretizer from top level."""
        monkeypatch.delitem(ce.__dict__, "EntropyDiscretizer", raising=False)
        
        with pytest.warns(DeprecationWarning, match="EntropyDiscretizer.*deprecated.*v0.11.0"):
            _ = ce.EntropyDiscretizer
    
    def test_should_emit_deprecation_for_regressor_discretizer(self, monkeypatch):
        """Should warn when accessing RegressorDiscretizer from top level."""
        monkeypatch.delitem(ce.__dict__, "RegressorDiscretizer", raising=False)
        
        with pytest.warns(DeprecationWarning, match="RegressorDiscretizer.*deprecated.*v0.11.0"):
            _ = ce.RegressorDiscretizer
    
    def test_should_emit_deprecation_for_binary_entropy_discretizer(self, monkeypatch):
        """Should warn when accessing BinaryEntropyDiscretizer from top level."""
        monkeypatch.delitem(ce.__dict__, "BinaryEntropyDiscretizer", raising=False)
        
        with pytest.warns(DeprecationWarning, match="BinaryEntropyDiscretizer.*deprecated.*v0.11.0"):
            _ = ce.BinaryEntropyDiscretizer
    
    def test_should_emit_deprecation_for_binary_regressor_discretizer(self, monkeypatch):
        """Should warn when accessing BinaryRegressorDiscretizer from top level."""
        monkeypatch.delitem(ce.__dict__, "BinaryRegressorDiscretizer", raising=False)
        
        with pytest.warns(DeprecationWarning, match="BinaryRegressorDiscretizer.*deprecated.*v0.11.0"):
            _ = ce.BinaryRegressorDiscretizer
    
    # Calibrators
    
    def test_should_emit_deprecation_for_interval_regressor(self, monkeypatch):
        """Should warn when accessing IntervalRegressor from top level."""
        monkeypatch.delitem(ce.__dict__, "IntervalRegressor", raising=False)
        
        with pytest.warns(DeprecationWarning, match="IntervalRegressor.*deprecated.*v0.11.0"):
            _ = ce.IntervalRegressor
    
    def test_should_emit_deprecation_for_venn_abers(self, monkeypatch):
        """Should warn when accessing VennAbers from top level."""
        monkeypatch.delitem(ce.__dict__, "VennAbers", raising=False)
        
        with pytest.warns(DeprecationWarning, match="VennAbers.*deprecated.*v0.11.0"):
            _ = ce.VennAbers
    
    # Viz namespace
    
    def test_should_emit_deprecation_for_viz_namespace(self, monkeypatch):
        """Should warn when accessing viz namespace from top level."""
        monkeypatch.delitem(ce.__dict__, "viz", raising=False)
        
        with pytest.warns(DeprecationWarning, match="viz.*deprecated.*v0.11.0"):
            _ = ce.viz


class TestSanctionedSymbolsNoWarnings:
    """Verify that sanctioned symbols do NOT emit deprecation warnings."""
    
    def test_should_not_warn_for_calibrated_explainer(self, monkeypatch):
        """Sanctioned: CalibratedExplainer should not emit warnings."""
        monkeypatch.delitem(ce.__dict__, "CalibratedExplainer", raising=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should not raise
            _ = ce.CalibratedExplainer
    
    def test_should_not_warn_for_wrap_calibrated_explainer(self, monkeypatch):
        """Sanctioned: WrapCalibratedExplainer should not emit warnings."""
        monkeypatch.delitem(ce.__dict__, "WrapCalibratedExplainer", raising=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should not raise
            _ = ce.WrapCalibratedExplainer
    
    def test_should_not_warn_for_transform_to_numeric(self, monkeypatch):
        """Sanctioned: transform_to_numeric should not emit warnings."""
        monkeypatch.delitem(ce.__dict__, "transform_to_numeric", raising=False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should not raise
            _ = ce.transform_to_numeric
```

**Run tests to verify:**
```bash
pytest tests/unit/test_package_init.py::TestDeprecatedPublicApiSymbols -v
pytest tests/unit/test_package_init.py::TestSanctionedSymbolsNoWarnings -v
```

---

### STEP 5: Update CHANGELOG (20 min)

**File:** `CHANGELOG.md`

Add to `## Unreleased / [v0.10.0-dev]` section:

```markdown
### ⚠️ Deprecations

- **Top-level explanation class imports** are deprecated in v0.10.0 and will be removed in v0.11.0
  - ❌ Deprecated: `from calibrated_explanations import CalibratedExplanations, FactualExplanation, AlternativeExplanation, FastExplanation`
  - ✓ Use instead: `from calibrated_explanations.explanations import CalibratedExplanations, FactualExplanation`
  - ✓ Use instead: `from calibrated_explanations.explanations.explanation import AlternativeExplanation, FastExplanation`

- **Top-level discretizer imports** are deprecated in v0.10.0 and will be removed in v0.11.0
  - ❌ Deprecated: `from calibrated_explanations import EntropyDiscretizer, RegressorDiscretizer, BinaryEntropyDiscretizer, BinaryRegressorDiscretizer`
  - ✓ Use instead: `from calibrated_explanations.utils.discretizers import EntropyDiscretizer, RegressorDiscretizer, BinaryEntropyDiscretizer, BinaryRegressorDiscretizer`

- **Top-level calibrator imports** are deprecated in v0.10.0 and will be removed in v0.11.0
  - ❌ Deprecated: `from calibrated_explanations import IntervalRegressor, VennAbers`
  - ✓ Use instead: `from calibrated_explanations.calibration import IntervalRegressor, VennAbers`

- **Top-level `viz` namespace import** is deprecated in v0.10.0 and will be removed in v0.11.0
  - ❌ Deprecated: `from calibrated_explanations import viz`
  - ✓ Use instead: `from calibrated_explanations.viz import PlotSpec, plots, matplotlib_adapter`

**Rationale:** ADR-001 Stage 3 narrows the public API surface to sanctioned entry points only (CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric). Domain classes, utilities, and visualization components should be imported from their respective submodules. See [migration guide](./docs/migration/api_surface_narrowing.md) for details.
```

---

### STEP 6: Create Migration Guide (45 min)

**File:** `docs/migration/api_surface_narrowing.md` (NEW)

```markdown
# API Surface Narrowing: v0.10.0 → v0.11.0 Migration Guide

As of v0.10.0, the public API exported from `calibrated_explanations` is being narrowed to align with ADR-001 
package layout guidelines. This ensures a clear separation between entry points (the sanctioned façade) and 
internal domain classes.

## What's Changing?

### Public API (Unchanged – No Action Required)

The following symbols remain available at the top level:

```python
from calibrated_explanations import (
    CalibratedExplainer,
    WrapCalibratedExplainer,
    transform_to_numeric,
)
```

These three symbols form the **sanctioned entry point façade** and are stable API.

### Deprecated Symbols (Action Required by v0.11.0)

The following symbols are deprecated in v0.10.0 and will be **removed in v0.11.0**:

1. Explanation classes
2. Discretizers
3. Calibrators
4. Visualization namespace

---

## Migration Examples

### 1. Explanation Classes

#### Before (v0.10.0, deprecated)
```python
from calibrated_explanations import (
    CalibratedExplanations,
    AlternativeExplanations,
    FactualExplanation,
    AlternativeExplanation,
    FastExplanation,
)
```

#### After (v0.11.0 required)
```python
# For explanation collections
from calibrated_explanations.explanations import (
    CalibratedExplanations,
    AlternativeExplanations,
)

# For individual explanation domain classes
from calibrated_explanations.explanations.explanation import (
    FactualExplanation,
    AlternativeExplanation,
    FastExplanation,
)
```

---

### 2. Discretizers

#### Before (v0.10.0, deprecated)
```python
from calibrated_explanations import (
    EntropyDiscretizer,
    RegressorDiscretizer,
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
)

discretizer = EntropyDiscretizer(bins=10)
```

#### After (v0.11.0 required)
```python
from calibrated_explanations.utils.discretizers import (
    EntropyDiscretizer,
    RegressorDiscretizer,
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
)

discretizer = EntropyDiscretizer(bins=10)
```

---

### 3. Calibrators

#### Before (v0.10.0, deprecated)
```python
from calibrated_explanations import IntervalRegressor, VennAbers

# Use in calibration workflow
regressor = IntervalRegressor(...)
venn_abers = VennAbers(...)
```

#### After (v0.11.0 required)
```python
from calibrated_explanations.calibration import IntervalRegressor, VennAbers

# Use in calibration workflow (unchanged)
regressor = IntervalRegressor(...)
venn_abers = VennAbers(...)
```

---

### 4. Visualization Namespace

#### Before (v0.10.0, deprecated)
```python
from calibrated_explanations import viz

# Access PlotSpec abstraction
plotspec = viz.PlotSpec(...)

# Create plots
viz.plots.plot_factual(...)
viz.plots.plot_alternatives(...)
```

#### After (v0.11.0 required)
```python
from calibrated_explanations.viz import PlotSpec, plots

# Access PlotSpec abstraction
plotspec = PlotSpec(...)

# Create plots (unchanged)
plots.plot_factual(...)
plots.plot_alternatives(...)
```

---

## Deprecation Warning Format

Starting in v0.10.0, when you use a deprecated symbol, you'll see a warning like:

```
DeprecationWarning: 'CalibratedExplanations' imported from top level is deprecated and will be removed in v0.11.0.
  ❌ DEPRECATED: from calibrated_explanations import CalibratedExplanations
  ✓  RECOMMENDED: from calibrated_explanations.explanations import CalibratedExplanations

  Details: Explanation dataclasses are domain objects; import from the submodule.

See https://calibrated-explanations.readthedocs.io/en/latest/migration/api_surface_narrowing.html for migration guide.
```

To suppress warnings while migrating, use Python's standard `warnings` module:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Your code that uses deprecated imports
from calibrated_explanations import CalibratedExplanations
```

---

## Why This Change?

ADR-001 ("Package and Boundary Layout") establishes clear boundaries between:

- **Entry points** (sanctioned façade): `CalibratedExplainer`, `WrapCalibratedExplainer`, `transform_to_numeric`
- **Domain classes** (submodule imports): Explanation dataclasses, discretizers, calibrators
- **Experimental modules** (submodule imports): Visualization and plotting components

This narrowing:
✅ Reduces cognitive load when reading example code  
✅ Makes the public API contract explicit  
✅ Enables internal refactoring without breaking changes  
✅ Follows Python best practices (e.g., scikit-learn, pandas)

---

## Questions?

For more context, see:
- [ADR-001: Package and Boundary Layout](../architecture/adrs/ADR-001.md)
- [ADR-011: Deprecation Policy](../architecture/adrs/ADR-011.md)
- [Public API Documentation](../architecture/public_api.md)

Or open an issue on [GitHub](https://github.com/Moffran/calibrated_explanations/issues).
```

---

### STEP 7: Run Full Test Suite (30 min)

Verify all changes work correctly:

```bash
# Run package init tests
pytest tests/unit/test_package_init.py -v

# Run full test suite to catch any regressions
pytest tests/unit/ -v --tb=short

# Check coverage
pytest tests/unit/ --cov=src/calibrated_explanations --cov-fail-under=88 --cov-config=.coveragerc
```

---

### STEP 8: Documentation Update (30 min)

#### 8a. Update Architecture Docs

**File:** `docs/architecture/public_api.md` (NEW)

```markdown
# Public API Contract

## Sanctioned Entry Points (v0.10.0+)

The following symbols form the stable, public-facing API and are guaranteed to remain at the top level:

### Factories

- **`CalibratedExplainer`**: Core factory for creating calibrated explainers from any scikit-learn-compatible estimator
  ```python
  from calibrated_explanations import CalibratedExplainer
  explainer = CalibratedExplainer(estimator)
  ```

- **`WrapCalibratedExplainer`**: Convenience wrapper for immediate calibration workflows
  ```python
  from calibrated_explanations import WrapCalibratedExplainer
  explainer = WrapCalibratedExplainer(estimator)
  ```

### Utilities

- **`transform_to_numeric`**: High-level utility for preprocessing categorical features
  ```python
  from calibrated_explanations import transform_to_numeric
  x_numeric = transform_to_numeric(x)
  ```

## Submodule Imports (Use These Instead)

### Explanation Classes
```python
from calibrated_explanations.explanations import (
    CalibratedExplanations,        # Collection of factual explanations
    AlternativeExplanations,       # Collection of alternative explanations
)
from calibrated_explanations.explanations.explanation import (
    FactualExplanation,            # Single factual explanation
    AlternativeExplanation,        # Single alternative explanation
    FastExplanation,               # Fast explanation variant
)
```

### Calibrators
```python
from calibrated_explanations.calibration import (
    IntervalRegressor,             # Conformal prediction regressor
    VennAbers,                     # Venn-Abers calibration
)
```

### Discretizers
```python
from calibrated_explanations.utils.discretizers import (
    EntropyDiscretizer,            # Entropy-based binning
    RegressorDiscretizer,          # Regressor-based binning
    BinaryEntropyDiscretizer,      # Binary entropy variant
    BinaryRegressorDiscretizer,    # Binary regressor variant
)
```

### Visualization (Experimental)
```python
from calibrated_explanations.viz import (
    PlotSpec,                      # Plot specification abstraction
    plots,                         # Plot building functions
    matplotlib_adapter,            # Matplotlib rendering backend
)
```

## Deprecation Timeline

| Release | Status | Action |
| --- | --- | --- |
| v0.10.0 | 🔶 Deprecation Phase | Unsanctioned symbols emit warnings |
| v0.11.0 | ⛔ Removal Phase | Unsanctioned symbols removed; submodule imports required |
| v1.0.0+ | ✅ Stable | Sanctioned API locked; submodule API may evolve |

See [migration guide](../migration/api_surface_narrowing.md) for detailed migration instructions.
```

#### 8b. Update README Examples (if needed)

Scan `README.md` for any deprecated imports and update them.

---

## Validation Checklist

Before committing, verify:

- [ ] `src/calibrated_explanations/utils/deprecation.py` created and functional
- [ ] `src/calibrated_explanations/__init__.py` updated with all deprecation warnings
- [ ] Calibration import bug fixed (`..calibration` → `.calibration`)
- [ ] All tests in `tests/unit/test_package_init.py` pass
- [ ] No deprecation warnings emitted for sanctioned symbols
- [ ] Full test suite passes (coverage ≥88%)
- [ ] CHANGELOG updated with deprecation notices
- [ ] Migration guide created in docs
- [ ] Architecture docs updated

---

## Commit Message Template

```
feat(api): ADR-001 Stage 3 – Narrow public API surface with v0.10.0 deprecation gate

### Changes
- Implement central deprecation helper (utils.deprecation module)
- Emit DeprecationWarning for unsanctioned top-level exports:
  * Explanation classes (5x)
  * Discretizers (4x)
  * Calibrators (2x)
  * Visualization namespace
- Fix calibration import path bug (..calibration → .calibration)
- Add comprehensive deprecation tests

### Impact
- Users see clear, actionable warnings when using unsanctioned imports
- v0.11.0 will remove unsanctioned symbols; see migration guide
- All sanctioned symbols remain unchanged (no warnings)

### ADR Alignment
- ADR-001 Gap #5: "Public API surface overly broad" (severity 6, ADDRESSED)
- ADR-011: Deprecation Policy (two-release removal window implemented)

### Testing
- 14 new unit tests for deprecation warnings
- All existing tests passing
- Coverage maintained at 88%+

Closes #[ADR-001-STAGE-3]
Relates to #[ADR-011]
```

---

## Next Steps (v0.11.0)

1. **Remove all unsanctioned branches** from `__getattr__`
2. **Update `__all__` to sanctioned-only list**
3. **Scan codebase** for internal uses of deprecated imports and update them
4. **Update all docs/notebooks** to use new import paths
5. **Release as breaking change** with migration guide reference
6. **Remove deprecation.py** (no longer needed after v0.11.0)

