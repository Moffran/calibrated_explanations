# ADR-002 Migration Guide for Downstream Code

**Version**: v0.10.0
**Date**: 2025-11-29
**Scope**: Exception taxonomy changes, parameter guardrails, error diagnostics

---

## Quick Migration Path

### No Breaking Changes
**All implementations are backward-compatible**. Successful code paths are unaffected. Only exception types change.

### For Most Users: Zero Action Required ✅
If your code does **not** explicitly catch `ValueError` or `RuntimeError` from calibrated_explanations, no changes needed.

### For Code Catching Legacy Exceptions: Update Catch Blocks
If your code explicitly catches `ValueError` or `RuntimeError` from calibrated_explanations, update to catch ADR-002 exception types.

---

## Exception Type Migration Map

### Calibration Module Changes

#### VennAbers (`src/calibrated_explanations/calibration/venn_abers.py`)
| Location | Old Exception | New Exception | Context |
|----------|---------------|---------------|---------|
| L210 | `ValueError` | `ConfigurationError` | Mondrian calibration without bins |
| L243 | `ValueError` | `ConfigurationError` | Mondrian prediction without bins |

**Migration Code**:
```python
# v0.9.x
try:
    calibrator.predict_proba(X)
except ValueError as e:
    logger.error(f"Mondrian config error: {e}")

# v0.10.0+
try:
    calibrator.predict_proba(X)
except ConfigurationError as e:
    logger.error(f"Mondrian config error: {e}")
    # e.details contains {"context": "predict_proba", "requirement": "bins parameter"}
```

#### IntervalRegressor (`src/calibrated_explanations/calibration/interval_regressor.py`)
| Location | Old Exception | New Exception | Context |
|----------|---------------|---------------|---------|
| L171 | `ValueError` | `ConfigurationError` | Test bins without calibration bins |
| L182 | `ValueError` | `DataShapeError` | Bins length mismatch |
| L437 | `ValueError` | `ConfigurationError` | Mixing bins with no-bins |
| L439 | `ValueError` | `DataShapeError` | Bin count mismatch |

**Migration Code**:
```python
# v0.9.x
try:
    regressor.predict_intervals(X_test, bins=...)
except ValueError as e:
    logger.error(f"Shape or config error: {e}")

# v0.10.0+
try:
    regressor.predict_intervals(X_test, bins=...)
except (ConfigurationError, DataShapeError) as e:
    if isinstance(e, ConfigurationError):
        logger.error(f"Configuration conflict: {e.details}")
    else:
        logger.error(f"Shape mismatch: {e.details}")
```

### Plugins Module Changes

#### Base Plugin (`src/calibrated_explanations/plugins/base.py`)
| Old Exception | New Exception | Context |
|---------------|---------------|---------|
| `ValueError` | `ValidationError` | Plugin metadata validation failures |

**Migration Code**:
```python
# v0.9.x
try:
    validate_plugin_meta(plugin_dict)
except ValueError as e:
    logger.error(f"Invalid plugin: {e}")

# v0.10.0+
try:
    validate_plugin_meta(plugin_dict)
except ValidationError as e:
    logger.error(f"Invalid plugin: {e}")
    # e.details contains {"field": "...", "requirement": "..."}
```

#### Built-in Plugins (`src/calibrated_explanations/plugins/builtins.py`)
| Location | Old Exception | New Exception | Context |
|----------|---------------|---------------|---------|
| L184, L193 | `RuntimeError` | `NotFittedError` | Legacy interval context missing |
| L228 | `RuntimeError` | `ConfigurationError` | Unsupported model type |
| L242, L247 | `RuntimeError` | `NotFittedError` | Explainer not initialized |
| L357, L360 | `RuntimeError` | `NotFittedError` | Execution context missing |
| L742, L758, L766 | `RuntimeError` | `ConfigurationError` | PlotSpec payload mismatch |
| L1032 | `RuntimeError` | `ConfigurationError` | Renderer failure |
| L1073 | `RuntimeError` | `NotFittedError` | FAST interval context missing |

**Migration Code**:
```python
# v0.9.x
try:
    explainer.explain(X)
except RuntimeError as e:
    if "not fitted" in str(e):
        logger.error("Call fit() first")
    elif "unsupported" in str(e):
        logger.error("Model type not supported")

# v0.10.0+
try:
    explainer.explain(X)
except NotFittedError as e:
    logger.error(f"Not fitted: {explain_exception(e)}")
except ConfigurationError as e:
    logger.error(f"Config error: {explain_exception(e)}")
```

### Utils Module Changes

#### Helper Functions (`src/calibrated_explanations/utils/helper.py`)
| Location | Old Exception | New Exception | Context |
|----------|---------------|---------------|---------|
| L206 | `RuntimeError` | `NotFittedError` | `check_is_fitted()` state violation |

**Migration Code**:
```python
# v0.9.x
try:
    check_is_fitted(estimator)
except RuntimeError as e:
    logger.error(f"Not ready: {e}")

# v0.10.0+
try:
    check_is_fitted(estimator)
except NotFittedError as e:
    logger.error(f"Not ready: {explain_exception(e)}")
```

---

## Exception Type Summary

### Import Pattern (v0.10.0+)
```python
from calibrated_explanations.core.exceptions import (
    CalibratedError,  # Base class
    ValidationError,  # Input validation failures
    DataShapeError,   # Shape/dtype mismatches
    ConfigurationError,  # Configuration conflicts
    ModelNotSupportedError,  # Missing model methods
    NotFittedError,  # Fit-state violations
    ConvergenceError,  # Optimization failures
    SerializationError,  # Serialization failures
    explain_exception,  # Helper to format exceptions
)
```

### Exception Hierarchy
```
CalibratedError (base Exception)
├── ValidationError
│   ├── DataShapeError  # specializes ValidationError
├── ConfigurationError
├── ModelNotSupportedError
├── NotFittedError
├── ConvergenceError
└── SerializationError
```

---

## Parameter Guardrails (New in v0.10.0)

### Mutual Exclusivity Enforcement
The following parameter combinations are now rejected:

| Exclusive Group | Behavior | Details |
|-----------------|----------|---------|
| `threshold`, `confidence_level` | Cannot both be specified | Raises `ConfigurationError` with conflict details |

**Migration Code**:
```python
# v0.9.x - silently ignored conflicting params
explainer = CalibratedExplainer(
    ...,
    threshold=0.5,
    confidence_level=0.9,  # Which one wins? Undefined!
)

# v0.10.0+ - explicit error
try:
    explainer = CalibratedExplainer(
        ...,
        threshold=0.5,
        confidence_level=0.9,
    )
except ConfigurationError as e:
    logger.error(f"Choose one: {e.details['provided']}")
    # e.details = {
    #     "conflict": ("threshold", "confidence_level"),
    #     "provided": ["threshold", "confidence_level"],
    #     "requirement": "choose one or none"
    # }
```

### Parameter Validation API (New)
```python
from calibrated_explanations.api.params import validate_param_combination

# In your custom code:
user_kwargs = {"threshold": 0.5, "confidence_level": 0.9}

try:
    validate_param_combination(user_kwargs)
except ConfigurationError as e:
    logger.error(f"Invalid combination: {e}")
```

---

## Error Diagnostics Usage (New in v0.10.0)

### explain_exception() Helper
```python
from calibrated_explanations.core.exceptions import explain_exception, CalibratedError

try:
    calibrator.fit(X, y)
except CalibratedError as e:
    # Human-readable multi-line output
    print(explain_exception(e))
    # Output example:
    # ConfigurationError: X shape [100, 5] does not match expected [100, 10]
    #   Details: {'context': 'fit', 'param': 'X', 'expected_shape': (100, 10), 'actual_shape': (100, 5)}
```

### Structured Details Access
```python
from calibrated_explanations.core.exceptions import CalibratedError

try:
    calibrator.fit(X, y)
except CalibratedError as e:
    # Access structured diagnostic data
    context = e.details.get("context")
    param = e.details.get("param")
    requirement = e.details.get("requirement")

    # Log structured context for monitoring/alerting
    logger.error("Calibration error", extra={
        "error_type": type(e).__name__,
        "message": str(e),
        "context": context,
        "param": param,
    })
```

---

## Validation Helper Functions (Public API)

### Core Validation Functions
All validation functions are now available for direct use:

```python
from calibrated_explanations.core.validation import (
    validate_inputs,       # Validate X, y inputs
    validate_model,        # Validate model structure
    validate_fit_state,    # Check if estimator is fitted
    infer_task,           # Infer task type from data
    validate,             # Conditional guard clause
)

# Example: Conditional validation in custom code
from calibrated_explanations.core.exceptions import ValidationError

def my_function(X):
    validate(X is not None, ValidationError, "X cannot be None")
    validate(X.shape[0] > 0, ValidationError, "X must have at least 1 sample")
    # Function proceeds only if all validations pass
```

---

## Recommended Catch Patterns (v0.10.0+)

### Pattern 1: Broad Catch (Recommended for Most Code)
```python
from calibrated_explanations.core.exceptions import CalibratedError

try:
    result = explainer.explain(X)
except CalibratedError as e:
    logger.error(explain_exception(e))
```

### Pattern 2: Specific Exception Handling
```python
from calibrated_explanations.core.exceptions import (
    NotFittedError,
    ConfigurationError,
    ValidationError,
)

try:
    result = explainer.explain(X)
except NotFittedError as e:
    logger.error("Call fit() first")
except ConfigurationError as e:
    logger.error(f"Fix configuration: {e.details.get('requirement')}")
except ValidationError as e:
    logger.error(f"Invalid input: {e.details.get('param')}")
```

### Pattern 3: Structured Logging
```python
from calibrated_explanations.core.exceptions import CalibratedError

try:
    result = explainer.explain(X)
except CalibratedError as e:
    logger.error(
        "Calibrated explanation failed",
        extra={
            "error_type": type(e).__name__,
            "message": str(e),
            **e.details,  # Unpack structured details
        }
    )
```

---

## Backward Compatibility Notes

### What Changed
- Exception **types** changed
- Exception **messages** unchanged
- Exception **contexts** unchanged (same errors raised in same places)
- Exception **behavior** unchanged (same failure conditions)

### What Didn't Change
- ✅ Successful code paths unaffected
- ✅ Function signatures
- ✅ Return types
- ✅ Data flow
- ✅ Calibration algorithms
- ✅ Public API contracts

### When Code Breaks
Code breaks **only if** it:
1. Explicitly catches `ValueError` or `RuntimeError` from calibrated_explanations
2. Relies on `isinstance(e, ValueError)` or `isinstance(e, RuntimeError)` checks
3. Uses `.args[0]` to access error messages (still works, but use `str(e)` instead)

---

## Timeline

### v0.10.0 (Current)
- ADR-002 exceptions deployed
- Legacy exceptions replaced
- Parameter guardrails active
- 23 regression tests added

### v0.11.0 (Planned)
- Deprecation warnings for old catch patterns (optional)
- Extended guardrails for domain-specific constraints

### v1.0.0+ (Future)
- Legacy exception types may be removed if no evidence of external usage

---

## Support & Questions

### FAQ

**Q: Will my code break when I upgrade to v0.10.0?**
A: Only if you explicitly catch `ValueError` or `RuntimeError` from calibrated_explanations. Update those catch blocks to use ADR-002 exception types.

**Q: Can I continue catching `Exception`?**
A: Yes, all ADR-002 exceptions inherit from `Exception`. Generic `except Exception:` catches will continue working.

**Q: How do I access error details?**
A: Use `e.details` dict (e.g., `e.details['context']`). Use `explain_exception(e)` for human-readable formatting.

**Q: Are parameter guardrails backward compatible?**
A: Yes. Previously conflicting parameters silently failed; now they raise `ConfigurationError`. This is an improvement in clarity, not a breaking change.

**Q: Can I test the new exceptions?**
A: Yes. Test files are provided:
  - `tests/integration/test_exception_parity_calibration.py`
  - `tests/integration/test_exception_parity_plugins.py`
  - `tests/unit/core/test_validation_helpers.py`
  - `tests/unit/api/test_param_guardrails.py`

---

## Related Documentation

- **ADR-002**: `improvement_docs/adrs/ADR-002-validation-and-exception-design.md`
- **Exception Hierarchy**: `src/calibrated_explanations/core/exceptions.py`
- **Validation API**: `src/calibrated_explanations/core/validation.py`
- **Parameter Management**: `src/calibrated_explanations/api/params.py`
- **Completion Report**: `improvement_docs/adr mending/ADR-002/COMPLETION_REPORT.md`
