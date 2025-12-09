# ADR-002 Validation Parity: Implementation Plan

**Date Started**: 2025-11-29
**Target Release**: v0.10.0
**Status**: IN PROGRESS

## Scope

Deliver ADR-002 validation parity by:
1. Replacing legacy `ValueError`/`RuntimeError` with taxonomy classes
2. Implementing shared validators with reusable entry points
3. Adding parameter guards and consistent fit-state handling
4. Implementing structured error payload helpers
5. Adding comprehensive regression tests

---

## Gap Analysis Summary

From `improvement_docs/ADR-gap-analysis.md` (L44-L48):

| Rank | Gap | Severity | Notes |
|------|-----|----------|-------|
| 1 | Legacy `ValueError`/`RuntimeError` usage | 20 (critical) | Core/plugins still raise generic exceptions instead of ADR taxonomy |
| 2 | Validation API contract not implemented | 16 (critical) | `validate_inputs` exposes generic signature, duplicates logic |
| 3 | Structured error payload helpers absent | 12 (high) | Missing `validate()` helper and `explain_exception()` utility |
| 4 | `validate_param_combination` is a no-op | 9 (high) | Parameter guardrails unimplemented |
| 5 | Fit-state and alias handling inconsistent | 6 (medium) | Wrapper normalisation diverges from `canonicalize_kwargs` |

---

## Implementation Tasks

### Task 1: Replace Legacy Exceptions in Core Paths
**Priority**: Critical (severity 20)
**Status**: IN PROGRESS

#### Calibration Paths (6 raises)
- `src/calibrated_explanations/calibration/venn_abers.py` L210, L243: ValueError → ConfigurationError
- `src/calibrated_explanations/calibration/interval_regressor.py` L171, L182, L437, L439: ValueError → ConfigurationError/DataShapeError

#### Plugin Paths (20+ raises)
- `src/calibrated_explanations/plugins/builtins.py`: RuntimeError → NotFittedError/ConfigurationError
- `src/calibrated_explanations/plugins/explanations.py`: ValueError → ValidationError/ConfigurationError
- `src/calibrated_explanations/plugins/registry.py`: ValueError → ValidationError/ConfigurationError
- `src/calibrated_explanations/plugins/base.py`: ValueError → ValidationError/ConfigurationError

#### Utility Paths (key raises)
- `src/calibrated_explanations/utils/helper.py` L206: RuntimeError → NotFittedError
- `src/calibrated_explanations/utils/discretizers.py`: ValueError → ValidationError/DataShapeError

**Mapping Logic**:
- Parameter/shape mismatches → `DataShapeError` (extends `ValidationError`)
- Configuration/compatibility errors → `ConfigurationError`
- Missing required state → `NotFittedError`
- Input validation failures → `ValidationError`
- Model capability mismatches → `ModelNotSupportedError`

### Task 2: Implement Shared Validation Entry Points
**Priority**: Critical (severity 16)
**Status**: PENDING

#### Current State
- `validate_inputs()` exists but delegates to `validate_inputs_matrix()`
- `validate_model()` checks for `predict` method
- `validate_fit_state()` checks `fitted` attribute
- All are in `core/validation.py`

#### Required Enhancements
1. **Ensure ADR-002 signatures are honoured**:
   - `validate_inputs(x, y=None, task="auto", allow_nan=False, require_y=False, n_features=None, class_labels=None, check_finite=True) -> None`
   - `validate_model(model) -> None`
   - `validate_fit_state(explainer, require=True) -> None`
   - `infer_task(x, y, model) -> Literal["classification","regression"]`

2. **Wire through wrappers** (`WrapCalibratedExplainer`):
   - Use `validate_fit_state()` before operations requiring fitted state
   - Use `validate_inputs_matrix()` for data entry points

3. **Wire through plugins**:
   - Explain plugins: call `validate_inputs_matrix()` in `create()`
   - Registry validation: replace ValueError with ValidationError/ConfigurationError

4. **Add optional `validate()` helper**:
   - Signature: `validate(condition, exc_cls, message, *, details=None) -> None`
   - Enable pattern: `validate(len(x) > 0, ValidationError, "x must not be empty")`

### Task 3: Add Structured Error Payload Helpers
**Priority**: High (severity 12)
**Status**: PENDING

#### Implement in `core/exceptions.py`
1. **`explain_exception(e: Exception) -> str`**:
   - Convert `CalibratedError` instances to human-readable multi-line messages
   - Include `details` dict in formatted output
   - Example:
     ```
     ValidationError: Argument 'x' must have 5 features, got 3.
       Details: {"param": "x", "expected_features": 5, "actual_features": 3}
     ```

2. **Error details payloads**:
   - Ensure `CalibratedError.__init__` accepts `details` kwarg (already done)
   - Document usage pattern for diagnostics

3. **Wire through surfaces**:
   - `core/prediction_helpers.py`: attach error context to calibration failures
   - Plugin explain/export flows: capture and log error payloads

### Task 4: Implement Parameter Guardrails
**Priority**: High (severity 9)
**Status**: PENDING

#### Current State
- `api/params.py` contains `ALIAS_MAP`, `canonicalize_kwargs()`, `warn_on_aliases()`
- `validate_param_combination()` exists but does nothing (no-op)

#### Required Implementation
1. **Implement `validate_param_combination()`**:
   - Check for conflicting parameter combinations
   - Example: cannot specify both `threshold` and `confidence_level` simultaneously
   - Raise `ConfigurationError` with descriptive message
   - Add `details` dict with conflicting params

2. **Document in migration notes**:
   - Create `improvement_docs/adr mending/ADR-002/MIGRATION_NOTES.md`
   - List all parameter guardrails enforced
   - Provide migration guidance for affected operators

### Task 5: Harmonize Fit-State and Alias Handling
**Priority**: Medium (severity 6)
**Status**: PENDING

#### Wrapper Alignment (`WrapCalibratedExplainer`)
1. **Use `NotFittedError` consistently**:
   - `_assert_fitted()` already uses `NotFittedError` ✓
   - `_assert_calibrated()` already uses `NotFittedError` ✓
   - Verify all wrapper check paths use ADR-002 exceptions

2. **Use `canonicalize_kwargs()` for parameter normalization**:
   - Ensure wrapper init/fit/calibrate/explain accept both canonical and aliased params
   - Wire through existing `warn_on_aliases()` calls

3. **Extend contract tests**:
   - Add tests verifying wrapper raises `NotFittedError` (not `RuntimeError`)
   - Add tests verifying aliases are normalized and warned

#### Plugin Alignment
1. **Built-in plugins** (`builtins.py`):
   - Replace RuntimeError with NotFittedError for state checks
   - Replace ValueError with ConfigurationError for config mismatches

2. **Explanation plugins** (`explanations.py`):
   - Replace ValueError with ValidationError/ConfigurationError
   - Add fit-state validation in `create()` methods

### Task 6: Add Regression Tests for All Flows
**Priority**: High
**Status**: PENDING

#### Test Coverage Plan

**Unit Tests** (`tests/unit/core/`):
- Exception hierarchy and repr (already exists in `test_exceptions.py`)
- Validation function signatures and error messages
- `explain_exception()` formatting
- `validate_param_combination()` guardrails

**Integration Tests** (`tests/integration/core/`):
- Calibration flows raise `ConfigurationError` for invalid bins
- Plugin creation fails with appropriate exceptions
- Wrapper fit/calibrate/explain enforce fit-state with `NotFittedError`

**Regression Tests** (new files):
- `tests/integration/test_exception_parity_calibration.py`:
  - VennAbers with missing bins → ConfigurationError
  - IntervalRegressor with mismatched bins → DataShapeError

- `tests/integration/test_exception_parity_plugins.py`:
  - Plugin initialization without state → NotFittedError
  - Invalid plugin metadata → ValidationError/ConfigurationError

- `tests/integration/test_exception_parity_prediction.py`:
  - Prediction on unfitted explainer → NotFittedError
  - Invalid data shapes → DataShapeError

#### Coverage Requirements
- Maintain 88% overall coverage (`pytest --cov-fail-under=88`)
- Focus on error paths (branches near exception raises)

---

## Implementation Progress

- [ ] Task 1: Replace legacy exceptions
- [ ] Task 2: Implement shared validation entry points
- [ ] Task 3: Add structured error payload helpers
- [ ] Task 4: Implement parameter guardrails
- [ ] Task 5: Harmonize fit-state and alias handling
- [ ] Task 6: Add regression tests

---

## Files to Modify

### Core Exceptions & Validation
- `src/calibrated_explanations/core/exceptions.py` (add helpers)
- `src/calibrated_explanations/core/validation.py` (enhance, add `validate()`)
- `src/calibrated_explanations/api/params.py` (implement `validate_param_combination()`)

### Calibration
- `src/calibrated_explanations/calibration/venn_abers.py` (replace ValueError)
- `src/calibrated_explanations/calibration/interval_regressor.py` (replace ValueError)

### Plugins
- `src/calibrated_explanations/plugins/base.py` (replace ValueError)
- `src/calibrated_explanations/plugins/builtins.py` (replace RuntimeError/ValueError)
- `src/calibrated_explanations/plugins/explanations.py` (replace ValueError)
- `src/calibrated_explanations/plugins/registry.py` (replace ValueError)

### Utilities
- `src/calibrated_explanations/utils/helper.py` (replace RuntimeError)
- `src/calibrated_explanations/utils/discretizers.py` (replace ValueError)

### Tests
- `tests/unit/core/test_exceptions.py` (extend)
- `tests/unit/core/test_validation_unit.py` (extend)
- `tests/integration/core/test_validation_integration.py` (extend)
- NEW: `tests/integration/test_exception_parity_calibration.py`
- NEW: `tests/integration/test_exception_parity_plugins.py`
- NEW: `tests/integration/test_exception_parity_prediction.py`

---

## ADR References

- **ADR-002**: Exception Taxonomy and Validation Design
  - Exception taxonomy: base `CalibratedError` with subclasses
  - Validation contracts: `validate_inputs`, `validate_model`, `validate_fit_state`, `infer_task`
  - Optional `validate()` helper for conditions

- **ADR-001**: Core Decomposition Boundaries (adherence check)
  - Exceptions live in `core/exceptions.py`
  - Validation lives in `core/validation.py`
  - Parameter canonicalization lives in `api/params.py`

---

## Completion Criteria

1. ✅ All legacy `ValueError`/`RuntimeError` in core/calibration/plugins replaced with ADR-002 taxonomy
2. ✅ Shared validators wired through wrappers, plugins, and entry points
3. ✅ `explain_exception()` helper implemented and documented
4. ✅ `validate_param_combination()` implements real guardrails
5. ✅ Wrapper fit-state consistently uses `NotFittedError`
6. ✅ Comprehensive regression tests added with >88% coverage
7. ✅ Migration notes document all breaking changes
8. ✅ ADR-002 mending report documents all changes
