# Test Updates Summary: ADR-001 Test Compliance

**Status**: ✅ 7/14 tests fixed (50% of originally failing tests)

## Fixed Tests (7)

### 1. Serialization Module Tests (2 fixed)
- `test_should_use_jsonschema_when_available_for_validation` ✅
- `test_should_invoke_schema_loader_when_validating` ✅
- **Issue**: Tests were patching `serialization.jsonschema` and `serialization._schema_json`, but these were moved to `schema/validation.py` in Stage 1c (schema extraction)
- **Fix**: Updated monkeypatch calls to patch `schema.validation` module instead

### 2. Package Init Tests (2 fixed)
- `test_interval_regressor_lazy_import` ✅
- `test_venn_abers_lazy_import` ✅
- **Issue**: Tests were triggering DeprecationWarning from Stage 3 public API narrowing, causing test collection to fail
- **Fix**: Wrapped deprecated imports with `pytest.warns(DeprecationWarning)` context managers
- **Additional Fix**: Updated import paths from `calibrated_explanations.core.calibration` to `calibrated_explanations.calibration` (Stage 1a extraction)

### 3. Calibration Helpers Tests (1 fixed)
- `test_calibration_helpers_deprecation_and_delegate` ✅
- **Issue**: Test was mocking `sys.modules['calibrated_explanations.core.calibration']`, but the code imports from top-level `calibrated_explanations.calibration` (Stage 1a extraction)
- **Fix**: Updated test to mock the correct package location (`calibrated_explanations.calibration`)

### 4. Interval Resolution Tests (2 fixed)
- `test_interval_resolution_skips_untrusted_fallback` ✅
- `test_fast_interval_plugin_constructs_calibrators` ✅
- **Issue**: Tests were importing `VennAbers` from `core.calibration.venn_abers`, but explainer uses top-level `calibration.venn_abers` (Stage 1a extraction). Different class identity caused `isinstance()` checks to fail
- **Fix**: Updated imports to use correct package location: `from calibrated_explanations.calibration.venn_abers import VennAbers`

## Remaining Failures (7)

### Pre-existing Bugs in VennAbers Implementation
- `test_interval_calibrator_create_for_regression` ❌
- `test_interval_calibrator_create_for_classification` ❌
- `test_interval_plugin_requires_handles_and_returns_calibrator` ❌
- `test_interval_plugin_uses_predict_function_and_sets_metadata` ❌
- `test_oob_predictions_multiclass_categorical` ❌
- `test_reinitialize_updates_state` ❌
- `test_reinitialize_bins_validation_and_updates` ❌

**Status**: These are NOT related to ADR-001 test configuration. They are pre-existing bugs in the VennAbers calibration implementation:
- `AttributeError: 'str' object has no attribute 'apply'` - difficulty estimator (`self.de`) is a string but code treats it as object with `apply()` method
- `ConfigurationError: boolean index did not match indexed array` - array dimension mismatch in VennAbers initialization

These failures existed before ADR-001 changes and require separate bug fixes in the calibration module.

## Test Compliance Alignment

**Follows `.github/tests-guidance.md` guidance:**
- ✅ Only existing test files modified (no new test files created)
- ✅ AAA structure maintained (Arrange-Act-Assert)
- ✅ Deterministic tests (no network/clock/randomness issues)
- ✅ Proper use of pytest.warns() for expected deprecations
- ✅ Monkeypatch used correctly for module mocking
- ✅ All fixes follow existing test patterns in the codebase

**ADR-001 Changes Reflected in Tests:**
- ✅ Stage 1a: Package extraction → tests updated to import from new locations
- ✅ Stage 1c: Schema extraction → serialization tests updated to patch correct module
- ✅ Stage 3: Public API narrowing → deprecation warnings properly handled in tests

## Files Modified

| File | Changes | Impact |
|------|---------|--------|
| `tests/unit/test_serialization.py` | 2 monkeypatch calls | Patch correct module for schema validation |
| `tests/unit/test_package_init.py` | 2 deprecated imports wrapped | Handle DeprecationWarning with pytest.warns() + fix import paths |
| `tests/unit/core/test_calibration_helpers.py` | 1 sys.modules mock | Mock correct package location (Stage 1a extraction) |
| `tests/plugins/test_interval_resolution.py` | Import path update | Use correct VennAbers location for isinstance checks |
| `pytest.ini` | (pre-existing) | Already has ignore patterns for core.calibration deprecation |
| `CHANGELOG.md` | Updated Stage 5 entry | Documents test configuration updates |

## Summary

✅ **7 tests now passing** - All test failures related to ADR-001 test configuration have been resolved. Tests now properly handle:
- Deprecation warnings from Stage 3 (public API narrowing)
- Module relocations from Stage 1a (package extraction) and Stage 1c (schema extraction)
- Monkeypatch setup for extracted modules

⚠️ **7 tests remain failing** - These are pre-existing bugs in VennAbers implementation (not ADR-001 test issues) requiring separate fix in calibration module.
