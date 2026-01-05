> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-023: Matplotlib Adapter Coverage Exemption

**Status:** Accepted
**Date:** 2025-10-17
**Authors:** Core team
**Context:** v0.8.0 test infrastructure stabilization

## Context

During v0.8.0 release testing, the matplotlib visualization adapter (`src/calibrated_explanations/viz/matplotlib_adapter.py`) exhibited systematic test failures when executed with pytest-cov coverage instrumentation enabled. The failures manifested as:

```python
AttributeError: module 'matplotlib' has no attribute 'artist'
```

### Root Cause Analysis

1. **Matplotlib lazy loading mechanism**: Matplotlib 3.8.4 uses a `@functools.cache` decorated `__getattr__` method to lazily import submodules (`image`, `axes`, `artist`, `pyplot`) on first access.

2. **pytest-cov instrumentation timing**: The pytest-cov plugin instruments code **before** pytest loads any conftest.py files or test modules, creating a timing conflict.

3. **Failure pattern**:
   - Individual viz tests: **PASS** ✅ (when run with `pytest tests/unit/viz/test_*.py`)
   - Full test suite with coverage: **FAIL** ❌ (29 failures out of 639 tests)
   - Full test suite without coverage: **PASS** ✅ (with `pytest --no-cov`)

4. **Technical details**:
   - Stack trace: `matplotlib_adapter.py:103` → `import matplotlib.pyplot as plt` → `pyplot.py:691` → `@_copy_docstring_and_deprecators(matplotlib.artist.getp)` → `__getattr__('artist')` → AttributeError
   - The coverage instrumentation triggers matplotlib's lazy `__getattr__` before submodules are initialized
   - Multiple preloading strategies attempted (9 iterations across 6 files) failed due to pytest-cov's early instrumentation

### Investigation History

- Attempted matplotlib downgrade from 3.9.2 → 3.8.4
- Implemented module-level preloading in multiple conftest.py files
- Added pytest_configure hooks for early matplotlib imports
- Preloaded `matplotlib.{image,axes,artist,pyplot}` in 6 different locations
- All preloading strategies ineffective due to pytest-cov instrumentation timing

## Decision

**Exempt `src/calibrated_explanations/viz/matplotlib_adapter.py` from coverage reporting** while keeping all viz tests enabled in the test suite. Viz tests execute normally with `pytest --no-cov` and validate functionality; they are skipped in CI coverage runs.

### Rationale

1. **Test suite integrity**: All 639 tests pass when run without coverage (`pytest --no-cov`)
2. **Coverage threshold met**: Package-wide coverage (excluding matplotlib_adapter.py) reaches 85%+, meeting Standard-003 requirement
3. **Tests validate functionality**: Viz tests run normally without coverage instrumentation and validate behavior
4. **CI workflow**: CI runs tests with `pytest --no-cov -m viz` for viz validation, and `pytest` for coverage on remaining modules
5. **Isolated impact**: Only affects one adapter module (matplotlib_adapter.py), not the broader viz subsystem
6. **Pragmatic resolution**: The issue is a conflict between two external dependencies (matplotlib + pytest-cov), not a defect in our codebase
7. **Alternative approaches exhausted**: Technical investigation demonstrated no viable fix short of forking matplotlib or pytest-cov

### Implementation

Modified `.coveragerc`:

```ini
omit =
    # ... existing omissions ...
    # Exclude viz adapter from coverage due to matplotlib lazy loading conflicts
    # with pytest-cov instrumentation. See ADR-023 for rationale.
    src/calibrated_explanations/viz/matplotlib_adapter.py
```

## Consequences

### Positive

- ✅ All 639 tests pass when run without coverage (`pytest --no-cov`)
- ✅ Coverage threshold (85%) maintained for non-viz modules
- ✅ No changes to production code required
- ✅ Viz functionality fully tested and validated
- ✅ CI can run viz tests separately without coverage

### Negative

- ⚠️ One module exempted from coverage metrics (matplotlib_adapter.py)
- ⚠️ Manual review required for changes to matplotlib_adapter.py
- ⚠️ CI requires two test runs: `pytest` for coverage, `pytest --no-cov -m viz` for viz validation

### Neutral

- Viz tests run successfully without coverage instrumentation
- Issue may resolve naturally when matplotlib 4.x addresses lazy loading architecture
- Alternative: Could re-enable coverage if matplotlib releases fix or pytest-cov adjusts hook timing

## Monitoring & Review

- **Review trigger**: Any matplotlib major version upgrade (3.x → 4.x)
- **Re-evaluation**: When pytest-cov implements pre-instrumentation hooks
- **Maintenance**: Document split test workflow in contributor guide and CI configuration
- **Testing**: Run `pytest --no-cov -m viz` to validate viz tests; run `pytest` for coverage on remaining modules

## Alternatives Considered

1. **Further matplotlib downgrade (3.8.4 → 3.7.x)**: Risk of breaking API compatibility; 3.7.x EOL status
2. **Custom pytest plugin for early loading**: Complex implementation, uncertain success due to pytest-cov hook priority
3. **Disable viz tests entirely**: Loses validation of critical user-facing functionality
4. **Skip failing tests with marks**: Hides real functionality, reduces test coverage to 95.1%
5. **Fork matplotlib or pytest-cov**: Unsustainable maintenance burden

## References

- [Matplotlib Issue #28242](https://github.com/matplotlib/matplotlib/issues/28242) - Lazy loading AttributeError reports
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/en/latest/) - Coverage plugin behavior
- Standard-003: Test Coverage Standard (85% threshold requirement)
- Test infrastructure investigation: October 2025

---

**Supersedes:** None
**Superseded by:** None (active)
**Related:** Standard-003 (Test Coverage Standard)
