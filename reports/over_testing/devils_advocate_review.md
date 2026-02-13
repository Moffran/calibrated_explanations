# DEVILS-ADVOCATE: Consolidated Risk Assessment

## Review of Pruner Proposal

### What the pruner got RIGHT
- Correctly identified the 1-context data quality caveat as the critical blocker
- All 42 generated tests ARE import-only placeholders -- classification is correct
- The 3 already-skipped tests are safe to fully delete
- The overlap analysis between focused/auto_approved and unit tests is accurate

### CRITICAL FLAW: Coverage impact is NOT "negligible"

The pruner claims removing 42 generated tests + `test_force_mark_lines_for_coverage` is safe because impact is "NEGLIGIBLE". **This is dangerously wrong.**

**Worst-case calculation:**
- `test_force_mark_lines_for_coverage` pads **991 lines** (confirmed: it `exec(compile("pass\n"*600, source_path, "exec"))` for 3 source files)
- 42 generated tests each `importlib.import_module()` a module, executing all module-level code. Module-level code includes imports, class definitions, constant assignments, and decorator registration -- easily **20-50 lines per module** for this codebase
- Worst case: 991 + 42*20 = **1,831 lines** lost
- New coverage: 13,095/16,242 = **80.6%** -- FAR below the 90% floor

**More realistic estimate:**
- The force_mark test's 991 "unique_lines" is a 1-context artifact. The `exec` with fake filenames may only attribute ~600 lines per target file in coverage, not 991 truly unique lines
- Many generated test imports overlap with imports in hand-written tests
- Realistic loss: **300-800 lines** -- still potentially below 90%

**Mitigation**: Before removing ANY test, run `pytest --cov-fail-under=90` with those tests excluded to verify coverage holds. Never batch-remove without a verification run.

### RISK: Pruner didn't check evaluation/ and notebooks/
The pruner claims modules are "already tested elsewhere" but didn't verify whether `evaluation/*.py` (11 files) or `notebooks/*.ipynb` (15 files) exercise code paths not covered by the test suite. Some evaluation scripts use the library in ways tests may not cover.

### RISK: "Keep focused, remove unit" is backwards
The pruner recommends keeping focused/auto_approved tests and removing older unit tests. But the unit tests are MORE comprehensive (e.g., `test_to_python_number` in test_explanation_unit.py tests 6 cases including np.bool_ and None, while the focused version tests fewer). **The focused versions are minimal -- they may not cover all edge cases.** A safer approach: keep the unit tests, add focused versions' unique edge cases, then delete focused.

### RISK: Setup/teardown `inf` scores are misleading
The 50 `inf`-score tests include 15+ `|setup` and `|teardown` contexts. These are NOT separate tests -- they're pytest lifecycle hooks for real tests. Flagging them for removal is a misunderstanding of pytest-cov context naming.

---

## Review of Deadcode-Hunter Proposal

### What the deadcode-hunter got RIGHT
- Correctly identified that `_missing_` is a Python enum magic method (NOT dead code)
- Accurately classified all large uncovered blocks as untested production code, not dead code
- Properly checked lazy imports and found no hidden dead code
- Honest conclusion: "The deadcode-hunter has little to report" -- intellectual honesty is valuable

### MINOR GAP: Deprecation code not analyzed
The lazy import layer in `__init__.py` supports deprecated symbols (e.g., `from calibrated_explanations import plotting`). The deadcode-hunter noted this but didn't analyze whether the DEPRECATED CODE ITSELF has dead branches. Some deprecation paths may never be triggered in practice.

### GAP: testing/ module analysis was superficial
`testing/parity_compare.py` has 77 uncovered lines. The deadcode-hunter noted this but didn't investigate whether the parity tests are actually running. If they're not, the testing module may be partially dead.

### GAP: No analysis of the `perf/` module
`src/calibrated_explanations/perf/` is listed as "deprecated" in the project structure. The deadcode-hunter didn't check whether this module is genuinely deprecated and slated for removal.

---

## Consolidated Risk Assessment

### Per-Change Risk Ratings

| Change | Risk | Justification |
|--------|------|---------------|
| Delete 42 generated placeholder tests | **MEDIUM** | Coverage impact unknown without verification run. Could drop below 90% if module-level code contributes significantly. |
| Delete `test_force_mark_lines_for_coverage` | **HIGH** | Pads 991+ lines of coverage. Removing it WILL drop coverage significantly. Must be paired with real tests for the 3 target files (plotting.py, _helpers.py, explanations.py). |
| Delete 3 already-skipped tests | **LOW** | Already skipped, zero impact. Safe. |
| Consolidate overlapping unit tests | **MEDIUM** | Need to verify focused versions cover all edge cases before removing comprehensive unit tests. |
| Remove dead src code (none found) | **N/A** | No dead code to remove. |

### Mitigations

1. **Before any removal**: Run `pytest --cov-fail-under=90` with candidate tests excluded (use `--ignore` or `--deselect`) to verify coverage holds
2. **For force_mark removal**: Write real behavioral tests for `plotting.py`, `core/explain/_helpers.py`, and `explanations/explanations.py` BEFORE removing the padding test. These have large uncovered blocks (458, unknown, 168 lines respectively)
3. **For overlap consolidation**: Manually diff each focused/unit test pair to ensure no edge cases are lost before removing the "extra" version
4. **For all batches**: Commit removals and additions in the same commit so coverage never dips between changes

### Recommended Execution Order (safest first)

1. **Delete 3 already-skipped tests** (LOW risk, zero impact)
2. **Run `pytest --cov-context=test`** to generate reliable per-test data (PREREQUISITE for everything else)
3. **Write real tests** for plotting.py, _helpers.py, explanations.py to replace force_mark coverage
4. **Delete `test_force_mark_lines_for_coverage`** (only after step 3)
5. **Delete 42 generated tests** one batch at a time with coverage verification after each batch
6. **Consolidate overlapping tests** after confirming edge case coverage

### No-Go List

| Change | Reason |
|--------|--------|
| Bulk-delete generated tests without coverage verification | Could drop below 90% |
| Delete force_mark test without replacement tests | WILL drop below 90% |
| Remove any unit test in favor of focused version without edge case audit | May lose regression protection |
| Trust estimator `inf` scores for removal decisions | Data is from 1-context run |
| Remove `_missing_` from reject.py | It's a Python enum magic method, not dead code |

### Blind Spots Neither Agent Considered

1. **evaluation/ and notebooks/ coverage**: 11 evaluation scripts + 15 notebooks use the library. None of their code paths are reflected in coverage data. Removing tests that happen to cover these paths could break demos.
2. **The `perf/` deprecated module**: Neither agent investigated whether `src/calibrated_explanations/perf/` can be removed.
3. **Cross-Python-version conditional code**: Some code may only execute on Python 3.10 vs 3.13. The coverage data is from a single Python version.
4. **The force_mark test exists for a reason**: It was deliberately added to pass CI. Removing it without replacing the coverage it provides will fail CI. The pruner calling it "coverage fraud" is morally correct but practically dangerous without a replacement plan.
5. **Windows path issues**: `gaps.csv` uses forward slashes while the OS uses backslashes. Scripts may have path compatibility issues not caught by the analysis.
