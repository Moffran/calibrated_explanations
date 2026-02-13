# PROCESS-ARCHITECT PROPOSAL: Test Quality Improvement Process Redesign

## 1. Current Process Gap Analysis

### What the process prescribes (documented workflow)
`docs/improvement/archived/over_testing_method.md` and `finalize_over_testing.md` define a 5-step cycle:
1. Run heavy pipeline once with `--cov-context=test` -> produce per-test CSVs
2. Use estimator.py to rank low-value tests
3. Batch-remove zero-unique tests after estimator validation
4. Fill coverage gaps with focused tests
5. Verify with full pipeline run

### What was actually done (gaps)
| Prescribed Step | Actual State | Gap |
|-----------------|-------------|-----|
| Run `pytest --cov-context=test` | Run with only 1 context | **CRITICAL**: All per-test density data is unreliable |
| estimator.py recommends removals | Recommends 50 tests with `inf` scores | Scores are 0/0 artifacts, not real |
| Batch-remove zero-unique tests | 212 generated tests moved to backup | Removal was by keep-every-6th heuristic, not data-driven |
| Fill gaps with focused tests | 2 focused + 4 auto_approved tests added | Only 6 tests vs 247 gap blocks |
| Verify with full pipeline | Not done after bulk operations | No authoritative post-removal verification |

### Critical Gap: `--cov-context=test` not integrated anywhere standard
- NOT in `pyproject.toml` pytest addopts
- NOT in Makefile test targets
- NOT in any CI workflow
- Only exists in `run_over_testing_pipeline.py` which was apparently never run successfully
- The `triage.md` report explicitly warns: "Only one coverage context detected"

---

## 2. Script Audit Summary

### Over-Testing Scripts (scripts/over_testing/)

| Script | Status | Issues |
|--------|--------|--------|
| `estimator.py` | **Works** | Well-designed, conservative. No issues. |
| `gap_analyzer.py` | **Works** | Good gap detection. Flexible column name matching. |
| `generate_test_templates.py` | **Works but problematic** | Generates import-only placeholders (the 42 tests now flagged for removal). Template quality is too low. |
| `extract_per_test.py` | **Works** | Correctly extracts from .coverage contexts -- but requires `--cov-context=test` data to be useful. |
| `prune_generated_tests.py` | **Design flaw** | Conservative heuristic classifies ALL tests with `assert` as "questionable" -- but `assert isinstance(mod, types.ModuleType)` is NOT a meaningful assertion. Needs assertion quality threshold, not just presence. |
| `evaluate_cov_fill_adr030.py` | **Design flaw** | `has_assertion` check counts `isinstance` checks as real assertions. Should distinguish import-only assertions from behavioral assertions. |
| `coverage_xml_gaps.py` | **Redundant** | Overlaps with `gap_analyzer.py`. Both find uncovered blocks, one from XML, one from CSV. |
| `over_testing_report.py` | **Works** | Well-designed with `--require-multiple-contexts` guard. Correctly refuses to analyze 1-context data. |
| `over_testing_triage.py` | **Works** | Good hotspot ranking. Properly warns about 1-context data. |
| `run_over_testing_pipeline.py` | **Works** | Good orchestrator. Defaults to `--cov-context=test`. But was apparently never run to completion. |
| `inspect_coverage_db.py` | **Utility** | 13-line diagnostic. Fine as-is. |

### Anti-Pattern Scripts (scripts/anti-pattern-analysis/)

| Script | Status | Issues |
|--------|--------|--------|
| `detect_test_anti_patterns.py` | **Works** | Wrapper shim for real implementation. Fine. |
| `analyze_private_methods.py` | **Works well** | Found 349 definitions, clean categorization. Only 1 false-positive Pattern 3. |
| `scan_private_usage.py` | **Works** | Good allow-list integration with expiry support. |
| `generate_triage_report.py` | **Works** | Produces prioritized remediation list. |
| `summarize_analysis.py` | **Works** | Good dashboard output. |
| `find_shared_helpers.py` | **Not audited in detail** | Appears to find shared test helper patterns. |
| `update_allowlist.py` | **Not audited in detail** | Appears to update the private member allowlist. |
| `analyze_category_a.py` | **Not audited in detail** | Appears to analyze Category A internal logic testing. |

---

## 3. Redesigned Workflow

### Phase 0: Fix the Prerequisite (ONE-TIME)
```bash
# Generate reliable per-test coverage contexts
python scripts/over_testing/run_over_testing_pipeline.py \
  --pytest-args --cov-context=test -q \
  --continue-on-failure

# This will:
# 1. Run pytest with --cov-context=test (slow: expect 2-5x normal runtime)
# 2. Generate over_testing_report with multiple contexts
# 3. Generate over_testing_triage with hotspot data
```

**Expected output**: New `reports/over_testing/` data with reliable per-test unique line counts, hotspot analysis, and over-testing ratios.

**Time estimate**: If normal pytest takes 5 min, expect 10-25 min with `--cov-context=test`.

### Phase 1: Data-Driven Analysis (REPEATABLE)
```
Step 1.1: Run extract_per_test.py to produce reliable per_test_summary.csv
Step 1.2: Run gap_analyzer.py to identify untested blocks
Step 1.3: Run analyze_private_methods.py to find dead code
Step 1.4: Run detect_test_anti_patterns.py for quality scan
Step 1.5: Run estimator.py --recommend to rank low-value tests
```

### Phase 2: Triage and Proposal
```
Step 2.1: Classify recommended removals into batches (safe/medium/risky)
Step 2.2: For each batch, estimate coverage impact with estimator.py --remove-list
Step 2.3: For any batch that drops below 90%, identify compensating tests
Step 2.4: Produce proposal document with per-batch risk ratings
```

### Phase 3: Review and Approval
```
Step 3.1: Devils-advocate review of proposal
Step 3.2: Manual review by maintainer
Step 3.3: Sign-off on removal batches and compensating tests
```

### Phase 4: Implementation
```
Step 4.1: Write compensating tests FIRST (before any removals)
Step 4.2: Verify compensating tests pass: pytest --cov-fail-under=90
Step 4.3: Apply removal batches one at a time
Step 4.4: After each batch: pytest --cov-fail-under=90
Step 4.5: Commit removals + additions together
```

### Phase 5: Verification
```
Step 5.1: Run full over_testing_pipeline with --cov-context=test
Step 5.2: Compare before/after metrics
Step 5.3: Update reports/over_testing/ with new baselines
Step 5.4: Update ratcheting thresholds
```

---

## 4. Quality Gate Definitions

| Gate | Metric | Threshold | Enforcement | Status |
|------|--------|-----------|-------------|--------|
| Coverage floor | Line coverage % | >= 90% | CI (pytest --cov-fail-under=90) | **Active** |
| Coverage floor | Branch coverage | Enabled, advisory | CI coverage report | **Active** |
| Private member scan | Non-allowed violations | 0 | CI (scan_private_usage.py --check) | **Active** |
| Anti-pattern scan | New violations | 0 (baseline + ratchet) | CI (detect_test_anti_patterns.py) | **Active** |
| Over-testing density | Over-testing ratio | Ratcheting baseline | CI (advisory -> enforced) | **NOT ACTIVE -- needs Phase 0** |
| Assertion quality | Tests without behavioral assertions | Ratcheting baseline | CI (extend evaluate_cov_fill_adr030.py) | **NOT ACTIVE** |
| Test runtime budget | Slow test total time | Advisory budget (TBD) | CI (pytest --durations) | **NOT ACTIVE** |

### New Gate: Assertion Quality
Extend `evaluate_cov_fill_adr030.py` to distinguish:
- **Behavioral assertion**: Asserts on a value computed by the system under test
- **Import-only assertion**: `assert isinstance(mod, types.ModuleType)` -- NOT a real assertion
- **Execution-only test**: No assertions at all -- just `func()` with no checks

---

## 5. Script Improvement Recommendations

### Fix (critical)
1. **`generate_test_templates.py`**: Stop generating import-only placeholders. Generate templates with `# TODO: Add behavioral assertions` and `pytest.fail("Not implemented")` instead of `assert isinstance(mod, types.ModuleType)`.
2. **`evaluate_cov_fill_adr030.py`**: Add assertion quality classification (behavioral vs import-only vs none). Current `has_assertion=True` for `assert isinstance` is misleading.
3. **`prune_generated_tests.py`**: Use assertion quality classification from evaluate_cov_fill_adr030.py. Reclassify import-only assertion tests as "removable" not "questionable".

### Merge (redundancy)
4. **Merge `coverage_xml_gaps.py` into `gap_analyzer.py`**: Both find uncovered blocks. gap_analyzer works on CSV, coverage_xml_gaps on XML. Add an `--xml` flag to gap_analyzer.

### Create (missing)
5. **Create `scripts/over_testing/verify_removal_batch.py`**: Given a list of tests to remove, run pytest with those tests excluded and verify coverage >= 90%. This is the missing safety net.
6. **Create `Makefile` target `test-over-testing`**: Add `make test-over-testing` that runs `run_over_testing_pipeline.py` with proper flags.

---

## 6. CI Integration Plan

### Every PR (fast, existing)
- Coverage gate (>= 90%) -- already active
- Private member scan -- already active
- Anti-pattern scan -- already active

### Every PR (new, fast)
- Assertion quality scan on new/modified test files only (extend lint.yml)
- Marker hygiene check (warn on tests without proper markers)

### Weekly/Monthly (slow, new)
- Full `run_over_testing_pipeline.py` with `--cov-context=test`
- Compare over-testing density to ratcheting baseline
- Produce updated triage.md and hotspot reports
- File issues for new hotspots above threshold

### On-demand (manual)
- Full pruning cycle (Phase 1-5 above)
- Dead code analysis
- Test overlap consolidation

---

## 7. Metrics and Feedback Loop

### Track These Metrics Over Time
1. **Total test count** (should decrease or stabilize, not grow unbounded)
2. **Coverage %** (should stay >= 90% with real tests, not padding)
3. **Over-testing ratio** (% of lines hit by > N tests, should decrease)
4. **Anti-pattern count** (should decrease via ratchet)
5. **Average test runtime** (should not grow unbounded)
6. **Generated test count** (should reach 0)

### Feedback Loop
After each pruning cycle:
1. Record before/after for all 6 metrics
2. If coverage dropped: were compensating tests written?
3. If over-testing ratio increased: are new tests overlapping existing ones?
4. If test count grew: are new tests behavioral or padding?
5. Publish metrics in `reports/over_testing/metrics_history.csv` (new file)

---

## 8. Priority Ordering

1. **Run `run_over_testing_pipeline.py` with `--cov-context=test`** (unblocks everything)
2. **Fix `evaluate_cov_fill_adr030.py` assertion quality** (needed for accurate triage)
3. **Fix `prune_generated_tests.py` heuristic** (needed for safe pruning)
4. **Fix `generate_test_templates.py` template quality** (prevent new import-only tests)
5. **Create `verify_removal_batch.py`** (safety net for removals)
6. **Add `make test-over-testing` target** (convenience)
7. **Merge `coverage_xml_gaps.py` into `gap_analyzer.py`** (reduce redundancy)
8. **Add assertion quality check to CI** (prevent regression)
9. **Enable over-testing density gate in advisory mode** (ADR-030 Phase 4)
10. **Establish metrics tracking and ratcheting baselines** (long-term health)

---

## 9. Handling the Generated Test Backlog

### Immediate action (42 remaining files)
All 42 files are import-only placeholders. Process:
1. Verify coverage holds without them: `pytest --ignore=tests/generated --cov-fail-under=90`
2. If coverage holds: delete all 42 files
3. If coverage drops: identify which modules lose coverage, write 1 behavioral test per module, then delete
4. Delete the `tests/generated/` directory entirely once empty

### Preventing future backlog
- Fix `generate_test_templates.py` to produce `pytest.fail("Not implemented")` templates
- Never merge generated tests without manual review
- Add CI check that rejects tests matching the import-only pattern
