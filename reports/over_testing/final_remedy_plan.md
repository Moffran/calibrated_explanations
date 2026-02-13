# Final Remedy Plan: Test Quality Improvement (Fact-Based)

## 2026-02-13 Execution Update (Current)

- Applied aggressive zero-unique pruning batch from `per_test_summary.csv` (`|run` contexts only), then iteratively kept only the removals that remained compatible with the 90% gate.
- Added targeted high-quality gap-closers instead of reintroducing low-value padding:
  - `tests/unit/testing/test_parity_compare.py`
  - `tests/unit/core/test_logging_context.py`
  - `tests/unit/test_ce_agent_utils.py`
  - `tests/unit/core/test_serialization_invariants_extra.py`
- Re-ran full method verification:
  - `pytest --cov-fail-under=90` -> **PASS**, total coverage **90.00%** (`968 passed, 1 skipped`)
  - `python scripts/over_testing/run_over_testing_pipeline.py` -> **PASS**
  - `python scripts/over_testing/extract_per_test.py` -> **PASS**
  - `python scripts/over_testing/detect_redundant_tests.py` -> **PASS**

### Current measured outcomes

| Metric | Value |
|---|---|
| Coverage gate | **90.00% PASS** |
| Per-test contexts | **1054** |
| Zero-unique `|run` tests | **98** |
| `<5` unique `|run` tests | **560** |
| Exact duplicate groups | **19** |
| Subset redundancy groups | **69** |
| Potential redundant tests (report) | **361** |
| Net removed test defs in this execution | **47** |

## 2026-02-13 Execution Update (Additional Iteration)

- Completed an additional over-testing reduction pass concentrated on low-signal redundant `|run` contexts and retained only changes compatible with the hard 90% coverage gate.
- Removed four redundant tests (all mapped by subset redundancy report) from:
  - `tests/plugins/test_cli_additional.py`
  - `tests/unit/utils/test_deprecations_helper.py`
  - `tests/unit/test_ce_agent_utils.py`
  - `tests/unit/core/test_validation_unit.py`
- Kept `tests/plugins/test_cli.py::test_cmd_list_intervals_empty` to preserve gate stability after attempted removal (coverage dipped to 89.98% without it).
- Re-verified end-to-end:
  - `pytest --cov=src/calibrated_explanations --cov-context=test --cov-fail-under=90` -> **PASS** (`954 passed, 1 skipped`, **90.00%**)
  - `python scripts/over_testing/run_over_testing_pipeline.py` -> **PASS**
  - `python scripts/over_testing/extract_per_test.py` -> **PASS**
  - `python scripts/over_testing/detect_redundant_tests.py` -> **PASS**

### Measured outcomes (additional iteration)

| Metric | Value |
|---|---|
| Per-test contexts | **1038** |
| Exact duplicate groups | **18** |
| Subset redundancy groups | **56** |
| Potential redundant tests (report) | **345** |
| Redundant `|run` contexts (in report) | **44** |

> Consolidated from: pruner, deadcode-hunter, process-architect, and devils-advocate analyses.
> Updated with **verified multi-context coverage data** (2,967 contexts, run 2026-02-12).
>
> **Baseline**: 16,079 statements | 1,217 missed | 90.39% coverage | 2,595 tests passed

---

## Iteration Update (2026-02-13)

### Batch execution

- Removed another **100 zero-unique tests** in clustered areas to maximize meaningful gap surfacing:
  - `tests/unit/core/test_calibrated_explainer_runtime_helpers.py` (40)
  - `tests/plugins/test_builtins_behaviour.py` (34)
  - `tests/plugins/test_registry_metadata_ext.py` (26)
- Added/expanded behavioral replacements:
  - `tests/unit/core/test_removal_batch_replacements.py`
  - `tests/unit/test_quick_adapters_and_shims.py`
  - `tests/unit/viz/test_smoke_coverage.py`
- Stabilized shim-forwarding tests against suite-level warning policy variance:
  - `tests/unit/test_perf_cache_shim.py`
  - `tests/unit/test_perf_parallel.py`
  - `tests/unit/test_perf_parallel_shim.py`

### Verification status

- `python scripts/anti-pattern-analysis/scan_private_usage.py --check`: **PASS**
- `pytest --tb=no -q`: **PASS** (`1670 passed, 1 skipped`)

### Process-architect conclusion for this iteration

- The current removal-plus-replacement workflow is **efficient as-is** and remains the right default.
- One operational hardening is now recommended as a standard step:
  - Make shim/deprecation tests policy-agnostic (assert forwarding behavior, not warning filter mechanics) to avoid full-suite flakiness.
- No structural redesign is required before the next 100-test batch.

---

## Executive Summary

The test suite has **no dead source code** (codebase is clean) but has significant **test quality debt**:

| Issue | Impact | Verified? |
|-------|--------|-----------|
| 42 import-only placeholder tests in `tests/generated/` | Zero behavioral value | **YES** -- removing them keeps coverage at **90.37%** (passes 90% gate) |
| 1 coverage-padding test (`test_force_mark_lines_for_coverage`) | Inflates **1,028 unique lines** artificially | **YES** -- removing it drops coverage to **89.88%** (fails 90% gate by 0.12%) |
| 2,511 tests with zero unique lines | Overlap/setup overhead | **YES** -- confirmed with 2,967-context data |
| 311 tests with 1-4 unique lines | Low-value candidates | **YES** -- confirmed with multi-context data |
| Over-testing hotspots: `helper.py` hit by 599 tests | Excessive redundancy | **YES** -- verified hotspot data |
| 3 already-skipped "batch1" tests | Dead weight | **YES** -- already marked with skip reason |

**Key insight**: The gap between 89.88% (without force_mark) and 90.00% is only **~20 real statement lines**. Writing a handful of behavioral tests for the padded modules will permanently close the gap.

---

## Verified Facts (from `--cov-context=test` run)

| Metric | Value |
|--------|-------|
| Total statements | 16,079 |
| Covered statements | 14,862 (90.39%) |
| Missed statements | 1,217 |
| Total test contexts | 2,967 |
| Tests with 0 unique lines | 2,511 (84.6% of all tests!) |
| Tests with 1-4 unique lines | 311 |
| `force_mark` unique lines | 1,028 |
| Coverage WITHOUT generated tests | **90.37%** (PASSES) |
| Coverage WITHOUT force_mark | **89.88%** (FAILS by 0.12%) |
| Top hotspot: `helper.py:72` | Hit by 599 tests |
| Top hotspot: `registry.py:108-112` | Hit by 597 tests |
| Most over-tested file | `sequential.py` (82.7% of lines hit by 20+ tests) |

---

## Phase 1: Safe Immediate Actions [VERIFIED SAFE]

### 1.1 Delete 3 already-skipped tests

| File | Reason |
|------|--------|
| [test_calibrated_explainer_additional_behavioral.py:177](tests/core/test_calibrated_explainer_additional_behavioral.py#L177) | "duplicative metadata accessor test" |
| [test_cache.py:659](tests/unit/perf/test_cache.py#L659) | "flaky perf import-mock test" |
| [test_narrative_generator.py:17](tests/unit/core/test_narrative_generator.py#L17) | "duplicate serialization negative-path test" |

**Risk**: LOW -- already skipped, zero coverage impact.

### 1.2 Delete 42 generated placeholder tests [VERIFIED SAFE]

**Verified**: `pytest --ignore=tests/generated --cov-fail-under=90` **PASSES at 90.37%**.

All 42 are identical `assert isinstance(mod, types.ModuleType)` placeholders. Safe to delete the entire `tests/generated/` directory.

**Risk**: LOW -- verified that coverage stays above 90%.

### 1.3 Remove `test_force_mark_lines_for_coverage` [REQUIRES ~20 LINES OF REAL COVERAGE]

**Verified**: Removing this single test drops coverage from 90.39% to **89.88%**.

This test `exec(compile("pass\n"*600, source_path, "exec"))` for 3 files:
- `plotting.py` (currently 81.4% covered -- 133 missed statements)
- `core/explain/_helpers.py` (currently 97.0% covered -- only 4 missed)
- `explanations/explanations.py` (currently 86.9% covered -- 67 missed statements)

**The gap is only 0.12% (~20 statement lines).** A single well-targeted behavioral test for `plotting.py` or `explanations/explanations.py` will close this gap.

**Action**:
1. Write 1-3 behavioral tests covering ~20 new statement lines in `plotting.py` or `explanations/explanations.py`
2. Verify `pytest --deselect tests/unit/core/test_calibrated_explainer_additional.py::test_force_mark_lines_for_coverage --cov-fail-under=90` passes
3. Delete the force_mark test

**Risk**: LOW once compensating tests cover ~20 statements.

---

## Phase 2: Address 2,511 Zero-Unique-Lines Tests

With reliable multi-context data, **2,511 tests** (84.6%) contribute zero unique lines. This does NOT mean they should all be removed -- most are legitimately testing behavior that overlaps with other tests. However, it reveals massive over-testing of core paths.

### 2.1 The 50 estimator-recommended `inf`-score tests

These are the strongest removal candidates -- tests contributing zero unique lines. Many are `|setup` and `|teardown` contexts (pytest lifecycle, not real tests), but several are actual test functions with genuinely zero unique coverage.

**Action**: Review each of the 50 `inf`-score test functions (not setup/teardown contexts) and determine:
- Is it testing a unique behavior despite sharing code paths?
- Is it truly redundant with another test?
- Is it a pure setup/teardown artifact?

### 2.2 Over-testing hotspot consolidation

The top over-tested files:

| File | Over-ratio | Lines hit by 20+ tests |
|------|-----------|----------------------|
| `core/explain/sequential.py` | 82.7% | 110 of 133 lines |
| `core/explain/feature_task.py` | 76.7% | 329 of 429 lines |
| `core/explain/_computation.py` | 76.3% | 222 of 291 lines |
| `core/explain/_legacy_explain.py` | 74.1% | 197 of 266 lines |
| `utils/discretizers.py` | 68.8% | 108 of 157 lines |
| `calibration/summaries.py` | 64.5% | 40 of 62 lines (max: **472 tests!**) |

**These are the real over-testing targets.** Use `hotspot_contexts.json` to identify which tests are hitting these files, then consolidate redundant tests.

### 2.3 Known overlapping test pairs

| Functions | Test Locations | Action |
|-----------|---------------|--------|
| `to_python_number` | focused/ + unit/explanations/ + unit/test_explanation_helpers_extra | Diff, keep most comprehensive |
| `derive_threshold_labels` | focused/ + unit/plotting/ + plugins/smoke + plugins/coverage | Diff, keep most comprehensive |
| `format_save_path` | focused/ + unit/plotting/ | Diff, keep most comprehensive |
| cache hash/size | auto_approved/ + unit/cache/ | Diff, keep most comprehensive |
| config split/coerce | auto_approved/ + unit/core/ | Diff, keep most comprehensive |

---

## Phase 3: Script and Process Improvements

### 3.1 Fix scripts (from process-architect audit)

| Script | Fix | Priority |
|--------|-----|----------|
| `evaluate_cov_fill_adr030.py` | Distinguish behavioral vs import-only assertions | **HIGH** |
| `prune_generated_tests.py` | Reclassify import-only-assertion tests as "removable" | **HIGH** |
| `generate_test_templates.py` | Use `pytest.fail("Not implemented")` instead of import-only templates | **HIGH** |
| `coverage_xml_gaps.py` | Merge into `gap_analyzer.py` with `--xml` flag | LOW |

### 3.2 New tooling

| Tool | Purpose |
|------|---------|
| `verify_removal_batch.py` | Verify coverage >= 90% after excluding a batch of tests |
| `make test-over-testing` | Convenience Makefile target for `run_over_testing_pipeline.py` |

### 3.3 CI additions

| Check | When | Mode |
|-------|------|------|
| Assertion quality scan on new tests | Every PR | Warn -> Block |
| Over-testing density report | Weekly scheduled | Advisory with ratchet |
| Marker hygiene | Every PR | Warn on missing markers |

---

## Phase 4: Long-Term Quality Health

### ADR-030 Phase Completion

| Phase | Status | Next |
|-------|--------|------|
| Phase 1: Baseline + "no new violations" | **Complete** | Maintain |
| Phase 2: Assertion/determinism/mocking checks | **In progress** | Fix evaluate_cov_fill |
| Phase 3: Marker hygiene and slow budgets | **Not started** | After Phase 2 |
| Phase 4: Over-testing density gate | **NOW UNBLOCKED** | Reliable data available |

### Metrics baseline (as of 2026-02-12)

| Metric | Value | Target |
|--------|-------|--------|
| Total tests | 2,595 | Reduce by ~50-100 |
| Coverage (real) | 90.39% | >= 90% without padding |
| Tests with 0 unique lines | 2,511 | Reduce through consolidation |
| Generated test count | 42 | 0 |
| Over-testing ratio (sequential.py) | 82.7% | < 50% |
| Max hotspot count | 599 (helper.py:72) | < 200 |

---

## Risk Registry (Updated with Facts)

| Change | Risk | Verified Impact |
|--------|------|-----------------|
| Delete 3 skipped tests | **LOW** | Zero coverage change |
| Delete 42 generated tests | **LOW** | Coverage stays at 90.37% (verified) |
| Delete force_mark test alone | **MEDIUM** | Drops to 89.88% (verified) -- needs ~20 lines of real tests first |
| Delete force_mark + compensating tests | **LOW** | Write ~20 lines of real coverage, then delete |
| Consolidate overlapping tests | **MEDIUM** | Diff each pair first |
| Enable over-testing density gate | **LOW** | Reliable data now available |

---

## No-Go List

| Action | Reason |
|--------|--------|
| Delete `force_mark` without compensating tests | Drops to 89.88% (verified) |
| Remove `_missing_` from `reject.py` | Python enum magic method, not dead code |
| Remove large uncovered src blocks | Untested production code, not dead code |
| Remove deprecated symbol support | Required until v0.11.0 per ADR-011 (now at v0.11.0-dev) |

---

## Execution Checklist

```
IMMEDIATE (all verified safe):
  [1] Delete 3 already-skipped tests
  [2] Delete tests/generated/ directory (42 files) -- verified: coverage stays at 90.37%

NEXT (small effort, high value):
  [3] Write 1-3 behavioral tests covering ~20 lines in plotting.py or explanations.py
  [4] Delete test_force_mark_lines_for_coverage -- verified: only 0.12% gap to close
  [5] Fix evaluate_cov_fill_adr030.py assertion quality classification

CONSOLIDATION (medium effort):
  [6] Review 50 inf-score tests from estimator
  [7] Consolidate 5 known overlapping test pairs
  [8] Use hotspot_contexts.json to find redundant tests for sequential.py, feature_task.py

PROCESS (ongoing):
  [9] Fix generate_test_templates.py template quality
  [10] Create verify_removal_batch.py
  [11] Enable over-testing density gate in advisory mode (ADR-030 Phase 4)
  [12] Add assertion quality check to CI
```

---

## Source Reports

- [pruner_proposal.md](pruner_proposal.md) -- Test removal analysis
- [deadcode_hunter_proposal.md](deadcode_hunter_proposal.md) -- Dead code analysis (result: no dead code found)
- [process_architect_proposal.md](process_architect_proposal.md) -- Process redesign
- [devils_advocate_review.md](devils_advocate_review.md) -- Risk assessment

## Data Sources (verified 2026-02-12)

- `reports/over_testing/per_test_summary.csv` -- 2,967 contexts, reliable unique_lines
- `reports/over_testing/triage.md` -- Over-testing hotspots with context data
- `reports/over_testing/hotspot_contexts.json` -- Per-hotspot test lists
- `reports/over_testing/summary.json` -- Coverage summary
- `reports/over_testing/line_coverage_counts.csv` -- Per-line hit counts
