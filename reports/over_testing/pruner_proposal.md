# PRUNER PROPOSAL: Test Removal Recommendations

## Critical Data Quality Caveat

**All per-test density data is UNRELIABLE.** The `per_test_summary.csv` was generated with only 1 coverage context (not `--cov-context=test`). This means:
- `unique_lines` values do NOT reflect true per-test uniqueness -- they reflect single-context attribution
- `runtime` is 0 for all 2,940 tests
- The estimator's `inf` scores are artifacts of 0/0 division, not real zero-contribution tests
- **No overlap detection is possible** without re-running with `--cov-context=test`

Until reliable multi-context data exists, removal decisions must be based on **content analysis**, not coverage metrics.

---

## 1. Generated Test Classification (42 files)

**Verdict: ALL 42 are REMOVE -- import-only placeholders with zero behavioral value.**

Every file follows the identical template:
```python
mod = importlib.import_module("calibrated_explanations.some.module")
assert isinstance(mod, types.ModuleType)
```

This tests only that a module is importable, not any behavior. The sole `extra_005` variant tests 3 modules but with the same `hasattr(mod, "__name__")` non-assertion.

**Module targets covered by the 42 files** (grouped by frequency):
| Module | Count | Already tested elsewhere? |
|--------|-------|--------------------------|
| `explanations.explanation` | 4 | Yes -- extensive unit tests |
| `plugins.registry/manager/builtins` | 6 | Yes -- plugin test suite |
| `plotting` | 2 | Yes -- plotting test suite |
| `core.calibrated_explainer` | 2 | Yes -- core test suite |
| `core.explain.orchestrator` | 2 | Yes -- orchestrator tests |
| All others | 24 | 1 each, all have existing tests |

**Coverage impact of removing all 42**: NEGLIGIBLE. Import-only tests contribute at most module-level statement coverage (typically 5-20 lines per module for import-time definitions). With 14,926 covered lines and a 90% floor at 14,618, removing ~200-400 import-attributed lines still stays above 90%.

---

## 2. Hand-Written Test Overlaps

### 2a. Coverage-padding test (HIGHEST PRIORITY for removal)
- **`test_force_mark_lines_for_coverage`** in [test_calibrated_explainer_additional.py:920](tests/unit/core/test_calibrated_explainer_additional.py#L920)
- Claims **991 unique_lines** -- the largest single test in the entire suite
- Compiles `pass` statements attributed to source filenames to artificially inflate coverage
- **This is exactly the anti-pattern ADR-030 aims to eliminate** -- coverage without behavioral value
- **Recommendation: REMOVE immediately** -- this is coverage fraud, not a test

### 2b. Already-skipped overtesting prune batch1 (3 tests)
These were already identified and skipped:
- [test_calibrated_explainer_additional_behavioral.py:177](tests/core/test_calibrated_explainer_additional_behavioral.py#L177) -- "duplicative metadata accessor test"
- [test_cache.py:659](tests/unit/perf/test_cache.py#L659) -- "flaky perf import-mock test"
- [test_narrative_generator.py:17](tests/unit/core/test_narrative_generator.py#L17) -- "duplicate serialization negative-path test"

**Recommendation**: Convert skips to actual deletions.

### 2c. Focused/auto_approved duplicating unit tests
| Focused/Auto Test | Overlapping Unit Test | Overlap Type |
|-------------------|----------------------|--------------|
| `focused/test_explanation_helpers_focused.py::test_to_python_number_variants` | `unit/explanations/test_explanation_unit.py::test_to_python_number` + `unit/test_explanation_helpers_extra.py::test_to_python_number_and_normalize` | **Triple coverage** of same function |
| `focused/test_plotting_ce_utils_focused.py::test_derive_threshold_labels_*` | `unit/plotting/test_label_helpers.py::testderive_threshold_labels` + `unit/plugins/test_builtins_smoke.py::test_derive_threshold_labels` + `unit/plugins/test_builtins_coverage.py::test_derive_threshold_labels` | **Quadruple coverage** |
| `focused/test_plotting_ce_utils_focused.py::test_format_save_path_variants` | `unit/plotting/test_label_helpers.py::test_format_save_path` | **Double coverage** |
| `auto_approved/test_auto_approved_003.py::test_cache_hash_and_size_estimator` | `unit/cache/test_cache_fallback.py::test_hash_part_*`, `test_make_key_*`, `test_default_size_estimator_*` | **Double coverage** |
| `auto_approved/test_auto_approved_004.py::test_config_helpers_split_and_coerce` | `unit/core/test_config_helpers.py::testsplit_csv[*]` | **Double coverage** |

**Recommendation**: Keep the **focused/auto_approved** versions (they are cleaner, ADR-030 compliant) and remove the older overlapping unit tests where the focused versions cover the same cases.

### 2d. Estimator recommendations (50 tests with `inf` score)
The estimator flagged 50 tests with `inf` value_score (0 unique_lines / 0 runtime). **These are unreliable due to 1-context data.** However, many are setup/teardown contexts (`|setup`, `|teardown`) which genuinely contribute zero behavioral coverage.

---

## 3. Recommended Removal Batches

### Batch 1: Safe Immediate (no prerequisite)
| Action | Items | Est. Coverage Impact |
|--------|-------|---------------------|
| Delete 42 generated placeholder tests | `tests/generated/test_cov_fill_*.py` | -200 to -400 lines (stays above 90%) |
| Delete `test_force_mark_lines_for_coverage` | 1 test | -991 artificial lines (real coverage unaffected) |
| Delete 3 already-skipped tests | 3 tests | Zero (already skipped) |
| **Total** | **46 tests** | **Stays above 90%** |

### Batch 2: Requires review (medium confidence)
| Action | Items | Notes |
|--------|-------|-------|
| Consolidate to_python_number tests | Remove 2 of 3 overlapping tests | Keep focused version |
| Consolidate derive_threshold_labels | Remove 3 of 4 overlapping tests | Keep focused version |
| Consolidate cache hash/size tests | Remove older unit test | Keep auto_approved version |

### Batch 3: Requires --cov-context=test data
- The 50 `inf`-score tests from estimator need multi-context validation
- True overlap detection across the full 2,940 test suite
- Identification of tests that contribute zero unique lines when properly measured

---

## 4. What CANNOT Be Determined Without `--cov-context=test`

1. True per-test unique line counts (current data is single-context)
2. Real test overlap patterns (which tests cover identical code paths)
3. Whether the 2,471 "zero unique lines" tests are genuinely redundant or just data artifacts
4. Accurate coverage impact estimates for batch removals
5. Over-testing hotspots (which lines are hit by 20+ tests)
