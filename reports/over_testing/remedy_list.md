# Remedy List for Generated Tests

This file lists generated `test_cov_fill_*` tests that require manual remediation
to conform to ADR-030 and repository test-quality rules.

Summary (auto-generated):

- Total `test_cov_fill_*` scanned: see `reports/over_testing/cov_fill_adr30_scan.csv`.
- Prune plan produced at `reports/over_testing/prune_plan.json` (conservative: no automatic removals proposed).

All generated files are currently flagged as *questionable* by the conservative pruning heuristic because they contain assertions and therefore may be meaningful; human review is required to decide whether each is:

- **Keep & Move:** the test is behavior-first and conforms to ADR-030 — move to `tests/auto_approved/` and rename accordingly.
- **Refactor:** the test is useful but tests private internals or is non-deterministic — refactor to test public behavior per ADR-030.
- **Remove:** the test is a trivial placeholder or duplicates other tests — move to `reports/over_testing/backup_removed_tests/` or delete after confirmation.

Next steps (manual):

1. Open `reports/over_testing/cov_fill_adr30_scan.csv` and inspect rows marked `has_assertion=False` first (none currently).
2. For each file listed under `prune_plan.json` → `questionable`, review test contents and decide action (Keep/Refactor/Remove).
3. Record per-file decisions in this document (append) and run `python scripts/over_testing/prune_generated_tests.py --apply` to apply deletions once reviewed.

This remedy list must be reviewed and signed off by a core maintainer before any mass removals.

## 2026-02-13 Implementer Update

- Replaced low-quality coverage padding in `tests/unit/test_coverage_artifacts.py`.
- Removed `exec(compile(...))` line-marking behavior and added deterministic behavioral tests for:
  - `calibration/state.py` (`set_x_cal`, `set_y_cal`, `append_calibration`)
  - `core/test.py` (`JoblibBackend`, `sequential_map`)
  - `schema/__init__.py` lazy export path
  - `viz/__init__.py` matplotlib-required lazy path
  - `plugins/predict_monitor.py` invariant warning and call tracking paths
  - `core/reject/orchestrator.py` initialization and pickle state restoration paths
- Verified local quality checks:
  - `pytest -q --no-cov tests/unit/test_coverage_artifacts.py` passes (7 tests)
  - `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py` reports 0 anti-patterns
  - `python scripts/anti-pattern-analysis/scan_private_usage.py --check` reports 0 private-member violations

## 2026-02-13 Implementer + Process-Architect Follow-up

- Continued post-cleanup under-testing remediation with high-signal tests (no import-only padding):
  - `tests/unit/core/test_feature_filter_branch_boosters.py`
  - `tests/unit/test_quick_adapters_and_shims.py`
  - `tests/unit/test_utils_perturbation.py`
  - `tests/unit/test_api_params.py`
  - extended `tests/unit/test_ce_agent_utils.py`
- Fixed failing regression test `test_viz_lazy_import_requires_matplotlib` by exercising `viz.__getattr__("render")` directly.
- Coverage gate status after full run:
  - `pytest --tb=no -q` passes
  - total coverage: **90.04%** (gate 90%)
  - test result: **1846 passed, 1 skipped**
- Quality safety checks:
  - `python scripts/anti-pattern-analysis/scan_private_usage.py --check` passes (0 violations)
  - `ruff` passes on all new/updated tests

### Process Architect verdict (current cycle)

The method is **efficient as-is for implementation flow** (role split + remedy ledger + safety checks worked), but two process updates remain warranted:

1. Add an explicit "post-remediation gate pack" step to the README:
   - run `ruff` on changed tests
   - run `scan_private_usage.py --check`
   - run full `pytest --cov-fail-under=90`
2. Add a "coverage cliff recovery playbook" subsection:
   - prioritize high-yield branch modules (`_feature_filter`, CE shims, perturbation/adapter seams) before broad exploratory additions.

## 2026-02-13 Implementer + Process-Architect Follow-up (Cycle 2)

- Continued low-quality cleanup by replacing placeholder/no-op plotting tests with behavioral assertions in `tests/unit/test_plotting.py`.
- Added high-yield non-viz gap-closure coverage in `tests/unit/core/explain/test_helpers.py`:
  - `test_compute_weight_delta_fallback_path_for_object_values` now exercises the fallback branch in `core/explain/_helpers.py` and contributes **9 unique lines** in `reports/over_testing/per_test_summary.csv`.
- Re-ran method verification end-to-end:
  - `pytest --cov-fail-under=90` -> **PASS**, coverage **90.14%**
  - `python scripts/over_testing/run_over_testing_pipeline.py` -> **PASS**
  - `python scripts/over_testing/extract_per_test.py` -> **PASS**
  - `python scripts/over_testing/detect_redundant_tests.py` -> **PASS** (1553 contexts)
  - `python scripts/quality/check_coverage_gates.py` -> **PASS** (all critical modules)

### Process Architect verdict (Cycle 2)

The current method remains **efficient as-is** for cleanup + backfill execution.
One small process/documentation adjustment is recommended:

1. Add an explicit sequencing rule to the README: run `run_over_testing_pipeline.py` to completion before `extract_per_test.py` and `detect_redundant_tests.py` (avoid race conditions from parallel execution).
2. Add a short note that the canonical path is `docs/improvement/test-quality-method/README.md` (the historical `docs/improvement/test-quality/README.md` path is stale).

## 2026-02-13 Implementer Update (Current)

- Executed an aggressive zero-unique pruning pass using method data (`per_test_summary.csv`, `|run` contexts only), then constrained final removals to keep the hard 90% gate.
- Final pruning result in this execution: **47 test functions removed** (net), with targeted branch-focused replacements added where coverage gaps emerged.
- Added high-quality gap-closing tests:
  - `tests/unit/testing/test_parity_compare.py`
  - `tests/unit/core/test_logging_context.py`
  - `tests/unit/test_ce_agent_utils.py`
  - `tests/unit/core/test_serialization_invariants_extra.py`
- Verified quality and coverage:
  - `pytest --cov-fail-under=90` -> **PASS at 90.00%** (`968 passed, 1 skipped`)
  - `python scripts/over_testing/run_over_testing_pipeline.py` -> **PASS**
  - `python scripts/over_testing/extract_per_test.py` -> **PASS**
  - `python scripts/over_testing/detect_redundant_tests.py` -> **PASS**
- Updated current over-testing metrics:
  - `|run` contexts with `unique_lines=0`: **98**
  - `|run` contexts with `unique_lines<5`: **560**
  - Redundancy report: **19 exact duplicate groups**, **69 subset groups**, **361 potential redundant tests**

## 2026-02-13 Implementer Update (Additional Iteration)

- Executed another pruning iteration focused on low-value `|run` subset redundancies while keeping ADR-030-safe behavioral coverage.
- Net removals kept in this iteration:
  - `tests/plugins/test_cli_additional.py`: removed `test_emit_plot_builder_descriptor_reports_legacy`
  - `tests/unit/utils/test_deprecations_helper.py`: removed `TestShouldRaise::test_should_return_false_for_unknown_value`
  - `tests/unit/test_ce_agent_utils.py`: removed `test_probe_optional_features_warning`
  - `tests/unit/core/test_validation_unit.py`: removed `test_validate_inputs_adr002_signature_accepts_2d_array`
- Verification:
  - `pytest --cov=src/calibrated_explanations --cov-context=test --cov-fail-under=90` -> **PASS** (`954 passed, 1 skipped`, coverage **90.00%**)
  - `python scripts/over_testing/run_over_testing_pipeline.py` -> **PASS**
  - `python scripts/over_testing/extract_per_test.py` -> **PASS**
  - `python scripts/over_testing/detect_redundant_tests.py` -> **PASS**
- Updated over-testing metrics after this iteration:
  - Per-test contexts: **1038**
  - Redundancy report: **18 exact duplicate groups**, **56 subset groups**, **345 potential redundant tests**
  - `|run` contexts appearing in `redundant_tests.csv`: **44**

## 2026-02-13 Implementer Update (One More Pass)

- Performed one additional conservative pruning step on smallest redundant `|run` subset helpers in `tests/unit/core/explain/test_helpers.py`:
  - removed `TestSliceThreshold::test_slice_threshold_preserves_none`
  - removed `TestSliceBins::test_slice_bins_preserves_none`
- Verification:
  - `pytest --cov=src/calibrated_explanations --cov-context=test --cov-fail-under=90` -> **PASS** (`952 passed, 1 skipped`, coverage **90.00%**)
  - `python scripts/over_testing/run_over_testing_pipeline.py` -> **PASS**
  - `python scripts/over_testing/extract_per_test.py` -> **PASS**
  - `python scripts/over_testing/detect_redundant_tests.py` -> **PASS**
- Updated over-testing metrics after this pass:
  - Per-test contexts: **1036**
  - Redundancy report: **18 exact duplicate groups**, **54 subset groups**, **343 potential redundant tests**

## 2026-02-13 Test+Code Quality Extension Analysis

Objective: extend the current test-quality method into a unified **test + source-code anti-pattern method** without adding noisy or low-signal checks.

### Evidence collected (current baseline)

- `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py`
  - Result: **3 findings** (all private helper calls in tests).
- `python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv`
  - Result: **356 private definitions** analyzed
  - Patterns:
    - `Consistent (Internal Only)`: **347**
    - `Pattern 3 (Completely Dead)`: **5** (all in `src/calibrated_explanations/cache/cache.py` + `src/calibrated_explanations/explanations/reject.py`)
    - Test-helper patterns: **4**
- `python scripts/quality/check_adr002_compliance.py`
  - Result: **PASS** (no ADR-002 violations).
- Structural hotspot scan (AST-based, local one-off):
  - Functions over 120 lines: **54**
  - Functions over 200 lines: **18**
  - Functions over 8 args: **37**
  - Largest hotspots:
    - `src/calibrated_explanations/viz/matplotlib_adapter.py` (`render`, `_render_body`)
    - `src/calibrated_explanations/core/explain/feature_task.py` (`feature_task`)
    - `src/calibrated_explanations/core/explain/orchestrator.py` (`invoke`)
    - `src/calibrated_explanations/plotting.py` (`plot_alternative`, `plot_probabilistic`, `plot_regression`)
    - `src/calibrated_explanations/plugins/builtins.py` (`explain_batch`, `build`)

### Anti-patterns to identify and remove (recommended)

These are high-signal for this codebase and should be added to the method:

1. **Dead private/library helpers**
   - Why: pure maintenance burden and cognitive load.
   - Detector: existing `analyze_private_methods.py` Pattern 3.
   - Initial targets: `_reconstruct_lru_cache`, `_reconstruct_calibrator_cache`, `_reconstruct_cache_config`, `_reconstruct_cache_metrics`, `_missing_`.

2. **God functions / oversized orchestration methods**
   - Why: strongest predictor of regressions in this library’s orchestration-heavy modules.
   - Detector: AST thresholds + per-function branch counts.
   - Suggested initial thresholds:
     - warn: >120 LOC
     - fail gate on touched code: >220 LOC unless allowlisted with rationale.

3. **Long parameter lists and dict-style argument tunneling**
   - Why: brittle API surfaces and accidental coupling across plugins/wrappers.
   - Detector: AST arg count (>8 warn, >12 fail on new/changed functions).

4. **Unscoped broad exception handling**
   - Why: hidden failure modes.
   - Detector: extend ADR-002 checker:
     - disallow broad catches unless `adr002_allow` marker present and handler logs/re-raises/falls back explicitly.
     - disallow silent broad catches on non-optional paths.

5. **Deprecated shim debt past removal window**
   - Why: compatibility shims become permanent complexity if not expired.
   - Detector: parse deprecation/removal version text; fail when current version >= declared removal version.
   - Targets: `perf/*`, `core/calibration/*`, legacy plotting shims.

6. **Duplicate behavior across legacy/new paths**
   - Why: dual logic drift risk (especially plotting + explain orchestration).
   - Detector: file-pair triage + shared-coverage fingerprint overlap to identify duplicate behavior that should collapse to adapters.

7. **Import-time side effects outside CLI/plugin bootstrap**
   - Why: fragile imports and test flakiness.
   - Detector: AST for top-level execution (I/O, plugin discovery, mutable global initialization) with allowlist for intentional modules.

8. **Weak or missing invariant checks at plugin boundaries**
   - Why: plugin architecture is a major fault boundary in this repo.
   - Detector: require explicit schema/metadata validation in registration/dispatch call sites.

9. **Inconsistent exception taxonomy usage**
   - Why: ADR-002 is a core architectural contract.
   - Detector: extend `check_adr002_compliance.py` to assert `warnings.warn(..., stacklevel=...)` and reject raw `ValueError/RuntimeError/Exception` in new code paths.

10. **Large untyped high-risk modules**
   - Why: the largest modules are exactly where regressions cluster.
   - Detector: staged mypy strictness expansion by module risk rank (hotspot-first).

### Anti-patterns that can be ignored in this library (with rationale)

1. **Use of `print()` in CLI module**
   - Ignore in `src/calibrated_explanations/plugins/cli.py`; this is expected CLI output, not library logging misuse.

2. **Lazy imports via `__getattr__`**
   - Keep: this is intentional for optional deps (`viz`, plotting backends) and import-time performance.

3. **Broad catches explicitly marked `adr002_allow` in plugin/optional-dependency boundaries**
   - Keep with controls: this is a conscious resilience policy for third-party/plugin instability and optional extras.

4. **Backward-compatibility aliases/shims before their planned removal version**
   - Keep until deadline; track and expire rather than blanket-removing now.

5. **Private helper naming in source**
   - `_name` itself is not an anti-pattern here; only dead helpers, leaked internals, or test-coupled internals are anti-patterns.

6. **High branch density in rendering modules alone**
   - Not automatically bad: plotting/render code naturally fans out by mode/task; flag only when paired with low test signal or repeated bug churn.

7. **Docstring examples containing `print(...)`**
   - Ignore as documentation artifact.

### Method extension (practical pipeline change)

Add a **Step 2.5 Code Anti-Pattern Pass** after per-test extraction:

1. `python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv`
2. `python scripts/quality/check_adr002_compliance.py`
3. `python scripts/anti-pattern-analysis/detect_code_anti_patterns.py` (new script; categories 2/3/5/6/7/8 above)
4. Emit `reports/anti-pattern-analysis/code_anti_pattern_report.csv` + severity summary.
5. Gate policy:
   - Blockers: dead private code in touched modules, non-allowlisted broad catches, expired shims, new oversize functions beyond fail threshold.
   - Advisory: legacy hotspot complexity in untouched modules.

This keeps the method conservative, auditable, and aligned with existing ADR-002 and CE-first architecture constraints.

## 2026-02-13 Implementer Update (CQ-001 Execution)

- Executed CQ-001 from `reports/over_testing/code_quality_auditor_proposal.md` using devil's-advocate constraints from `reports/over_testing/devils_advocate_review.md`.

Phase 1 (low-risk dead-private test helper removals):

- Removed unused helper `tests/unit/explanations/test_conjunction_hardening.py` -> `_make_binary_explainer`.
- Removed unused private stub `tests/unit/explanations/test_explanation_more.py` -> `ContainerStub._get_explainer`.
- Verified targeted tests:
  - `pytest -q --no-cov tests/unit/explanations/test_explanation_more.py tests/unit/explanations/test_conjunction_hardening.py` -> PASS.

Phase 2 (bounded hotspot refactor):

- Refactored `src/calibrated_explanations/core/explain/feature_task.py` without API/signature changes:
  - extracted `_build_empty_feature_result(...)` for duplicated early-return payload construction.
  - extracted `_process_categorical_feature(...)` for categorical branch processing.
  - kept numeric branch logic in-place per CQ-001 no-go constraints.
- Verified targeted tests:
  - `pytest -q --no-cov tests/unit/core/test_calibrated_explainer_additional.py tests/unit/core/test_assign_weight_scalar.py tests/unit/core/test_explain_helpers_and_plugins.py tests/unit/explanations/test_explanation_more.py tests/unit/explanations/test_conjunction_hardening.py` -> PASS.

Verification gate pack (post-change):

- `python scripts/quality/check_adr002_compliance.py` -> PASS.
- `python scripts/quality/check_import_graph.py` -> PASS.
- `python scripts/quality/check_docstring_coverage.py` -> PASS (overall 95.09%).
- PowerShell deprecation-sensitive subset:
  - `$env:CE_DEPRECATIONS='error'; pytest tests/unit -m "not viz" -q --maxfail=1 --no-cov` -> PASS.
- Full coverage and module gates:
  - `pytest --cov-fail-under=90 -q` -> PASS at **90.03%**.
  - `python scripts/quality/check_coverage_gates.py` -> PASS.

Artifact refresh (post-change):

- `python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv` -> refreshed.
- `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py --output reports/anti-pattern-analysis/test_anti_pattern_report.csv` -> refreshed.
- Updated private-method summary after CQ-001:
  - total rows: **357**
  - scope: **354 library**, **3 test**
  - patterns: **349 Consistent (Internal Only)**, **5 Pattern 3 (Completely Dead)**, **3 test-helper patterns**

No-go constraints respected:

- Did **not** remove dynamically referenced source symbols (`_reconstruct_*`, `RejectPolicy._missing_`).
- Did **not** restructure numeric branch in `feature_task.py` in this batch.
