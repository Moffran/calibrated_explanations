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
