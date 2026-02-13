# PRUNER PROPOSAL (Updated 2026-02-13)

## Data Quality Status

Per-test context quality is now valid:
- `reports/over_testing/metadata.json` -> `contexts_detected: 1785`, no warnings
- Fresh full run: `1543 passed, 1 skipped`
- Coverage gate passes at 90.00%

Important caveat still present:
- `reports/over_testing/per_test_summary.csv` has `runtime=0` for all rows, so estimator `value_score` inflates to `inf` for many tests.

## Current Low-Value Inventory

From `reports/over_testing/per_test_summary.csv`:
- Test contexts analyzed: 1784
- `unique_lines == 0`: 1144
- `1 <= unique_lines <= 4`: 416
- Files with **all** test contexts at `unique_lines == 0`: 1

File with all-zero unique coverage:
- `tests/unit/test_exec_core_reject_module.py`

## Candidate Batches for Next Removal Iteration

### Batch A (immediate, very low risk)

1. Remove `tests/unit/test_exec_core_reject_module.py`
- Entire file is all-zero unique and already functionally superseded by stronger reject-shim tests.

### Batch B (extremely low additional value, but not all-zero)

These files show very high zero-rate and near-zero unique contribution (keep coverage checks between chunks):

| File | Tests | Unique Sum | Zero % | Mean Unique |
| --- | ---: | ---: | ---: | ---: |
| `tests/unit/viz/test_plotspec_mvp.py` | 21 | 1 | 95.2% | 0.05 |
| `tests/unit/utils/test_deprecations_helper.py` | 51 | 3 | 96.1% | 0.06 |
| `tests/plugins/test_cli_additional.py` | 16 | 1 | 93.8% | 0.06 |
| `tests/unit/viz/test_viz_builders.py` | 14 | 1 | 92.9% | 0.07 |
| `tests/unit/core/test_logging_context.py` | 12 | 1 | 91.7% | 0.08 |
| `tests/unit/explanations/test_calibrated_explanations.py` | 34 | 3 | 94.1% | 0.09 |

Recommendation:
- Remove these in chunks of 2-3 files, run full `pytest --tb=no -q` after each chunk.
- If coverage dips below 90%, pause removals and add targeted behavioral tests (see test-creator proposal).

## Hotspot Context for Consolidation

`reports/over_testing/triage.md` still shows dense overlap:
- `core/explain/sequential.py` over-ratio 0.8271
- `core/explain/_computation.py` over-ratio 0.701
- `explanations/_conjunctions.py` over-ratio 0.6209

The next value gain should come from consolidation around these hotspots, not broad random pruning.

## Tooling Issue Found During Refresh

`scripts/over_testing/select_zero_unique_files.py` currently writes lines as `path,count`.
- `estimator.py --remove-list` expects node IDs / test names, so this causes warnings like:
  - `warning: test tests/unit/test_exec_core_reject_module.py,1 not found in per-test CSV`

Fix needed before automated remove-list simulation can be trusted.

## Recommended Safe Actions (Now)

1. Remove `tests/unit/test_exec_core_reject_module.py`.
2. Start Batch B with `test_plotspec_mvp.py` + `test_cli_additional.py`.
3. Re-run full suite and coverage after each mini-batch.
4. Backfill with targeted high-yield tests when coverage approaches 90.0.
