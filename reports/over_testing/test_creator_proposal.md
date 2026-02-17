# Test Creator Proposal (2026-02-17)

## Evidence
- `python scripts/over_testing/gap_analyzer.py --line-csv reports/over_testing/line_coverage_counts.csv --threshold 10`
- `python scripts/quality/check_coverage_gates.py coverage.xml`
- Coverage report from over-testing pipeline (`reports/over_testing/summary.json`, terminal output).

## Prioritized Targets
1. `src/calibrated_explanations/preprocessing/builtin_encoder.py` (81.9%): compact missed branches and exception paths, high gain per test line.
2. `src/calibrated_explanations/calibration/venn_abers.py` missing restore/checksum branches.
3. `src/calibrated_explanations/calibration/interval_regressor.py` missing restore/checksum branches.

## Implemented in this cycle
- Added `tests/unit/core/test_builtin_encoder.py` with 5 behavioral tests targeting:
  - not-fitted guard,
  - unseen policy (`ignore` and `error`),
  - mapping snapshot/set_mapping none path,
  - `_safe_val` fallback path via public `fit` API.

## Expected impact
- Close most uncovered branches in `builtin_encoder.py` without private-member access.
- New tests are deterministic and assert behavior (ADR-030 aligned).
