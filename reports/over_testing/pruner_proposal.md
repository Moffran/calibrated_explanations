# Pruner Proposal (2026-02-17)

## Evidence
- `python scripts/over_testing/extract_per_test.py`
- `python scripts/over_testing/detect_redundant_tests.py`
- `python scripts/over_testing/estimator.py --per-test reports/over_testing/per_test_summary.csv --baseline reports/over_testing/baseline_summary.json --recommend --budget 50`

## Findings
- Per-test contexts: 1119 tests in `reports/over_testing/per_test_summary.csv`.
- Zero-unique tests: 357.
- Low-unique tests (`<5`): 828.
- Redundancy report: 26 exact duplicate groups, 46 subset groups (`reports/over_testing/redundant_tests.csv`).
- `tests/generated/` directory is absent in current workspace.

## Recommendation
- Do not remove large batches blindly in this pass.
- Prioritize compensating high-signal tests first in low-coverage hotspots, then prune top estimator candidates in guarded batches.
