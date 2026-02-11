# Over-Testing Triage Report

## Metadata

- **min_count**: 20
- **summary**: reports\over_testing\summary.json
- **lines**: reports\over_testing\line_coverage_counts.csv
- **blocks**: reports\over_testing\block_coverage_counts.csv
- **coverage_file**: .coverage
- **contexts_detected**: 1
- **context_regex**: None
- **source_root**: src\calibrated_explanations
- **over_testing_threshold**: 20
- **warnings**: ['Only one coverage context detected. Over-testing analysis is unreliable; run pytest with --cov-context=test.']

## Top over-tested files

| File | Over ratio | Max count | Over-threshold lines | Lines covered |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\__init__.py | 0.0 | 1 | 0 | 30 |
| src\calibrated_explanations\api\__init__.py | 0.0 | 1 | 0 | 6 |
| src\calibrated_explanations\api\config.py | 0.0 | 1 | 0 | 70 |
| src\calibrated_explanations\api\params.py | 0.0 | 1 | 0 | 15 |
| src\calibrated_explanations\api\quick.py | 0.0 | 1 | 0 | 17 |
| src\calibrated_explanations\cache\__init__.py | 0.0 | 1 | 0 | 24 |
| src\calibrated_explanations\cache\cache.py | 0.0 | 1 | 0 | 98 |
| src\calibrated_explanations\cache\explanation_cache.py | 0.0 | 1 | 0 | 34 |
| src\calibrated_explanations\calibration\__init__.py | 0.0 | 1 | 0 | 24 |
| src\calibrated_explanations\calibration\interval_learner.py | 0.0 | 1 | 0 | 15 |
| src\calibrated_explanations\calibration\interval_regressor.py | 0.0 | 1 | 0 | 56 |
| src\calibrated_explanations\calibration\interval_wrappers.py | 0.0 | 1 | 0 | 25 |
| src\calibrated_explanations\calibration\state.py | 0.0 | 1 | 0 | 19 |
| src\calibrated_explanations\calibration\summaries.py | 0.0 | 1 | 0 | 11 |
| src\calibrated_explanations\calibration\venn_abers.py | 0.0 | 1 | 0 | 23 |

## Top hotspot lines

| File | Line | Test count |
| --- | --- | --- |

## Top hotspot blocks

| File | Start | End | Test count | Length |
| --- | --- | --- | --- | --- |

## Suggested process

1. Run pytest with `--cov-context=test` to record per-test contexts.
2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.
3. Run this script and inspect the top hotspots above.
4. For each hotspot, review tests for duplicate assertions or setup-only coverage.
5. Consolidate redundant tests and re-run the reports to confirm lower counts.
