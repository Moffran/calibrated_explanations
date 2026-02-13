# Over-Testing Triage Report

## Metadata

- **min_count**: 20
- **summary**: reports\over_testing\summary.json
- **lines**: reports\over_testing\line_coverage_counts.csv
- **blocks**: reports\over_testing\block_coverage_counts.csv
- **coverage_file**: .coverage
- **contexts_detected**: 2967
- **context_regex**: None
- **source_root**: src\calibrated_explanations
- **over_testing_threshold**: 20
- **warnings**: []

## Top over-tested files

| File | Over ratio | Max count | Over-threshold lines | Lines covered |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\core\explain\sequential.py | 0.8271 | 172 | 110 | 133 |
| src\calibrated_explanations\core\explain\feature_task.py | 0.7669 | 190 | 329 | 429 |
| src\calibrated_explanations\core\explain\_computation.py | 0.7629 | 193 | 222 | 291 |
| src\calibrated_explanations\core\explain\_legacy_explain.py | 0.7406 | 44 | 197 | 266 |
| src\calibrated_explanations\utils\discretizers.py | 0.6879 | 202 | 108 | 157 |
| src\calibrated_explanations\explanations\_conjunctions.py | 0.6783 | 196 | 97 | 143 |
| src\calibrated_explanations\calibration\summaries.py | 0.6452 | 472 | 40 | 62 |
| src\calibrated_explanations\core\explain\orchestrator.py | 0.5452 | 473 | 380 | 697 |
| src\calibrated_explanations\core\prediction\orchestrator.py | 0.5316 | 486 | 261 | 491 |
| src\calibrated_explanations\utils\deprecation.py | 0.5238 | 21 | 11 | 21 |
| src\calibrated_explanations\plugins\base.py | 0.5088 | 212 | 29 | 57 |
| src\calibrated_explanations\core\explain\_shared.py | 0.4975 | 174 | 99 | 199 |
| src\calibrated_explanations\explanations\explanation.py | 0.4971 | 279 | 1020 | 2052 |
| src\calibrated_explanations\plugins\manager.py | 0.4902 | 474 | 225 | 459 |
| src\calibrated_explanations\core\prediction\validation.py | 0.4828 | 420 | 14 | 29 |

## Top hotspot lines

| File | Line | Test count |
| --- | --- | --- |
| src\calibrated_explanations\utils\helper.py | 72 | 599 |
| src\calibrated_explanations\utils\helper.py | 82 | 599 |
| src\calibrated_explanations\utils\helper.py | 83 | 599 |
| src\calibrated_explanations\plugins\registry.py | 108 | 597 |
| src\calibrated_explanations\plugins\registry.py | 109 | 597 |
| src\calibrated_explanations\plugins\registry.py | 110 | 597 |
| src\calibrated_explanations\plugins\registry.py | 111 | 597 |
| src\calibrated_explanations\plugins\registry.py | 112 | 597 |
| src\calibrated_explanations\plugins\registry.py | 114 | 597 |
| src\calibrated_explanations\plugins\registry.py | 119 | 597 |
| src\calibrated_explanations\plugins\registry.py | 120 | 597 |
| src\calibrated_explanations\utils\helper.py | 95 | 597 |
| src\calibrated_explanations\utils\helper.py | 100 | 597 |
| src\calibrated_explanations\utils\helper.py | 73 | 596 |
| src\calibrated_explanations\utils\helper.py | 103 | 595 |
| src\calibrated_explanations\utils\helper.py | 106 | 595 |
| src\calibrated_explanations\utils\helper.py | 108 | 595 |
| src\calibrated_explanations\utils\helper.py | 111 | 594 |
| src\calibrated_explanations\utils\helper.py | 114 | 586 |
| src\calibrated_explanations\logging.py | 72 | 576 |
| src\calibrated_explanations\logging.py | 73 | 576 |
| src\calibrated_explanations\logging.py | 74 | 576 |
| src\calibrated_explanations\logging.py | 76 | 576 |
| src\calibrated_explanations\logging.py | 77 | 576 |
| src\calibrated_explanations\logging.py | 79 | 576 |
| src\calibrated_explanations\logging.py | 75 | 575 |
| src\calibrated_explanations\logging.py | 80 | 575 |
| src\calibrated_explanations\plugins\registry.py | 1028 | 558 |
| src\calibrated_explanations\plugins\registry.py | 1033 | 558 |
| src\calibrated_explanations\plugins\registry.py | 1034 | 558 |

## Top hotspot blocks

| File | Start | End | Test count | Length |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\utils\helper.py | 72 | 72 | 599 | 1 |
| src\calibrated_explanations\utils\helper.py | 82 | 83 | 599 | 2 |
| src\calibrated_explanations\plugins\registry.py | 108 | 112 | 597 | 5 |
| src\calibrated_explanations\plugins\registry.py | 114 | 114 | 597 | 1 |
| src\calibrated_explanations\plugins\registry.py | 119 | 120 | 597 | 2 |
| src\calibrated_explanations\utils\helper.py | 95 | 95 | 597 | 1 |
| src\calibrated_explanations\utils\helper.py | 100 | 100 | 597 | 1 |
| src\calibrated_explanations\utils\helper.py | 73 | 73 | 596 | 1 |
| src\calibrated_explanations\utils\helper.py | 103 | 103 | 595 | 1 |
| src\calibrated_explanations\utils\helper.py | 106 | 106 | 595 | 1 |
| src\calibrated_explanations\utils\helper.py | 108 | 108 | 595 | 1 |
| src\calibrated_explanations\utils\helper.py | 111 | 111 | 594 | 1 |
| src\calibrated_explanations\utils\helper.py | 114 | 114 | 586 | 1 |
| src\calibrated_explanations\logging.py | 72 | 74 | 576 | 3 |
| src\calibrated_explanations\logging.py | 76 | 77 | 576 | 2 |
| src\calibrated_explanations\logging.py | 79 | 79 | 576 | 1 |
| src\calibrated_explanations\logging.py | 75 | 75 | 575 | 1 |
| src\calibrated_explanations\logging.py | 80 | 80 | 575 | 1 |
| src\calibrated_explanations\plugins\registry.py | 1028 | 1028 | 558 | 1 |
| src\calibrated_explanations\plugins\registry.py | 1033 | 1036 | 558 | 4 |
| src\calibrated_explanations\plugins\registry.py | 1038 | 1039 | 558 | 2 |
| src\calibrated_explanations\plugins\registry.py | 1042 | 1042 | 558 | 1 |
| src\calibrated_explanations\plugins\registry.py | 1045 | 1045 | 558 | 1 |
| src\calibrated_explanations\plugins\registry.py | 1048 | 1048 | 558 | 1 |
| src\calibrated_explanations\plugins\registry.py | 1050 | 1050 | 558 | 1 |
| src\calibrated_explanations\core\reject\policy.py | 22 | 22 | 534 | 1 |
| src\calibrated_explanations\core\reject\policy.py | 31 | 31 | 534 | 1 |
| src\calibrated_explanations\core\prediction\interval_summary.py | 26 | 27 | 525 | 2 |
| src\calibrated_explanations\core\calibrated_explainer.py | 392 | 392 | 517 | 1 |
| src\calibrated_explanations\core\calibrated_explainer.py | 394 | 395 | 517 | 2 |

## Suggested process

1. Run pytest with `--cov-context=test` to record per-test contexts.
2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.
3. Run this script and inspect the top hotspots above.
4. For each hotspot, review tests for duplicate assertions or setup-only coverage.
5. Consolidate redundant tests and re-run the reports to confirm lower counts.
