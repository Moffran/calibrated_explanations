# Over-Testing Triage Report

## Metadata

- **min_count**: 20
- **summary**: reports\over_testing\summary.json
- **lines**: reports\over_testing\line_coverage_counts.csv
- **blocks**: reports\over_testing\block_coverage_counts.csv
- **coverage_file**: .coverage.over_testing
- **contexts_detected**: 1037
- **context_regex**: None
- **source_root**: src\calibrated_explanations
- **over_testing_threshold**: 20
- **warnings**: []

## Top over-tested files

| File | Over ratio | Max count | Over-threshold lines | Lines covered |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\core\explain\sequential.py | 0.8271 | 40 | 110 | 133 |
| src\calibrated_explanations\core\explain\_computation.py | 0.6564 | 44 | 191 | 291 |
| src\calibrated_explanations\core\explain\feature_task.py | 0.5501 | 43 | 236 | 429 |
| src\calibrated_explanations\calibration\summaries.py | 0.5323 | 89 | 33 | 62 |
| src\calibrated_explanations\core\explain\orchestrator.py | 0.5207 | 107 | 352 | 676 |
| src\calibrated_explanations\core\explain\_shared.py | 0.4975 | 42 | 99 | 199 |
| src\calibrated_explanations\core\prediction\validation.py | 0.4828 | 85 | 14 | 29 |
| src\calibrated_explanations\plugins\base.py | 0.4727 | 37 | 26 | 55 |
| src\calibrated_explanations\core\prediction\orchestrator.py | 0.4625 | 108 | 222 | 480 |
| src\calibrated_explanations\plugins\manager.py | 0.4469 | 118 | 202 | 452 |
| src\calibrated_explanations\core\explain\parallel_runtime.py | 0.3591 | 40 | 65 | 181 |
| src\calibrated_explanations\core\explain\_helpers.py | 0.3556 | 42 | 48 | 135 |
| src\calibrated_explanations\explanations\_conjunctions.py | 0.3399 | 50 | 52 | 153 |
| src\calibrated_explanations\utils\discretizers.py | 0.3376 | 48 | 53 | 157 |
| src\calibrated_explanations\calibration\state.py | 0.3333 | 98 | 16 | 48 |

## Top hotspot lines

| File | Line | Test count |
| --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 33 | 191 |
| src\calibrated_explanations\utils\exceptions.py | 34 | 191 |
| src\calibrated_explanations\utils\helper.py | 72 | 134 |
| src\calibrated_explanations\utils\helper.py | 82 | 134 |
| src\calibrated_explanations\utils\helper.py | 83 | 134 |
| src\calibrated_explanations\utils\helper.py | 73 | 133 |
| src\calibrated_explanations\utils\helper.py | 95 | 133 |
| src\calibrated_explanations\utils\helper.py | 100 | 133 |
| src\calibrated_explanations\utils\helper.py | 103 | 133 |
| src\calibrated_explanations\utils\helper.py | 106 | 133 |
| src\calibrated_explanations\utils\helper.py | 108 | 133 |
| src\calibrated_explanations\utils\helper.py | 111 | 133 |
| src\calibrated_explanations\utils\helper.py | 114 | 132 |
| src\calibrated_explanations\plugins\registry.py | 108 | 122 |
| src\calibrated_explanations\plugins\registry.py | 109 | 122 |
| src\calibrated_explanations\plugins\registry.py | 110 | 122 |
| src\calibrated_explanations\plugins\registry.py | 111 | 122 |
| src\calibrated_explanations\plugins\registry.py | 112 | 122 |
| src\calibrated_explanations\plugins\registry.py | 114 | 122 |
| src\calibrated_explanations\plugins\registry.py | 119 | 122 |
| src\calibrated_explanations\plugins\registry.py | 120 | 122 |
| src\calibrated_explanations\logging.py | 72 | 120 |
| src\calibrated_explanations\logging.py | 73 | 120 |
| src\calibrated_explanations\logging.py | 74 | 120 |
| src\calibrated_explanations\logging.py | 76 | 120 |
| src\calibrated_explanations\logging.py | 77 | 120 |
| src\calibrated_explanations\logging.py | 79 | 120 |
| src\calibrated_explanations\logging.py | 75 | 119 |
| src\calibrated_explanations\logging.py | 80 | 119 |
| src\calibrated_explanations\plugins\manager.py | 88 | 118 |

## Top hotspot blocks

| File | Start | End | Test count | Length |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 33 | 34 | 191 | 2 |
| src\calibrated_explanations\utils\helper.py | 72 | 72 | 134 | 1 |
| src\calibrated_explanations\utils\helper.py | 82 | 83 | 134 | 2 |
| src\calibrated_explanations\utils\helper.py | 73 | 73 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 95 | 95 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 100 | 100 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 103 | 103 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 106 | 106 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 108 | 108 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 111 | 111 | 133 | 1 |
| src\calibrated_explanations\utils\helper.py | 114 | 114 | 132 | 1 |
| src\calibrated_explanations\plugins\registry.py | 108 | 112 | 122 | 5 |
| src\calibrated_explanations\plugins\registry.py | 114 | 114 | 122 | 1 |
| src\calibrated_explanations\plugins\registry.py | 119 | 120 | 122 | 2 |
| src\calibrated_explanations\logging.py | 72 | 74 | 120 | 3 |
| src\calibrated_explanations\logging.py | 76 | 77 | 120 | 2 |
| src\calibrated_explanations\logging.py | 79 | 79 | 120 | 1 |
| src\calibrated_explanations\logging.py | 75 | 75 | 119 | 1 |
| src\calibrated_explanations\logging.py | 80 | 80 | 119 | 1 |
| src\calibrated_explanations\plugins\manager.py | 88 | 89 | 118 | 2 |
| src\calibrated_explanations\plugins\manager.py | 92 | 93 | 118 | 2 |
| src\calibrated_explanations\plugins\manager.py | 96 | 99 | 118 | 4 |
| src\calibrated_explanations\plugins\manager.py | 102 | 104 | 118 | 3 |
| src\calibrated_explanations\plugins\manager.py | 107 | 110 | 118 | 4 |
| src\calibrated_explanations\plugins\manager.py | 113 | 115 | 118 | 3 |
| src\calibrated_explanations\plugins\manager.py | 117 | 119 | 118 | 3 |
| src\calibrated_explanations\plugins\manager.py | 121 | 123 | 118 | 3 |
| src\calibrated_explanations\plugins\manager.py | 125 | 128 | 118 | 4 |
| src\calibrated_explanations\plugins\manager.py | 130 | 132 | 118 | 3 |
| src\calibrated_explanations\plugins\manager.py | 136 | 136 | 118 | 1 |

## Suggested process

1. Run pytest with `--cov-context=test` to record per-test contexts.
2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.
3. Run this script and inspect the top hotspots above.
4. For each hotspot, review tests for duplicate assertions or setup-only coverage.
5. Consolidate redundant tests and re-run the reports to confirm lower counts.
