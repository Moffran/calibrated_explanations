# Over-Testing Triage Report

## Metadata

- **min_count**: 20
- **summary**: reports\over_testing\summary.json
- **lines**: reports\over_testing\line_coverage_counts.csv
- **blocks**: reports\over_testing\block_coverage_counts.csv
- **coverage_file**: .coverage
- **contexts_detected**: 1089
- **context_regex**: None
- **source_root**: src\calibrated_explanations
- **over_testing_threshold**: 20
- **warnings**: []

## Top over-tested files

| File | Over ratio | Max count | Over-threshold lines | Lines covered |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\core\explain\sequential.py | 0.8271 | 42 | 110 | 133 |
| src\calibrated_explanations\core\explain\_computation.py | 0.6564 | 46 | 191 | 291 |
| src\calibrated_explanations\core\explain\feature_task.py | 0.5501 | 45 | 236 | 429 |
| src\calibrated_explanations\calibration\summaries.py | 0.5323 | 93 | 33 | 62 |
| src\calibrated_explanations\core\explain\orchestrator.py | 0.5207 | 111 | 352 | 676 |
| src\calibrated_explanations\core\explain\_shared.py | 0.4975 | 44 | 99 | 199 |
| src\calibrated_explanations\core\prediction\validation.py | 0.4828 | 90 | 14 | 29 |
| src\calibrated_explanations\plugins\base.py | 0.4727 | 38 | 26 | 55 |
| src\calibrated_explanations\core\prediction\orchestrator.py | 0.4625 | 115 | 222 | 480 |
| src\calibrated_explanations\explanations\_conjunctions.py | 0.4575 | 53 | 70 | 153 |
| src\calibrated_explanations\plugins\manager.py | 0.4469 | 127 | 202 | 452 |
| src\calibrated_explanations\core\explain\parallel_runtime.py | 0.3591 | 42 | 65 | 181 |
| src\calibrated_explanations\core\explain\_helpers.py | 0.3556 | 44 | 48 | 135 |
| src\calibrated_explanations\utils\discretizers.py | 0.3376 | 50 | 53 | 157 |
| src\calibrated_explanations\calibration\state.py | 0.3333 | 102 | 16 | 48 |

## Top hotspot lines

| File | Line | Test count |
| --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 33 | 193 |
| src\calibrated_explanations\utils\exceptions.py | 34 | 193 |
| src\calibrated_explanations\utils\helper.py | 72 | 137 |
| src\calibrated_explanations\utils\helper.py | 82 | 137 |
| src\calibrated_explanations\utils\helper.py | 83 | 137 |
| src\calibrated_explanations\utils\helper.py | 73 | 136 |
| src\calibrated_explanations\utils\helper.py | 95 | 136 |
| src\calibrated_explanations\utils\helper.py | 100 | 136 |
| src\calibrated_explanations\utils\helper.py | 103 | 136 |
| src\calibrated_explanations\utils\helper.py | 106 | 136 |
| src\calibrated_explanations\utils\helper.py | 108 | 136 |
| src\calibrated_explanations\utils\helper.py | 111 | 136 |
| src\calibrated_explanations\utils\helper.py | 114 | 135 |
| src\calibrated_explanations\plugins\registry.py | 108 | 129 |
| src\calibrated_explanations\plugins\registry.py | 109 | 129 |
| src\calibrated_explanations\plugins\registry.py | 110 | 129 |
| src\calibrated_explanations\plugins\registry.py | 111 | 129 |
| src\calibrated_explanations\plugins\registry.py | 112 | 129 |
| src\calibrated_explanations\plugins\registry.py | 114 | 129 |
| src\calibrated_explanations\plugins\registry.py | 119 | 129 |
| src\calibrated_explanations\plugins\registry.py | 120 | 129 |
| src\calibrated_explanations\plugins\manager.py | 88 | 127 |
| src\calibrated_explanations\plugins\manager.py | 89 | 127 |
| src\calibrated_explanations\plugins\manager.py | 92 | 127 |
| src\calibrated_explanations\plugins\manager.py | 93 | 127 |
| src\calibrated_explanations\plugins\manager.py | 96 | 127 |
| src\calibrated_explanations\plugins\manager.py | 97 | 127 |
| src\calibrated_explanations\plugins\manager.py | 98 | 127 |
| src\calibrated_explanations\plugins\manager.py | 99 | 127 |
| src\calibrated_explanations\plugins\manager.py | 102 | 127 |

## Top hotspot blocks

| File | Start | End | Test count | Length |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 33 | 34 | 193 | 2 |
| src\calibrated_explanations\utils\helper.py | 72 | 72 | 137 | 1 |
| src\calibrated_explanations\utils\helper.py | 82 | 83 | 137 | 2 |
| src\calibrated_explanations\utils\helper.py | 73 | 73 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 95 | 95 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 100 | 100 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 103 | 103 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 106 | 106 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 108 | 108 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 111 | 111 | 136 | 1 |
| src\calibrated_explanations\utils\helper.py | 114 | 114 | 135 | 1 |
| src\calibrated_explanations\plugins\registry.py | 108 | 112 | 129 | 5 |
| src\calibrated_explanations\plugins\registry.py | 114 | 114 | 129 | 1 |
| src\calibrated_explanations\plugins\registry.py | 119 | 120 | 129 | 2 |
| src\calibrated_explanations\plugins\manager.py | 88 | 89 | 127 | 2 |
| src\calibrated_explanations\plugins\manager.py | 92 | 93 | 127 | 2 |
| src\calibrated_explanations\plugins\manager.py | 96 | 99 | 127 | 4 |
| src\calibrated_explanations\plugins\manager.py | 102 | 104 | 127 | 3 |
| src\calibrated_explanations\plugins\manager.py | 107 | 110 | 127 | 4 |
| src\calibrated_explanations\plugins\manager.py | 113 | 115 | 127 | 3 |
| src\calibrated_explanations\plugins\manager.py | 117 | 119 | 127 | 3 |
| src\calibrated_explanations\plugins\manager.py | 121 | 123 | 127 | 3 |
| src\calibrated_explanations\plugins\manager.py | 125 | 128 | 127 | 4 |
| src\calibrated_explanations\plugins\manager.py | 130 | 132 | 127 | 3 |
| src\calibrated_explanations\plugins\manager.py | 136 | 136 | 127 | 1 |
| src\calibrated_explanations\plugins\manager.py | 139 | 144 | 127 | 6 |
| src\calibrated_explanations\plugins\manager.py | 147 | 149 | 127 | 3 |
| src\calibrated_explanations\logging.py | 72 | 77 | 126 | 6 |
| src\calibrated_explanations\logging.py | 79 | 80 | 126 | 2 |
| src\calibrated_explanations\core\config_helpers.py | 220 | 220 | 123 | 1 |

## Suggested process

1. Run pytest with `--cov-context=test` to record per-test contexts.
2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.
3. Run this script and inspect the top hotspots above.
4. For each hotspot, review tests for duplicate assertions or setup-only coverage.
5. Consolidate redundant tests and re-run the reports to confirm lower counts.
