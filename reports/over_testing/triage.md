# Over-Testing Triage Report

## Metadata

- **min_count**: 20
- **summary**: reports\over_testing\summary.json
- **lines**: reports\over_testing\line_coverage_counts.csv
- **blocks**: reports\over_testing\block_coverage_counts.csv
- **coverage_file**: .coverage.over_testing
- **contexts_detected**: 1125
- **context_regex**: None
- **source_root**: src\calibrated_explanations
- **over_testing_threshold**: 20
- **warnings**: []

## Top over-tested files

| File | Over ratio | Max count | Over-threshold lines | Lines covered |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\core\explain\sequential.py | 0.8271 | 54 | 110 | 133 |
| src\calibrated_explanations\core\explain\_computation.py | 0.701 | 58 | 204 | 291 |
| src\calibrated_explanations\explanations\_conjunctions.py | 0.634 | 64 | 97 | 153 |
| src\calibrated_explanations\calibration\summaries.py | 0.5806 | 129 | 36 | 62 |
| src\calibrated_explanations\core\explain\orchestrator.py | 0.5436 | 148 | 368 | 677 |
| src\calibrated_explanations\core\explain\_shared.py | 0.4975 | 55 | 99 | 199 |
| src\calibrated_explanations\core\explain\feature_task.py | 0.4968 | 57 | 235 | 473 |
| src\calibrated_explanations\core\prediction\validation.py | 0.4828 | 125 | 14 | 29 |
| src\calibrated_explanations\plugins\base.py | 0.4727 | 38 | 26 | 55 |
| src\calibrated_explanations\core\prediction\orchestrator.py | 0.4683 | 148 | 229 | 489 |
| src\calibrated_explanations\plugins\manager.py | 0.4469 | 172 | 202 | 452 |
| src\calibrated_explanations\utils\discretizers.py | 0.4167 | 61 | 65 | 156 |
| src\calibrated_explanations\core\discretizer_config.py | 0.3663 | 62 | 37 | 101 |
| src\calibrated_explanations\core\explain\parallel_runtime.py | 0.3591 | 53 | 65 | 181 |
| src\calibrated_explanations\core\explain\_helpers.py | 0.3556 | 56 | 48 | 135 |

## Top hotspot lines

| File | Line | Test count |
| --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 34 | 223 |
| src\calibrated_explanations\utils\exceptions.py | 35 | 223 |
| src\calibrated_explanations\utils\helper.py | 72 | 192 |
| src\calibrated_explanations\utils\helper.py | 82 | 192 |
| src\calibrated_explanations\utils\helper.py | 83 | 192 |
| src\calibrated_explanations\utils\helper.py | 73 | 191 |
| src\calibrated_explanations\utils\helper.py | 95 | 191 |
| src\calibrated_explanations\utils\helper.py | 100 | 191 |
| src\calibrated_explanations\utils\helper.py | 103 | 191 |
| src\calibrated_explanations\utils\helper.py | 106 | 191 |
| src\calibrated_explanations\utils\helper.py | 108 | 191 |
| src\calibrated_explanations\utils\helper.py | 111 | 191 |
| src\calibrated_explanations\utils\helper.py | 114 | 191 |
| src\calibrated_explanations\utils\helper.py | 192 | 177 |
| src\calibrated_explanations\utils\helper.py | 194 | 177 |
| src\calibrated_explanations\utils\helper.py | 195 | 177 |
| src\calibrated_explanations\utils\helper.py | 196 | 177 |
| src\calibrated_explanations\utils\helper.py | 200 | 177 |
| src\calibrated_explanations\plugins\manager.py | 88 | 172 |
| src\calibrated_explanations\plugins\manager.py | 89 | 172 |
| src\calibrated_explanations\plugins\manager.py | 92 | 172 |
| src\calibrated_explanations\plugins\manager.py | 93 | 172 |
| src\calibrated_explanations\plugins\manager.py | 96 | 172 |
| src\calibrated_explanations\plugins\manager.py | 97 | 172 |
| src\calibrated_explanations\plugins\manager.py | 98 | 172 |
| src\calibrated_explanations\plugins\manager.py | 99 | 172 |
| src\calibrated_explanations\plugins\manager.py | 102 | 172 |
| src\calibrated_explanations\plugins\manager.py | 103 | 172 |
| src\calibrated_explanations\plugins\manager.py | 104 | 172 |
| src\calibrated_explanations\plugins\manager.py | 107 | 172 |

## Top hotspot blocks

| File | Start | End | Test count | Length |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 34 | 35 | 223 | 2 |
| src\calibrated_explanations\utils\helper.py | 72 | 72 | 192 | 1 |
| src\calibrated_explanations\utils\helper.py | 82 | 83 | 192 | 2 |
| src\calibrated_explanations\utils\helper.py | 73 | 73 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 95 | 95 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 100 | 100 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 103 | 103 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 106 | 106 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 108 | 108 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 111 | 111 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 114 | 114 | 191 | 1 |
| src\calibrated_explanations\utils\helper.py | 192 | 192 | 177 | 1 |
| src\calibrated_explanations\utils\helper.py | 194 | 196 | 177 | 3 |
| src\calibrated_explanations\utils\helper.py | 200 | 200 | 177 | 1 |
| src\calibrated_explanations\plugins\manager.py | 88 | 89 | 172 | 2 |
| src\calibrated_explanations\plugins\manager.py | 92 | 93 | 172 | 2 |
| src\calibrated_explanations\plugins\manager.py | 96 | 99 | 172 | 4 |
| src\calibrated_explanations\plugins\manager.py | 102 | 104 | 172 | 3 |
| src\calibrated_explanations\plugins\manager.py | 107 | 110 | 172 | 4 |
| src\calibrated_explanations\plugins\manager.py | 113 | 115 | 172 | 3 |
| src\calibrated_explanations\plugins\manager.py | 117 | 119 | 172 | 3 |
| src\calibrated_explanations\plugins\manager.py | 121 | 123 | 172 | 3 |
| src\calibrated_explanations\plugins\manager.py | 125 | 128 | 172 | 4 |
| src\calibrated_explanations\plugins\manager.py | 130 | 132 | 172 | 3 |
| src\calibrated_explanations\plugins\manager.py | 136 | 136 | 172 | 1 |
| src\calibrated_explanations\plugins\manager.py | 139 | 144 | 172 | 6 |
| src\calibrated_explanations\plugins\manager.py | 147 | 149 | 172 | 3 |
| src\calibrated_explanations\plugins\registry.py | 108 | 112 | 161 | 5 |
| src\calibrated_explanations\plugins\registry.py | 114 | 114 | 161 | 1 |
| src\calibrated_explanations\plugins\registry.py | 119 | 120 | 161 | 2 |

## Suggested process

1. Run pytest with `--cov-context=test` to record per-test contexts.
2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.
3. Run this script and inspect the top hotspots above.
4. For each hotspot, review tests for duplicate assertions or setup-only coverage.
5. Consolidate redundant tests and re-run the reports to confirm lower counts.
