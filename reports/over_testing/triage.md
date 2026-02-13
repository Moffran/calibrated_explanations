# Over-Testing Triage Report

## Metadata

- **min_count**: 20
- **summary**: reports\over_testing\summary.json
- **lines**: reports\over_testing\line_coverage_counts.csv
- **blocks**: reports\over_testing\block_coverage_counts.csv
- **coverage_file**: .coverage
- **contexts_detected**: 1550
- **context_regex**: None
- **source_root**: src\calibrated_explanations
- **over_testing_threshold**: 20
- **warnings**: []

## Top over-tested files

| File | Over ratio | Max count | Over-threshold lines | Lines covered |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\core\explain\sequential.py | 0.8271 | 58 | 110 | 133 |
| src\calibrated_explanations\core\explain\_computation.py | 0.6564 | 64 | 191 | 291 |
| src\calibrated_explanations\explanations\_conjunctions.py | 0.6013 | 74 | 92 | 153 |
| src\calibrated_explanations\calibration\summaries.py | 0.5806 | 124 | 36 | 62 |
| src\calibrated_explanations\core\explain\feature_task.py | 0.5501 | 65 | 236 | 429 |
| src\calibrated_explanations\core\explain\orchestrator.py | 0.5444 | 163 | 368 | 676 |
| src\calibrated_explanations\core\explain\_shared.py | 0.4975 | 60 | 99 | 199 |
| src\calibrated_explanations\core\prediction\validation.py | 0.4828 | 121 | 14 | 29 |
| src\calibrated_explanations\core\prediction\orchestrator.py | 0.4667 | 155 | 224 | 480 |
| src\calibrated_explanations\plugins\base.py | 0.4643 | 60 | 26 | 56 |
| src\calibrated_explanations\plugins\manager.py | 0.4469 | 186 | 202 | 452 |
| src\calibrated_explanations\core\explain\_helpers.py | 0.381 | 63 | 48 | 126 |
| src\calibrated_explanations\core\explain\parallel_runtime.py | 0.3591 | 58 | 65 | 181 |
| src\calibrated_explanations\utils\discretizers.py | 0.3376 | 70 | 53 | 157 |
| src\calibrated_explanations\core\config_helpers.py | 0.3333 | 175 | 28 | 84 |

## Top hotspot lines

| File | Line | Test count |
| --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 33 | 222 |
| src\calibrated_explanations\utils\exceptions.py | 34 | 222 |
| src\calibrated_explanations\plugins\manager.py | 88 | 186 |
| src\calibrated_explanations\plugins\manager.py | 89 | 186 |
| src\calibrated_explanations\plugins\manager.py | 92 | 186 |
| src\calibrated_explanations\plugins\manager.py | 93 | 186 |
| src\calibrated_explanations\plugins\manager.py | 96 | 186 |
| src\calibrated_explanations\plugins\manager.py | 97 | 186 |
| src\calibrated_explanations\plugins\manager.py | 98 | 186 |
| src\calibrated_explanations\plugins\manager.py | 99 | 186 |
| src\calibrated_explanations\plugins\manager.py | 102 | 186 |
| src\calibrated_explanations\plugins\manager.py | 103 | 186 |
| src\calibrated_explanations\plugins\manager.py | 104 | 186 |
| src\calibrated_explanations\plugins\manager.py | 107 | 186 |
| src\calibrated_explanations\plugins\manager.py | 108 | 186 |
| src\calibrated_explanations\plugins\manager.py | 109 | 186 |
| src\calibrated_explanations\plugins\manager.py | 110 | 186 |
| src\calibrated_explanations\plugins\manager.py | 113 | 186 |
| src\calibrated_explanations\plugins\manager.py | 114 | 186 |
| src\calibrated_explanations\plugins\manager.py | 115 | 186 |
| src\calibrated_explanations\plugins\manager.py | 117 | 186 |
| src\calibrated_explanations\plugins\manager.py | 118 | 186 |
| src\calibrated_explanations\plugins\manager.py | 119 | 186 |
| src\calibrated_explanations\plugins\manager.py | 121 | 186 |
| src\calibrated_explanations\plugins\manager.py | 122 | 186 |
| src\calibrated_explanations\plugins\manager.py | 123 | 186 |
| src\calibrated_explanations\plugins\manager.py | 125 | 186 |
| src\calibrated_explanations\plugins\manager.py | 126 | 186 |
| src\calibrated_explanations\plugins\manager.py | 127 | 186 |
| src\calibrated_explanations\plugins\manager.py | 128 | 186 |

## Top hotspot blocks

| File | Start | End | Test count | Length |
| --- | --- | --- | --- | --- |
| src\calibrated_explanations\utils\exceptions.py | 33 | 34 | 222 | 2 |
| src\calibrated_explanations\plugins\manager.py | 88 | 89 | 186 | 2 |
| src\calibrated_explanations\plugins\manager.py | 92 | 93 | 186 | 2 |
| src\calibrated_explanations\plugins\manager.py | 96 | 99 | 186 | 4 |
| src\calibrated_explanations\plugins\manager.py | 102 | 104 | 186 | 3 |
| src\calibrated_explanations\plugins\manager.py | 107 | 110 | 186 | 4 |
| src\calibrated_explanations\plugins\manager.py | 113 | 115 | 186 | 3 |
| src\calibrated_explanations\plugins\manager.py | 117 | 119 | 186 | 3 |
| src\calibrated_explanations\plugins\manager.py | 121 | 123 | 186 | 3 |
| src\calibrated_explanations\plugins\manager.py | 125 | 128 | 186 | 4 |
| src\calibrated_explanations\plugins\manager.py | 130 | 132 | 186 | 3 |
| src\calibrated_explanations\plugins\manager.py | 136 | 136 | 186 | 1 |
| src\calibrated_explanations\plugins\manager.py | 139 | 144 | 186 | 6 |
| src\calibrated_explanations\plugins\manager.py | 147 | 149 | 186 | 3 |
| src\calibrated_explanations\plugins\registry.py | 108 | 112 | 183 | 5 |
| src\calibrated_explanations\plugins\registry.py | 114 | 114 | 183 | 1 |
| src\calibrated_explanations\plugins\registry.py | 119 | 120 | 183 | 2 |
| src\calibrated_explanations\utils\helper.py | 72 | 72 | 183 | 1 |
| src\calibrated_explanations\utils\helper.py | 82 | 83 | 183 | 2 |
| src\calibrated_explanations\utils\helper.py | 73 | 73 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 95 | 95 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 100 | 100 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 103 | 103 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 106 | 106 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 108 | 108 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 111 | 111 | 182 | 1 |
| src\calibrated_explanations\utils\helper.py | 114 | 114 | 180 | 1 |
| src\calibrated_explanations\logging.py | 72 | 74 | 177 | 3 |
| src\calibrated_explanations\logging.py | 76 | 77 | 177 | 2 |
| src\calibrated_explanations\logging.py | 79 | 79 | 177 | 1 |

## Suggested process

1. Run pytest with `--cov-context=test` to record per-test contexts.
2. Run `scripts/over_testing/over_testing_report.py --require-multiple-contexts`.
3. Run this script and inspect the top hotspots above.
4. For each hotspot, review tests for duplicate assertions or setup-only coverage.
5. Consolidate redundant tests and re-run the reports to confirm lower counts.
