**Over-Testing Triage: Fast estimator-driven workflow**

Purpose
- Provide a fast, safe way to simulate large batches of test removals without running the slow `run_over_testing_pipeline.py` after every micro-change.
- Offer small utilities to identify coverage gaps and scaffold minimal focused tests to raise baseline coverage to the required threshold.

Core scripts
- `scripts/over_testing/estimator.py` — read per-test unique-line CSV and baseline line coverage CSV/JSON; estimate coverage after removing a set of tests and recommend low-value tests.
- `scripts/over_testing/gap_analyzer.py` — parse line-level coverage CSV and print contiguous untested blocks meeting a length threshold.
- `scripts/over_testing/generate_test_templates.py` — create minimal test templates for manual completion to fill identified gaps.

Recommended process when baseline < COVERAGE_TARGET (90%)
1. Run the heavy pipeline once to get accurate `reports/over-testing/*` outputs.
2. Use `estimator.py` with `--recommend` to get candidates sorted by low value_score (unique_lines/runtime).
3. If many zero-unique tests exist, remove them in a large batch (e.g., hundreds) after validating with `estimator.py --remove-list`.
4. If the estimator predicts coverage < 90% after removals, run `gap_analyzer.py` to locate untested blocks and generate templates using `generate_test_templates.py`.
5. Fill in the minimal tests (one test per control path), commit removals + added tests in the same branch, then run the full pipeline once to verify.

Safety rules
- Never apply removals whose estimated coverage < 90%.
- Always skip or protect `tests/docs` and tests with `docs`/`rtd` markers.
- Keep batch sizes large enough to amortize the cost of the slow pipeline (prefer hundreds when safe).

Command examples
```
python scripts/over_testing/estimator.py --per-test reports/over-testing/per_test_summary.csv --baseline reports/over-testing/line_coverage_counts.csv --recommend --budget 1000
python scripts/over_testing/estimator.py --per-test reports/over-testing/per_test_summary.csv --baseline reports/over-testing/line_coverage_counts.csv --remove-list candidates.txt
python scripts/over_testing/gap_analyzer.py --line-csv reports/over-testing/line_coverage_counts.csv --threshold 20 > gaps.csv
python scripts/over_testing/generate_test_templates.py --gaps-csv gaps.csv --out-dir tests/generated
```

When to run the full pipeline
- After applying a large, estimator-approved batch of removals and any minimal tests that restore coverage — run `python scripts/over-testing/run_over_testing_pipeline.py` once to get authoritative metrics.

Notes
- These tools operate on the CSV/JSON artifacts produced by the heavy pipeline; they are intended to be conservative helpers and not replacements for a final authoritative run.
