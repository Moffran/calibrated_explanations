Estimator & workflow scripts for fast, safe over-testing triage.

Files:
- `estimator.py` — simulate coverage impact of removing tests using per-test unique-line counts.
- `gap_analyzer.py` — detect contiguous untested blocks from a line-level CSV.
- `detect_redundant_tests.py` — Identify tests with identical coverage fingerprints (redundant).
- `generate_test_templates.py` — create minimal test templates for manual completion.

Quick workflow:
1. Run the heavy pipeline once to produce `reports/over-testing/*` CSVs.
2. Use `detect_redundant_tests.py` to identify test groups with identical fingerprints.
3. Use `estimator.py --per-test <per_test.csv> --baseline <line_coverage_counts.csv> --recommend` to get recommended low-value tests.
4. Create a `candidates.txt` with tests selected for removal, then `estimator.py --per-test ... --baseline ... --remove-list candidates.txt` to estimate resulting coverage.
5. If coverage is below `0.90`, use `gap_analyzer.py` to find untested blocks and `generate_test_templates.py` to scaffold minimal tests.
6. Commit large safe removal batches and run the full pipeline once to verify.

See docs/improvement/archived/over_testing_method.md for a detailed method and examples.

Updating `redundant_tests.csv` (guidance):

- Preferred: always regenerate the report by running the pipeline and script:

	```bash
	python scripts/over_testing/run_over_testing_pipeline.py
	python scripts/over_testing/extract_per_test.py
	python scripts/over_testing/detect_redundant_tests.py
	```

- If you have a confirmed false-positive or need to record an exceptional decision,
	do NOT overwrite the generated file without audit. Instead, create
	`reports/over_testing/redundant_tests_review.csv` with these columns:

	`fingerprint,test_count,lines_covered,unique_lines_per_test,description,tests,status,reviewer,notes`

	Use `status` = `ACCEPTED` / `REJECTED` / `UNDER_REVIEW`. Commit the review CSV
	and reference it from `reports/over_testing/final_remedy_plan.md`.
