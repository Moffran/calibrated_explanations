Estimator & workflow scripts for fast, safe over-testing triage.

Files:
- `estimator.py` — simulate coverage impact of removing tests using per-test unique-line counts.
- `gap_analyzer.py` — detect contiguous untested blocks from a line-level CSV.
- `generate_test_templates.py` — create minimal test templates for manual completion.

Quick workflow:
1. Run the heavy pipeline once to produce `reports/over-testing/*` CSVs.
2. Use `estimator.py --per-test <per_test.csv> --baseline <line_coverage_counts.csv> --recommend` to get recommended low-value tests.
3. Create a `candidates.txt` with tests selected for removal, then `estimator.py --per-test ... --baseline ... --remove-list candidates.txt` to estimate resulting coverage.
4. If coverage is below `0.90`, use `gap_analyzer.py` to find untested blocks and `generate_test_templates.py` to scaffold minimal tests.
5. Commit large safe removal batches and run the full pipeline once to verify.

See docs/over_testing_method.md for a detailed method and examples.
