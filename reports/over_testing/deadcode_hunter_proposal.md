# Deadcode Hunter Proposal (2026-02-17)

## Evidence
- `python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv`
- `python scripts/over_testing/gap_analyzer.py --line-csv reports/over_testing/line_coverage_counts.csv --threshold 10`

## Findings
- Private-method patterns:
  - `Pattern 3 (Completely Dead)`: 1 (`_missing_` in `src/calibrated_explanations/explanations/reject.py`).
  - `Pattern 1`: 2.
  - `Pattern 2`: 3.
- Largest uncovered contiguous blocks include:
  - `core/wrap_explainer.py:1303-1335` (33)
  - `parallel/parallel.py:341-370` (30)
  - `viz/narrative_plugin.py:519-543` (25)

## Recommendation
- Treat only `_missing_` as near-term dead-code candidate.
- Classify large uncovered blocks as test gaps first, not deletion targets.
