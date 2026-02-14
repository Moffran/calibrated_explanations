# ANTI-PATTERN AUDITOR PROPOSAL (Updated 2026-02-13)

## Audit Result

Current anti-pattern posture is strong and clean.

### Scanner outputs

- `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py`
  - Found: **0 anti-patterns**
- `python scripts/anti-pattern-analysis/scan_private_usage.py --check`
  - Found: **0 private-member violations**
- `reports/anti-pattern-analysis/test_anti_pattern_report.csv`
  - header only (no findings)
- `reports/anti-pattern-analysis/private_usage_scan.csv`
  - header only (no findings)

## Private Method Pattern Findings

From `private_method_analysis.csv`:
- `Pattern 2 (Local Test Helper)`: 2
  - `_get_explainer` (`tests/unit/explanations/test_explanation_more.py`)
  - `_make_binary_explainer` (`tests/unit/explanations/test_conjunction_hardening.py`)
- `Pattern 3 (Completely Dead)`: `_missing_` in source (see deadcode proposal; likely protocol method false-positive)

These are advisory only; no immediate blocker.

## Marker / Hygiene Status

No anti-pattern scanner blockers were produced in this run.
Unknown-marker warnings may still appear during pytest execution, but these were not surfaced as anti-pattern violations by the current scanner configuration.

## Recommendations

1. Keep current standards: no private-member test access and no placeholder assertions.
2. Optional hygiene follow-up:
- rename local underscored helpers in tests to explicit helper names,
- centralize shared helpers discovered by `find_shared_helpers.py` where practical.
3. Maintain scanner checks in every pruning iteration to prevent regressions.

## Severity Summary

| Category | Count | Severity |
| --- | ---: | --- |
| Hard blockers | 0 | None |
| Medium advisory | 2 | Local test-helper naming/duplication |
| Low advisory | 1 | Source `_missing_` static-analysis false-positive |
