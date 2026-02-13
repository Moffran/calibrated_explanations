# DEADCODE-HUNTER PROPOSAL (Updated 2026-02-13)

## Summary

No high-confidence dead production code was identified.

## Private Method Analysis

Source: `reports/anti-pattern-analysis/private_method_analysis.csv`

Pattern counts:
- `Consistent (Internal Only)`: 348
- `Pattern 2 (Local Test Helper)`: 2 (both in tests)
- `Pattern 3 (Completely Dead)`: 1

Pattern 3 symbol:
- `_missing_` in `src/calibrated_explanations/explanations/reject.py`

Assessment:
- `_missing_` is an Enum protocol hook and can be invoked by Enum construction semantics.
- Treat as **not removable** without semantic regression risk.

## Large Gap Review

From `reports/over_testing/gaps.csv`, largest uncovered blocks remain in runtime-heavy modules:
- `plotting.py` large segments (e.g. 1029-1486)
- `core/calibrated_explainer.py` large segments
- `core/explain/orchestrator.py` and helpers
- `explanations/explanation.py` and `explanations/explanations.py`

Classification:
- Predominantly **untested or under-tested production behavior**, not dead code.
- Reachability is supported by active public APIs and plugin paths.

## Lazy/Dynamic Reachability Check

Given active plugin registry usage and lazy imports (`__init__.py`), apparent low-hit branches can still be reachable through:
- plugin discovery/overrides
- optional integrations
- conditional runtime branches

No removal candidates in `src/` are recommended from deadcode perspective this iteration.

## Recommendations

1. Keep dead-code removal scope minimal (none in `src/` this round).
2. Prioritize behavioral tests for high-miss modules (see test-creator proposal) over source deletion.
3. If dead-code cleanup is required, only consider symbols with both:
- zero src usage,
- and no protocol/dynamic dispatch role.

Current confirmed dead/removable findings in source: **none**.
