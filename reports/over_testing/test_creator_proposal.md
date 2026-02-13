# TEST-CREATOR PROPOSAL (Updated 2026-02-13)

## Objective

Support continued removal of no/very-low-value tests while preserving the 90% global gate and 95% critical-module gates.

Current status:
- Global coverage: 90.00%
- Critical gates: all pass (`scripts/quality/check_coverage_gates.py`)

## Highest-Impact Gap Targets

From fresh coverage (`coverage.xml` + pytest output), top miss-heavy modules:
- `explanations/explanation.py` (190 misses)
- `plotting.py` (174 misses)
- `explanations/explanations.py` (91 misses)
- `core/explain/orchestrator.py` (68 misses)
- `viz/builders.py` (61 misses)

For removal safety, prioritize small, deterministic public-path tests in medium-size helpers first.

## Tier 1 Targets (best gain/effort)

### 1) `plotting.py`

Strategy:
- Public helper/function branch tests with controlled import failure and style-chain input permutations.

Expected gain:
- 10-25 statements per compact test module.

### 2) `core/discretizer_config.py`

Strategy:
- `instantiate_discretizer` edge handling for condition label fallback paths and mode-sensitive construction.

Expected gain:
- 8-15 statements.

### 3) `core/reject.py` shim

Strategy:
- Direct file execution import fallback and explicit re-raise paths under controlled import failure.

Expected gain:
- 5-10 statements.

### 4) `viz/plotspec.py` mapping compatibility

Strategy:
- Public dataclass mapping-style helper methods (`setdefault`, `get`, `__getitem__`).

Expected gain:
- 6-12 statements.

## Tier 2 Targets

### 5) `plugins/explanations_fast.py`

Strategy:
- Registration no-op path (descriptor already present) + registration path assertions.

### 6) `viz/coloring.py`

Strategy:
- Normal + fallback conversion branches in `get_fill_color` and `color_brew` shape checks.

### 7) `cache/__init__.py` and `parallel/__init__.py`

Strategy:
- Lazy package exports and `AttributeError` path checks via public import surface.

## Proposed Backfill Plan for Next Removal Cycle

Before or during removal of low-value files:
1. Add 3-5 compact behavioral tests in targets 1-4.
2. Remove first mini-batch (2-3 low-value files).
3. Re-run full suite and coverage.
4. Repeat until near 90.1%; then pause removals and add one more Tier 1 slice.

## Constraints Compliance

- Public APIs only (private scanner currently clean).
- Deterministic tests only.
- No import-only placeholders.

## Expected Outcome

Sufficient buffer to continue pruning extremely low-value test files while keeping:
- global coverage >= 90%
- critical module gates green.
