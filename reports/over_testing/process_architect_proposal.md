# PROCESS-ARCHITECT PROPOSAL (Updated 2026-02-13)

## Executive Assessment

The test-quality method is operational and effective after this refresh.

Evidence:
- Full pipeline rerun succeeded: `scripts/over_testing/run_over_testing_pipeline.py`
- Valid context data: `contexts_detected = 1785`, no warnings
- Full suite green: `1543 passed, 1 skipped`
- Global coverage gate passes at 90.00%
- Critical per-module gates pass (`check_coverage_gates.py`)

## What Changed vs Prior State

Previous critical blocker (single-context data) is resolved.

Now reliable for decision-making:
- `per_test_summary.csv` unique-lines signal
- hotspot context maps
- over-testing triage ratios

## Process Gaps Still Open

### 1) Runtime signal in per-test summary is not useful

`per_test_summary.csv` still has `runtime=0`, causing estimator recommendations to be dominated by `inf` scores.

Impact:
- `estimator.py --recommend` ranking quality is degraded.

Recommendation:
- Patch `extract_per_test.py` to emit non-zero runtime (or remove runtime from score formula when unavailable).

### 2) Remove-list tooling mismatch

`scripts/over_testing/select_zero_unique_files.py` writes `path,count` rows.
`estimator.py --remove-list` expects test identifiers, leading to warnings and ineffective simulation.

Recommendation:
- Output one plain identifier per line (no count suffix), or update estimator parser accordingly.

### 3) Method docs are stale in role files

Role docs still describe the old one-context caveat as current state.

Recommendation:
- Update role instruction templates to avoid outdated assumptions and reduce analysis drift.

## Current Workflow Quality (Practical)

The following loop is efficient and should remain default:

1. `run_over_testing_pipeline.py`
2. `extract_per_test.py`
3. generate proposals
4. remove low-value tests in chunks
5. backfill with high-quality behavioral tests when near the gate
6. full-suite verification

This has already proven stable in the current repository state.

## CI Integration Recommendations

Per-PR:
- Keep existing gates: coverage >= 90, private-usage scan, anti-pattern scan.
- Add sanity check that `metadata.json` contexts > 1 in periodic quality jobs.

Scheduled (weekly):
- Run full over-testing pipeline and publish updated triage artifacts.
- Record trend metrics (zero-unique contexts, hotspot ratios, top-overlap files).

## Proposed Priority Order

1. Fix `select_zero_unique_files.py` output format.
2. Improve runtime signal in `extract_per_test.py` / estimator scoring.
3. Refresh role docs to match current data-quality status.
4. Continue chunked low-value pruning with test-creator backfill.

## Verdict

Process does **not** require structural redesign.
It is efficient as-is, with two tactical script fixes needed to improve recommendation quality.
