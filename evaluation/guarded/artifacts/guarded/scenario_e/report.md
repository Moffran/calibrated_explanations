# Scenario E: Edge Case Behavior

## Setup

- Cases: E1, E2, E3, E4, E5, E6, E7, E8
- Scope: parameter-space boundaries and failure-mode checks
- Output: PASS/FAIL regression artifact, not a paper-facing benchmark

## Purpose

Scenario E asks: does the guard's API behave predictably at the extremes of its parameter space — without exceptions, with documented behavior?

Each case targets a specific code-path boundary. A PASS means the observed behavior matches what is expected or documented. Cases E2 and E4 have expected behavior that may seem surprising (guard does nothing, or saturates) — these are design boundaries, not bugs.

## How to read this report

This scenario is intentionally case-based rather than aggregate. The question is not whether the average metric looks good; it is whether the implementation fails in specific brittle situations that users will hit in practice.

A FAIL indicates either a crash, silent corruption, or behavior that contradicts the documented contract for that edge case. A PASS means the observed behavior is acceptable, even if that behavior is a known limitation rather than a strength.

## Results

**7 PASS / 1 FAIL** out of 8 cases.

| Case | Status | Details |
|---|---|---|
| E1 | ✓ PASS | All 5 OOD instances had 0 emitted rules. API stable. |
| E2 | ✓ PASS | n_cal=200, significance=0.001. Unsmoothed estimator: min p-value can be 0. Observed n_removed_guard=1, min_p_value=0.000 |
| E3 | ✓ PASS | n_neighbors=1 completed without exception. No NaN/inf in p-values. Note: high variance across seeds is expected with n_n |
| E4 | ✗ FAIL | OSError: [WinError -1066598273] Windows Error 0xc06d007f |
| E5 | ✓ PASS | Merge integrity confirmed. Total merged bins: 59. All merged bins are conforming; no non-conforming bin is merged. |
| E6 | ✓ PASS | Single-feature dataset processed without error. Unique features in audit: 1 (expected 1). |
| E7 | ✓ PASS | 10 identical calibration-set instances processed. No NaN/inf. p-values are > 0 for the majority of intervals (instance i |
| E8 | ✓ PASS | Guard raised ValueError on NaN-feature input: Input X contains NaN. NearestNeighbors does not accept missing values enco |

## Failures (if any)

**E4** — OSError: [WinError -1066598273] Windows Error 0xc06d007f

Expected: No exception when n_neighbors equals or exceeds n_cal. Guard must fail open: fraction_removed ≤ 0.1 (guard is effectively disabled).

## Interpretation

**E2 (significance < 1/n_cal)**: p-values are discrete with step 1/n_cal. Very small significance settings therefore stop being meaningful on finite calibration sets. This is a configuration boundary users need documented.

**E4 (n_neighbors >= n_cal)**: The guard saturates k_actual at n_cal. With all calibration points included as neighbours, all test instances appear in-distribution (guard fails open). Verified by fraction_removed check: must be ≤ 10% when saturated.

**E5 (merge_adjacent)**: Both negative (non-conforming not merged) and positive (merged bins are conforming) integrity checks are enforced.

**E7 (identical instances)**: Calibration-set member must yield p > 0 for the majority of intervals; all-zero p is treated as a FAIL.

**E8 (NaN features)**: Guard must either raise a clear exception or process without NaN contamination propagating into clean rows.

Scenario E should stay in the engineering suite. It is useful because it defines the operational envelope of the guard, not because it provides a headline effectiveness result.
