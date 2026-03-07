# Scenario E: Edge Case Behavior

## What this scenario tests

Scenario E asks: does the guard's API behave predictably at the extremes of its parameter space — without exceptions, with documented behavior?

Each case targets a specific code-path boundary. A PASS means the observed behavior matches what is expected or documented. Cases E2 and E4 have expected behavior that may seem surprising (guard does nothing, or saturates) — these are design boundaries, not bugs.

## Results

**7 PASS / 0 FAIL** out of 7 cases.

| Case | Status | Details |
|---|---|---|
| E1 | ✓ PASS | All 5 OOD instances had 0 emitted rules. API stable. |
| E2 | ✓ PASS | n_cal=200, significance=0.001. Unsmoothed estimator: min p-value can be 0. Observed n_removed_guard=1, min_p_value=0.000 |
| E3 | ✓ PASS | n_neighbors=1 completed without exception. No NaN/inf in p-values. Note: high variance across seeds is expected with n_n |
| E4 | ✓ PASS | n_neighbors=40 > n_cal=30. No crash. Audit rows: 95. Note: with saturated k, all p-values may be uniformly high. |
| E5 | ✓ PASS | No non-conforming bins tagged as is_merged=True. Total merged bins: 59. |
| E6 | ✓ PASS | Single-feature dataset processed without error. Unique features in audit: 1 (expected 1). |
| E7 | ✓ PASS | 10 identical test instances processed. No NaN/inf in p-values. |

## Failures (if any)

No failures.

## Design boundaries documented by PASS cases

**E2 (significance < 1/n_cal)**: p-values are discrete with step 1/n_cal. When significance < 1/n_cal, the guard can never reject any interval — it silently behaves as significance=0. Users must be warned about this.

**E4 (n_neighbors >= n_cal)**: The guard saturates k_actual at n_cal. With all calibration points included as neighbors, all test instances may appear in-distribution. This is the correct saturating behavior.
