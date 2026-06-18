# CE-REQ-PRED-INTERVAL-BOUNDS-001 — Conformal Interval Percentile Selection API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-INTERVAL-BOUNDS-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-PRED-001 |
| status | active |

## Scope

Public API:
- `WrapCalibratedExplainer.predict(X, uq_interval=True, low_high_percentiles=(lo, hi))`

Applicable task types: regression.

Applicable workflow: standard offline fit-calibrate-predict with percentile selection.

This requirement covers the `low_high_percentiles` parameter only. The base
`uq_interval=True` interval shape and ordering contract is covered separately by
`CE-REQ-PRED-API-001`. `low_high_percentiles` is ignored for classification tasks
and when a `threshold` is provided (probabilistic regression).

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated for a regression task,
calling `predict(X, uq_interval=True, low_high_percentiles=(lo, hi))` must:

1. Accept the parameter without raising an exception for any `0 < lo < hi < 100`.
2. Return the standard `(y_hat, (low, high))` 2-tuple (same shape invariants as
   `CE-REQ-PRED-API-001`: `len(y_hat) == len(X)`, `low` and `high` not `None`).
3. Return ordered bounds: `low[i] <= high[i]` for all `i`.
4. Default to `(5, 95)` when the parameter is omitted — i.e., a call without
   `low_high_percentiles` produces the same bounds as `low_high_percentiles=(5, 95)`.
5. Produce narrower or equal bounds when the percentile range is contracted:
   for `(lo2, hi2)` with `lo <= lo2` and `hi2 <= hi`, the resulting bounds satisfy
   `low[i] <= low2[i]` and `high2[i] <= high[i]` for all `i`.
6. **One-sided lower-unbounded interval** (`lo = -np.inf`): accepted without error;
   the lower bound is set to `min(y_cal)` for every instance (a constant floor), while
   the upper bound is computed at the `hi`-th percentile as normal.
7. **One-sided upper-unbounded interval** (`hi = np.inf`): accepted without error;
   the upper bound is set to `max(y_cal)` for every instance (a constant ceiling), while
   the lower bound is computed at the `lo`-th percentile as normal.
8. One-sided intervals still satisfy `low[i] <= high[i]` for all `i`.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal` for a regression task:

- `predict(X_test, uq_interval=True, low_high_percentiles=(10, 90))` completes
  without error.
- The return value is a 2-tuple `(y_hat, (low, high))`.
- `len(y_hat) == len(X_test)`.
- For all `i`: `low[i] <= high[i]`.
- The same call with the default `(5, 95)` produces bounds that satisfy
  `low_5_95[i] <= low_10_90[i]` and `high_10_90[i] <= high_5_95[i]` for all `i`
  (the `(5, 95)` interval is at least as wide as the `(10, 90)` interval).
- `predict(X_test, uq_interval=True, low_high_percentiles=(-np.inf, 90))` completes
  without error; all lower bounds equal `min(y_cal)`; for all `i`: `low[i] <= high[i]`.
- `predict(X_test, uq_interval=True, low_high_percentiles=(10, np.inf))` completes
  without error; all upper bounds equal `max(y_cal)`; for all `i`: `low[i] <= high[i]`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs:
- `test_should_accept_low_high_percentiles_and_return_ordered_bounds_when_regression`
- `test_should_return_narrower_interval_when_percentiles_are_closer`
- `test_should_accept_neg_inf_lower_bound_and_return_constant_floor_when_regression`
- `test_should_accept_pos_inf_upper_bound_and_return_constant_ceiling_when_regression`

(in `tests/capabilities/test_prediction_contracts.py`)

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies the API contract and structural invariants only. It does not verify:

- That the returned bounds correspond to any specific theoretical conformal coverage level.
- Statistical validity of the intervals under the exchangeability assumption.
- That the monotonicity property holds under all model families or calibration set sizes
  (it is a structural property of quantile selection from a CPS; edge cases at
  degenerate calibration sets may violate it numerically).

See `CE-CAP-PRED-001` for the full assumption statement.
