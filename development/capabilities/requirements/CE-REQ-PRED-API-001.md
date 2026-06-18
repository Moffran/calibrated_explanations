# CE-REQ-PRED-API-001 — Uncertainty Interval API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-PRED-001 |
| status | active |

## Scope

Public APIs:
- `WrapCalibratedExplainer.predict(X, uq_interval=True)`
- `WrapCalibratedExplainer.predict_proba(X, uq_interval=True)` (classification)

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: standard offline fit-calibrate-predict.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated, calling
`predict(X, uq_interval=True)` must:

1. Return a tuple `(predictions, (low, high))` where `predictions`, `low`, and
   `high` each have the same length as the number of rows in `X`.
2. For each instance `i`, the ordering invariant `low[i] <= predictions[i]` must
   hold for the point prediction relative to the lower bound.
3. For each instance `i`, the ordering invariant `predictions[i] <= high[i]` must
   hold for the point prediction relative to the upper bound.
4. `low` and `high` must be numeric arrays, not `None`.

For `predict_proba(X, uq_interval=True)` (classification), the same shape and
ordering invariants apply to the returned probability and bounds arrays.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted and calibrated on compatible data:

- `predict(X_test, uq_interval=True)` returns a 2-tuple `(y_hat, (low, high))`.
- `len(y_hat) == len(X_test)`.
- For all `i`: `low[i] <= high[i]` (interval bounds are ordered).
- `low` and `high` are not `None`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID: `test_should_return_uncertainty_interval_when_uq_interval_true`
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

This requirement verifies the API contract and structural invariants (return shape,
`low <= high`) only. It does not verify:

- That coverage at any stated confidence level is achieved.
- Statistical validity of the intervals under the exchangeability assumption.
- That the calibration set size is sufficient for accurate coverage.
- Asymptotic or finite-sample coverage guarantees from conformal prediction theory.

See `CE-CAP-PRED-001` for the full assumption statement.
