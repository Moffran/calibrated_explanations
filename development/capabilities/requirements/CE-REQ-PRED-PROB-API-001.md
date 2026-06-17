# CE-REQ-PRED-PROB-API-001 — Probabilistic Regression Threshold Query API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-PROB-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-PRED-PROB-001 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.predict_proba(X, threshold=y_threshold)`

Applicable task types: regression.

Applicable workflow: standard offline fit-calibrate-predict with threshold query.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated for a regression task,
calling `predict_proba(X, threshold=y_threshold)` must:

1. Return without raising an exception for valid inputs `X` and a scalar `threshold`.
2. Return an array-like with length `len(X)`.
3. All returned values must be in `[0, 1]`.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal` for a regression task, and a scalar threshold `y_threshold`:

- `predict_proba(X_test, threshold=y_threshold)` completes without error.
- `len(predict_proba(X_test, threshold=y_threshold)) == len(X_test)`.
- All values in `predict_proba(X_test, threshold=y_threshold)` are in `[0, 1]`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test IDs:
- `test_should_return_bounded_probabilities_when_regression_threshold_query`
- `test_should_return_correct_length_when_regression_threshold_query`

(in `tests/capabilities/test_probabilistic_regression_contracts.py`)

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

This requirement verifies the API contract and output bounds only. It does not verify:

- CPS coverage guarantees at any specific significance level.
- Frequency-calibration of P(Y > threshold | X).
- Distribution shift robustness.

See `CE-CAP-PRED-PROB-001` for the full assumption statement.
