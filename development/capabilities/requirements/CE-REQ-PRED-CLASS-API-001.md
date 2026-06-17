# CE-REQ-PRED-CLASS-API-001 — Classification Prediction API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-CLASS-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-PRED-CLASS-001 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.predict_proba(X)` and `WrapCalibratedExplainer.predict(X)`

Applicable task types: binary classification, multiclass classification.

Applicable workflow: standard offline fit-calibrate-predict.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated for a classification task,
calling `predict_proba(X)` and `predict(X)` must:

1. Return without raising an exception for valid inputs `X`.
2. `predict_proba(X)` returns an array-like with shape `(len(X), n_classes)` where all
   values are in `[0, 1]`.
3. `predict(X)` returns an array-like with length `len(X)`.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal` for binary classification:

- `predict_proba(X_test)` completes without error.
- `len(predict_proba(X_test)) == len(X_test)`.
- All values in `predict_proba(X_test)` are in `[0, 1]`.
- `predict(X_test)` completes without error.
- `len(predict(X_test)) == len(X_test)`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test IDs:
- `test_should_return_bounded_probabilities_when_classification_fitted_and_calibrated`
- `test_should_return_class_labels_when_classification_fitted_and_calibrated`

(in `tests/capabilities/test_classification_contracts.py`)

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

- Venn-Abers calibration validity in a finite-sample sense.
- Coverage guarantees at any specific confidence level.
- Posterior probability accuracy (predict_proba values ≠ true posteriors).
- Distribution shift robustness.

See `CE-CAP-PRED-CLASS-001` for the full assumption statement.
