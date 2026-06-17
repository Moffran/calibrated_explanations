# CE-REQ-EXPL-API-001 — Factual Explanation API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-001 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.explain_factual(X)`

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: standard offline fit-calibrate-explain.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated, calling
`explain_factual(X)` must:

1. Return without raising an exception for valid inputs `X`.
2. Return an object that supports indexing to retrieve per-instance explanations
   (`explanations[i]`).
3. Each per-instance explanation must expose feature-level contributions
   (accessible via `explanations[i].feature_weights` or equivalent public attribute).
4. The returned collection must have the same length as the number of rows in `X`.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal`:

- `explain_factual(X_test)` completes without error.
- `len(explain_factual(X_test)) == len(X_test)`.
- `explain_factual(X_test)[0]` is not `None`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID: `test_should_produce_factual_explanations_when_fitted_and_calibrated`
(in `tests/capabilities/test_explanation_contracts.py`)

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

This requirement verifies the API contract and structural invariants only. It does
not verify:

- Statistical calibration validity (depends on calibration-set representativeness).
- Exchangeability of calibration and test distributions.
- Coverage guarantees at any specific confidence level.
- Correctness of feature attribution magnitudes.

See `CE-CAP-EXPL-001` for the full assumption statement.
