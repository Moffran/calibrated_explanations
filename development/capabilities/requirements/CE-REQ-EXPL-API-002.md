# CE-REQ-EXPL-API-002 — Alternative Explanation API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-API-002 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-002 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.explore_alternatives(X)`

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: standard offline fit-calibrate-explain.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated, calling
`explore_alternatives(X)` must:

1. Return without raising an exception for valid inputs `X`.
2. Return an object that supports indexing to retrieve per-instance alternative
   explanations (`explanations[i]`).
3. The returned collection must have the same length as the number of rows in `X`.
4. Each per-instance explanation must be distinct from a factual explanation — it
   represents shifts toward alternative predicted outcomes.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal`:

- `explore_alternatives(X_test)` completes without error.
- `len(explore_alternatives(X_test)) == len(X_test)`.
- `explore_alternatives(X_test)[0]` is not `None`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID: `test_should_produce_alternative_explanations_when_fitted_and_calibrated`
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

- That the alternative scenarios would be achievable in practice.
- That changed feature values remain within the natural data distribution.
- Statistical validity of the alternative exploration under distribution shift.
- Exchangeability of calibration and test distributions.

See `CE-CAP-EXPL-002` for the full assumption statement.
