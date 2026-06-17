# CE-REQ-REJECT-API-001 — Reject Policy API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-REJECT-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-REJECT-001 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.explain_factual(X, reject_policy=RejectPolicySpec.flag())`

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explain_factual with RejectPolicySpec.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated, calling
`explain_factual(X, reject_policy=RejectPolicySpec.flag())` must:

1. Return without raising an exception.
2. Return a collection not equal to `None`.
3. The returned collection supports `len()` with `len(result) == len(X)`.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal`:

- `explain_factual(X_test, reject_policy=RejectPolicySpec.flag())` completes without error.
- The result is not `None`.
- `len(result) == len(X_test)`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_return_explanations_when_reject_policy_flag_provided`

(in `tests/capabilities/test_reject_policy_contracts.py`)

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

This requirement verifies the API contract only. It does not verify:

- That rejection tags are statistically optimal or accurately identify uncertain instances.
- Coverage guarantees for the reject/defer decision.
- Accuracy of the NCF (non-conformity function) scoring.

See `CE-CAP-REJECT-001` for the full assumption statement.
