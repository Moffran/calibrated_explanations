# CE-REQ-NARR-API-001 — Narrative Output API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-NARR-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-NARR-001 |
| status | active |

## Scope

Public API: `CalibratedExplanations.to_narrative()`

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explain_factual-to_narrative.

## Observable behavior

When `explain_factual(X)` has returned a `CalibratedExplanations` instance, calling
`to_narrative()` on it must:

1. Return without raising an exception.
2. Return a non-`None` value.
3. For `output_format='text'`, the returned value must be a non-empty string.

## Acceptance criterion

For `explanations` being the result of `explain_factual(X_test)`:

- `explanations.to_narrative(output_format='text')` completes without error.
- The result is not `None`.
- `isinstance(result, str)` is `True`.
- `len(result) > 0`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_return_non_empty_string_when_narrative_text_format`

(in `tests/capabilities/test_narrative_contracts.py`)

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

This requirement verifies API contract and return type only. It does not verify:

- Narrative quality, fluency, or domain correctness.
- Template rendering accuracy.
- Regulatory suitability of generated text.
- DataFrame output format (requires pandas; tested separately).

See `CE-CAP-NARR-001` for the full assumption statement.
