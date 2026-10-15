# CE-REQ-PRED-PROB-INSTANCE-001 — Instance-wise Probabilistic Regression Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-PROB-INSTANCE-001 |
| obligation_type | output_contract |
| claim_refs | CE-CAP-PRED-PROB-001 |
| adr_refs | ADR-021 |
| status | active |
| verification_status | verified |
| applicable_on | WrapCalibratedExplainer.predict_proba with threshold for regression |
| tif_refs | CE-TIF-PRED-PROB-002 |
| verification_strength | numerical_behavior |
| evidence_level | raw_evidence |

## Scope

Public API: `WrapCalibratedExplainer.predict_proba(X, uq_interval=True, threshold=t)`
for probabilistic regression, with a scalar threshold `t` (event `y <= t`) or a
tuple threshold `(low, high)` (event `low < y <= high`).

Applicable task types: probabilistic_regression.

Applicable workflow: standard offline fit-calibrate-predict with threshold, with or
without Mondrian bins.

## Observable behavior

Thresholded regression is instance-wise. For a frozen calibration state, the
calibrated event probability and its Venn-Abers interval `(low, high)` for a test
object are a function of that object's model input (and Mondrian bin) only:

1. The output for an object is identical whether it is predicted alone or within a
   larger batch, at any batch position and under any permutation of the batch.
2. Two objects with identical model inputs receive identical outputs, including when
   they differ in features that the learner does not use.
3. Repeated identical calls return identical outputs.

The calibration chain is an inductive Venn-Abers predictor whose score is the
deterministic (unsmoothed) conformal predictive distribution value of a CPS fitted
on one half of the calibration set; the same score function is applied to the
other half (Venn-Abers calibration) and to test objects.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted and calibrated for regression, a threshold `t`
and a test matrix `X_test`:

- `predict_proba(X_test, uq_interval=True, threshold=t)` equals, element-wise and
  exactly (`np.array_equal`), the row-wise concatenation of
  `predict_proba(X_test[i:i+1], uq_interval=True, threshold=t)`, for the point
  estimate and both interval bounds.
- The same equality holds after reversing the row order of `X_test` and mapping
  results back.

## TIF reference

**TIF ID:** CE-TIF-PRED-PROB-002

TIF scenario function: `run_prob_regression_instance_tif_scenario()`
Observation fields: `max_abs_batch_deviation`, `max_abs_permutation_deviation`

## Verification method

Automated pytest tests calling CE-TIF-PRED-PROB-002, plus the integration suite
`tests/integration/core/test_thresholded_regression_batch_invariance.py` (multiple
learners, seeds, tied predictions, integer targets and Mondrian bins).

## Verification targets

- `pytest: tests/capabilities/test_probabilistic_regression_contracts.py::test_should_return_batch_invariant_probabilities_when_regression_threshold_query`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| tif_ids | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies instance-wise determinism of the calibrated output. It
does not verify:

- Frequency-calibration of the threshold event probability.
- Behaviour after `insert_calibration` (online calibration changes the frozen state;
  outputs are instance-wise with respect to the updated state).
- Learners that are themselves batch-dependent (for example, models whose
  `predict` normalises over the batch); CE cannot repair a learner's own coupling.
