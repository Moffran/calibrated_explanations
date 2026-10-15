# CE-TIF-PRED-PROB-002 — Instance-wise Probabilistic Regression TIF Verification Interface

## Identity

| Field | Value |
|---|---|
| tif_id | CE-TIF-PRED-PROB-002 |
| executable | `development/capabilities/verification/tif/tif_prob_regression_instance.py` |
| entry_functions | `run_prob_regression_instance_tif_scenario(threshold=0.0)` |
| evidence_builder | `build_evidence_payload()` |
| evidence_key | PRED-PROB-002 |
| verification_type | numerical_behavior |
| status | active |

## Requirements served

| Requirement | Observation fields used |
|---|---|
| CE-REQ-PRED-PROB-INSTANCE-001 | `max_abs_batch_deviation`, `max_abs_permutation_deviation` |

## Claims served

- CE-CAP-PRED-PROB-001

## ADR refs

- ADR-021

## Public API surface under test

- `WrapCalibratedExplainer(model).fit(X, y).calibrate(X_cal, y_cal)`
- `explainer.predict_proba(X_test, uq_interval=True, threshold=t)` → event probability
  `P(y <= t | x)` with Venn-Abers interval `(low, high)`

## Fixture / data contract

The TIF scenario creates a deterministic regression dataset using `sklearn.datasets.make_regression`:

```python
make_regression(
    n_samples=200,
    n_features=4,
    n_informative=3,
    random_state=42,
    noise=10.0,
)
```

The dataset is split deterministically:
- 12 instances for `X_test`
- Remaining 188 split 60%/40% into proper-train and calibration

The model used is `RandomForestRegressor(n_estimators=10, random_state=42)`.

No external data, network, or clock access. All randomness is seeded.

## WrapCalibratedExplainer workflow

```python
explainer = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=10, random_state=42))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal)

single = [explainer.predict_proba(X_test[i:i+1], uq_interval=True, threshold=0.0) for i in ...]
batched = explainer.predict_proba(X_test, uq_interval=True, threshold=0.0)
reversed_ = explainer.predict_proba(X_test[::-1], uq_interval=True, threshold=0.0)
```

## Observation fields

The TIF returns a `ProbRegressionInstanceObservation` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `max_abs_batch_deviation` | `float` | Max \|batched − one-at-a-time\| over point estimate, low and high |
| `max_abs_permutation_deviation` | `float` | Max \|reversed batch − one-at-a-time\| over the same values |
| `threshold` | `float` | The threshold used |
| `n_instances` | `int` | Number of test instances |

## Acceptance fields (mapping requirement → observation)

| Requirement | Acceptance criterion | Observation field | Expected |
|---|---|---|---|
| CE-REQ-PRED-PROB-INSTANCE-001 | Batched equals one-at-a-time | `max_abs_batch_deviation` | `== 0.0` |
| CE-REQ-PRED-PROB-INSTANCE-001 | Reversed batch equals one-at-a-time | `max_abs_permutation_deviation` | `== 0.0` |

## Evidence fields

A raw evidence record produced from this TIF should include:

```yaml
evidence_id: CE-EVID-PRED-PROB-002-<YYYYMMDD>
claim_ids:
  - CE-CAP-PRED-PROB-001
requirement_ids:
  - CE-REQ-PRED-PROB-INSTANCE-001
adr_refs:
  - ADR-021
tif_ids:
  - CE-TIF-PRED-PROB-002
verification_type: numerical_behavior
dataset_id: sklearn make_regression n_samples=200 n_features=4 n_informative=3 noise=10 random_seed=42
random_seed: 42
```

## TIF constraints

This TIF does NOT:
- Construct prediction objects directly
- Import from `calibrated_explanations.core.calibrated_explainer` directly
- Use private/internal CE APIs
- Perform final pytest assertions (only local sanity checks)
- Hide acceptance criteria (these belong in requirement files)
