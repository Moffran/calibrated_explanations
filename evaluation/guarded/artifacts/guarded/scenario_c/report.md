# Scenario C: Regression Invariants

## What this scenario tests

Scenario C asks: does the regression-specific code path preserve the interval invariant (low ≤ predict ≤ high)?

Two datasets: (1) synthetic sin(x) + noise with known OOD boundary at |x| > 3.5; (2) sklearn diabetes dataset.
Models: RandomForestRegressor and Ridge.
explain_guarded_factual is called for all configurations.

## Primary metric: interval invariant violations

**n_invariant_violations = 0**

PASS: no violations found. The interval invariant holds everywhere.

Interval invariant warnings captured: 0. The regression code path uses warnings.warn instead of raise, so violations would go unnoticed in normal use without this check.

## Secondary diagnostic: guard responsiveness to OOD

Mean intervals_removed_guard for ID vs OOD instances on synthetic dataset.

|                         |   intervals_removed_guard |
|:------------------------|--------------------------:|
| ('synthetic_1d', False) |                         0 |
| ('synthetic_1d', True)  |                         0 |

OOD instances (|x| > 3.5) are expected to have more intervals removed than ID instances (|x| ≤ 3). A non-zero difference supports guard responsiveness; zero for both may indicate the OOD shift is insufficient relative to the calibration distribution at this significance.
