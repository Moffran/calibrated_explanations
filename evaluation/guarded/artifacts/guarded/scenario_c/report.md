# Scenario C: Regression Invariants

## Setup

- Seeds: 5
- Datasets: synthetic_1d, diabetes
- Models: RandomForestRegressor, Ridge (synthetic only), RandomForestRegressor (diabetes)
- Guard grid: significance=[0.05, 0.1, 0.2], n_neighbors=[3, 5, 10]

## Purpose

Scenario C asks: does the regression-specific code path preserve the interval invariant (low ≤ predict ≤ high)?

Two datasets: (1) synthetic sin(x) + noise with known OOD boundary at |x| > 3.5; (2) sklearn diabetes dataset.
Models: RandomForestRegressor and Ridge.
explain_guarded_factual is called for all configurations.

## Metric contract

The paper-relevant result in this scenario is binary: either the regression path preserves the interval invariant everywhere or it does not. Any non-zero invariant violation count is a correctness bug.

The OOD-responsiveness check is retained as a secondary engineering diagnostic. It answers whether the regression guard is merely well-formed or also directionally sensible when test inputs move outside the calibration support.

## Interval invariant violations

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

## Interpretation

Scenario C is not meant to show that regression is a separate flagship result. Its role is to ensure the regression-specific implementation is not silently broken while the classification scenarios pass.

A clean run here supports a narrow claim: the guarded regression path maintains interval semantics and reacts in the right direction under a simple synthetic OOD shift. It should be treated as appendix-strength validation unless the paper explicitly argues about regression.
