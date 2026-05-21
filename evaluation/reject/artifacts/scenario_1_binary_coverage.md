# Scenario 1 — Binary marginal coverage sweep

Rows: 9

## Key findings

- Coverage is reported as standard label-set coverage from the conformal prediction sets.
- Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.
- Observed coverage violations: 5/9 (0.5556).
- Structural violations (CI upper bound < 1-epsilon, cannot be attributed to finite-sample noise): 0/9.

## Outcome snapshot

- **datasets**: 3
- **rows**: 9
- **violation_rate**: 0.5556
- **structural_violations**: 0
- **mean_coverage**: 0.9354

## Result table

| dataset | epsilon | n_cal | n_test | coverage | lower_ci | upper_ci | violation | structural_violation | reject_rate | accepted_accuracy_empirical |
|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.0100 | 114 | 114 | 0.9737 | 0.9250 | 0.9945 | yes | no | 0.0614 | 0.9720 |
| breast_cancer | 0.0500 | 114 | 114 | 0.9474 | 0.8890 | 0.9804 | yes | no | 0.0175 | 0.9643 |
| breast_cancer | 0.1000 | 114 | 114 | 0.9123 | 0.8446 | 0.9571 | no | no | 0.0614 | 0.9720 |
| colic | 0.0100 | 72 | 72 | 1.0000 | 0.9501 | 1.0000 | no | no | 1.0000 | nan |
| colic | 0.0500 | 72 | 72 | 0.8889 | 0.7928 | 0.9508 | yes | no | 0.1667 | 0.8667 |
| colic | 0.1000 | 72 | 72 | 0.8194 | 0.7111 | 0.9002 | yes | no | 0.0139 | 0.8451 |
| diabetes | 0.0100 | 154 | 154 | 0.9935 | 0.9644 | 0.9998 | no | no | 0.9026 | 0.9333 |
| diabetes | 0.0500 | 154 | 154 | 0.9870 | 0.9539 | 0.9984 | no | no | 0.7208 | 0.9535 |
| diabetes | 0.1000 | 154 | 154 | 0.8961 | 0.8368 | 0.9394 | yes | no | 0.3247 | 0.8462 |
