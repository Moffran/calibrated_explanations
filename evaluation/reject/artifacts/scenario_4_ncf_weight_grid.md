# Scenario 4 — NCF and blend weight grid

Rows: 40

## Key findings

- NCFs tested: default, ensured. Entropy and explicit hinge/margin modes are excluded from this suite.
- accept_rate is the fraction of accepted instances — NOT ICP label-set coverage.
- w >= 0.7 converges NCF behavior; w=0.3 amplifies differences between NCFs where present.
- Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.

## Outcome snapshot

- **rows**: 40
- **datasets**: 5
- **best_accuracy_delta**: 0.2255
- **ncfs_tested**: ['default', 'ensured']
- **w_values_tested**: [0.3, 0.5, 0.7, 1.0]

## Result table

| task_type | dataset | ncf | w | accept_rate | accepted_accuracy | accepted_accuracy_delta |
|---|---|---|---|---|---|---|
| binary | breast_cancer | default | 0.3000 | 0.9825 | 0.9643 | -0.0006 |
| binary | breast_cancer | default | 0.5000 | 0.9825 | 0.9643 | -0.0006 |
| binary | breast_cancer | default | 0.7000 | 0.9825 | 0.9643 | -0.0006 |
| binary | breast_cancer | default | 1.0000 | 0.9825 | 0.9643 | -0.0006 |
| binary | breast_cancer | ensured | 0.3000 | 0.5614 | 0.9844 | 0.0195 |
| binary | breast_cancer | ensured | 0.5000 | 0.6228 | 0.9718 | 0.0069 |
| binary | breast_cancer | ensured | 0.7000 | 0.9825 | 0.9643 | -0.0006 |
| binary | breast_cancer | ensured | 1.0000 | 0.9825 | 0.9643 | -0.0006 |
| binary | colic | default | 0.3000 | 0.8333 | 0.8667 | 0.0333 |
| binary | colic | default | 0.5000 | 0.8333 | 0.8667 | 0.0333 |
| binary | colic | default | 0.7000 | 0.8333 | 0.8667 | 0.0333 |
| binary | colic | default | 1.0000 | 0.8333 | 0.8667 | 0.0333 |

_Showing first 12 of 40 rows._
