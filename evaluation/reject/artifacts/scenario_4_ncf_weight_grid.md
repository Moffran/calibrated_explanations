# Scenario 4 — NCF and blend weight grid

Rows: 368

## Key findings

- NCFs tested: default, ensured. Entropy and explicit hinge/margin modes are excluded from this suite.
- accept_rate is the fraction of accepted instances — NOT ICP label-set coverage.
- w >= 0.7 converges NCF behavior; w=0.3 amplifies differences between NCFs where present.
- Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.

## Outcome snapshot

- **rows**: 368
- **datasets**: 46
- **best_accuracy_delta**: 0.3810
- **ncfs_tested**: ['default', 'ensured']
- **w_values_tested**: [0.3, 0.5, 0.7, 1.0]

## Result table

| task_type | dataset | ncf | w | accept_rate | accepted_accuracy | accepted_accuracy_delta |
|---|---|---|---|---|---|---|
| binary | breast_cancer | default | 0.3000 | 0.9737 | 0.9640 | 0.0078 |
| binary | breast_cancer | default | 0.5000 | 0.9737 | 0.9640 | 0.0078 |
| binary | breast_cancer | default | 0.7000 | 0.9737 | 0.9640 | 0.0078 |
| binary | breast_cancer | default | 1.0000 | 0.9737 | 0.9640 | 0.0078 |
| binary | breast_cancer | ensured | 0.3000 | 0.5877 | 0.9851 | 0.0289 |
| binary | breast_cancer | ensured | 0.5000 | 0.6316 | 0.9722 | 0.0161 |
| binary | breast_cancer | ensured | 0.7000 | 0.9474 | 0.9722 | 0.0161 |
| binary | breast_cancer | ensured | 1.0000 | 0.9737 | 0.9640 | 0.0078 |
| binary | colic | default | 0.3000 | 0.6944 | 0.9000 | 0.0667 |
| binary | colic | default | 0.5000 | 0.6944 | 0.9000 | 0.0667 |
| binary | colic | default | 0.7000 | 0.6944 | 0.9000 | 0.0667 |
| binary | colic | default | 1.0000 | 0.6944 | 0.9000 | 0.0667 |

_Showing first 12 of 368 rows._
