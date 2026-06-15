# Scenario 1 — Binary marginal coverage sweep

Rows: 390

## Key findings

- Coverage is reported as standard label-set coverage from the conformal prediction sets.
- Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.
- Observed coverage violations: 129/390 (0.3308).
- Structural violations (CI upper bound < 1-epsilon, cannot be attributed to finite-sample noise): 15/390.
- Uses 5-seed grid matching Scenarios 8-12 for stable violation counts.

## Outcome snapshot

- **datasets**: 26
- **seeds**: 5
- **rows**: 390
- **violation_rate**: 0.3308
- **structural_violations**: 15
- **mean_coverage**: 0.9482

## Coverage by epsilon

| epsilon | n_rows | violations | structural_violations | violation_rate | mean_coverage | mean_reject_rate |
|---|---|---|---|---|---|---|
| 0.0100 | 130.0000 | 14.0000 | 2.0000 | 0.1077 | 0.9971 | 0.8678 |
| 0.0500 | 130.0000 | 54.0000 | 6.0000 | 0.4154 | 0.9497 | 0.4524 |
| 0.1000 | 130.0000 | 61.0000 | 7.0000 | 0.4692 | 0.8980 | 0.2874 |

## All datasets — structural violations

| dataset | structural_violations | violations | mean_coverage | mean_reject_rate |
|---|---|---|---|---|
| creditA | 4 | 9 | 0.9309 | 0.3614 |
| heartS | 3 | 7 | 0.9185 | 0.5642 |
| liver | 2 | 4 | 0.9440 | 0.7478 |
| je4243 | 2 | 8 | 0.9151 | 0.7324 |
| colic | 1 | 7 | 0.9352 | 0.5093 |
| breast_cancer | 1 | 8 | 0.9474 | 0.0520 |
| pc1req | 1 | 6 | 0.9175 | 0.8032 |
| kc3 | 1 | 4 | 0.9528 | 0.5026 |
| german | 0 | 4 | 0.9567 | 0.7749 |
| diabetes | 0 | 1 | 0.9628 | 0.6420 |
| hepati | 0 | 0 | 0.9763 | 0.5849 |
| heartH | 0 | 7 | 0.9401 | 0.5153 |
| heartC | 0 | 4 | 0.9585 | 0.5956 |
| haberman | 0 | 5 | 0.9497 | 0.7404 |
| kc1 | 0 | 2 | 0.9643 | 0.7662 |
| je4042 | 0 | 4 | 0.9556 | 0.7407 |
| iono | 0 | 4 | 0.9543 | 0.4048 |
| kc2 | 0 | 2 | 0.9640 | 0.6306 |
| pc4 | 0 | 8 | 0.9470 | 0.1571 |
| sonar | 0 | 4 | 0.9571 | 0.4762 |
| spect | 0 | 3 | 0.9561 | 0.5955 |
| spectf | 0 | 3 | 0.9531 | 0.5778 |
| transfusion | 0 | 9 | 0.9393 | 0.6634 |
| ttt | 0 | 4 | 0.9542 | 0.0729 |
| vote | 0 | 8 | 0.9494 | 0.3455 |
| wbc | 0 | 4 | 0.9548 | 0.3756 |
