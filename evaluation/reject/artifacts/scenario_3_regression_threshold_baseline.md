# Scenario 3 - Thresholded regression binary-event reject validity

Rows: 264

## Key findings

- Thresholded regression reject is evaluated as binary conformal classification over event labels.
- Scalar event: y <= threshold. Interval event: low < y <= high.
- Coverage is empirical event-label coverage from conformal prediction sets over {0, 1}.
- Singleton precision, recall, and empirical singleton error are derived from those same event labels.
- No interval-width selection, interval coverage, or accepted-interval-width diagnostic is part of Scenario 3.
- Observed event-coverage violations: 69/264.
- Structural violations (CI upper bound < confidence): 8/264.

## Outcome snapshot

- **datasets**: 22
- **rows**: 264
- **mean_event_coverage**: 0.9384
- **mean_reject_rate**: 0.2189
- **mean_empirical_singleton_error**: 0.0811
- **coverage_violations**: 69
- **structural_violations**: 8

## By threshold type

| threshold_type | n_rows | mean_event_prevalence | mean_event_coverage | violations | structural_violations | mean_reject_rate | mean_singleton_precision | mean_singleton_recall |
|---|---|---|---|---|---|---|---|---|
| interval | 44 | 0.4840 | 0.9412 | 14 | 2 | 0.4147 | 0.8554 | 0.5313 |
| scalar | 220 | 0.3410 | 0.9378 | 55 | 6 | 0.1798 | 0.9316 | 0.7763 |

## Per-dataset event validity

| dataset | violations | structural_violations | mean_event_coverage | mean_reject_rate | mean_singleton_precision | mean_singleton_recall |
|---|---|---|---|---|---|---|
| comp | 6 | 3 | 0.9265 | 0.0831 | 0.9396 | 0.8616 |
| wizmir | 8 | 1 | 0.9181 | 0.0626 | 0.9784 | 0.9164 |
| bank8nm | 3 | 1 | 0.9290 | 0.1636 | 0.9154 | 0.7658 |
| communities | 5 | 1 | 0.9265 | 0.3444 | 0.8762 | 0.5821 |
| friedm | 6 | 1 | 0.9188 | 0.1049 | 0.9247 | 0.8278 |
| heating | 3 | 1 | 0.9513 | 0.0411 | 0.9912 | 0.9502 |
| mg | 5 | 0 | 0.9398 | 0.0809 | 0.9438 | 0.8673 |
| wineWhite | 3 | 0 | 0.9391 | 0.3845 | 0.8864 | 0.5546 |
| wineRed | 1 | 0 | 0.9409 | 0.4727 | 0.8465 | 0.4682 |
| treasury | 1 | 0 | 0.9575 | 0.0413 | 0.9910 | 0.9500 |
| stock | 3 | 0 | 0.9386 | 0.0456 | 0.9768 | 0.9320 |
| quakes | 0 | 0 | 0.9564 | 0.8805 | 0.6166 | 0.0759 |
| plastic | 0 | 0 | 0.9475 | 0.3402 | 0.9119 | 0.6073 |
| abalone | 2 | 0 | 0.9358 | 0.3254 | 0.8933 | 0.6104 |
| kin8nm | 7 | 0 | 0.9258 | 0.2518 | 0.9012 | 0.6766 |
| kin8fm | 1 | 0 | 0.9352 | 0.0711 | 0.9484 | 0.8811 |
| bank8fm | 4 | 0 | 0.9307 | 0.0490 | 0.9532 | 0.9063 |
| diabetes_reg | 2 | 0 | 0.9485 | 0.5861 | 0.8704 | 0.3624 |
| cooling | 3 | 0 | 0.9508 | 0.0433 | 0.9836 | 0.9410 |
| concreate | 2 | 0 | 0.9438 | 0.1040 | 0.9452 | 0.8471 |
| boston | 4 | 0 | 0.9314 | 0.1699 | 0.9303 | 0.7712 |
| housing | 0 | 0 | 0.9525 | 0.1704 | 0.9907 | 0.8250 |
