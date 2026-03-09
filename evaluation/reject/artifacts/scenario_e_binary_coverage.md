# Scenario E — Binary coverage sweep

Rows: 78

## Key findings

- Coverage is reported as standard label-set coverage from the conformal prediction sets.
- Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.
- Observed coverage violations: 31/78 (0.3974).

## Outcome snapshot

- **datasets**: 26
- **rows**: 78
- **violation_rate**: 0.3974
- **mean_coverage**: 0.9424

## Result table

| dataset | epsilon | n_cal | n_test | coverage | lower_ci | upper_ci | violation | reject_rate | accepted_accuracy_empirical |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.0100 | 114 | 114 | 0.9737 | 0.9250 | 0.9945 | yes | 0.0526 | 0.9722 |
| breast_cancer | 0.0500 | 114 | 114 | 0.9386 | 0.8776 | 0.9750 | yes | 0.0263 | 0.9640 |
| breast_cancer | 0.1000 | 114 | 114 | 0.9211 | 0.8554 | 0.9633 | no | 0.0526 | 0.9722 |
| colic | 0.0100 | 72 | 72 | 1.0000 | 0.9501 | 1.0000 | no | 1.0000 | nan |
| colic | 0.0500 | 72 | 72 | 0.9306 | 0.8453 | 0.9771 | yes | 0.3056 | 0.9000 |
| colic | 0.1000 | 72 | 72 | 0.8056 | 0.6953 | 0.8894 | yes | 0.0000 | 0.8333 |
| creditA | 0.0100 | 138 | 138 | 0.9928 | 0.9603 | 0.9998 | no | 0.9420 | 0.8750 |
| creditA | 0.0500 | 138 | 138 | 0.8841 | 0.8186 | 0.9323 | yes | 0.0942 | 0.8720 |
| creditA | 0.1000 | 138 | 138 | 0.8261 | 0.7524 | 0.8853 | yes | 0.0362 | 0.8571 |
| diabetes | 0.0100 | 154 | 154 | 0.9935 | 0.9644 | 0.9998 | no | 0.8377 | 0.9600 |
| diabetes | 0.0500 | 154 | 154 | 0.9545 | 0.9086 | 0.9815 | no | 0.5455 | 0.9000 |
| diabetes | 0.1000 | 154 | 154 | 0.9091 | 0.8522 | 0.9494 | no | 0.3701 | 0.8557 |

_Showing first 12 of 78 rows._
