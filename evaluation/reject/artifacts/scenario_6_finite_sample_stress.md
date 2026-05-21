# Scenario 6 — Finite-sample stress tests

Rows: 22

## Key findings

- Stress tests focus on finite-sample behavior and extreme confidence settings.
- The suite reports coverage violations empirically rather than claiming new guarantees.
- violation is computed from actual coverage, not hard-coded.
- extreme_confidence probe uses the same violation logic as small_calibration.

## Outcome snapshot

- **rows**: 22
- **violations**: 11
- **max_reject_rate**: 1.0000
- **small_cal_violations**: 10
- **extreme_conf_violations**: 1

## Result table

| dataset | probe | n_cal | epsilon | coverage | reject_rate | error_rate | violation | matched_count |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | small_calibration | 10 | 0.0500 | 1.0000 | 1.0000 | 0.0000 | no | 0 |
| breast_cancer | small_calibration | 10 | 0.1000 | 0.7544 | 0.2456 | 0.0000 | yes | 86 |
| breast_cancer | small_calibration | 10 | 0.2500 | 0.6316 | 0.3684 | 0.0000 | yes | 72 |
| breast_cancer | small_calibration | 20 | 0.0500 | 0.9211 | 0.0439 | 0.0064 | yes | 109 |
| breast_cancer | small_calibration | 20 | 0.1000 | 0.8421 | 0.1404 | 0.0000 | yes | 98 |
| breast_cancer | small_calibration | 20 | 0.2500 | 0.8333 | 0.1667 | 0.1000 | no | 95 |
| breast_cancer | small_calibration | 50 | 0.0500 | 0.9561 | 0.0000 | 0.0500 | no | 114 |
| breast_cancer | small_calibration | 50 | 0.1000 | 0.8947 | 0.0965 | 0.0039 | yes | 103 |
| breast_cancer | small_calibration | 50 | 0.2500 | 0.7719 | 0.2281 | 0.0284 | no | 88 |
| breast_cancer | extreme_confidence | 114 | 0.0100 | 0.9737 | 0.0614 | 0.0107 | yes | 107 |
| breast_cancer | extreme_confidence | 114 | 0.0050 | 1.0000 | 1.0000 | 0.0000 | no | 0 |
| colic | small_calibration | 10 | 0.0500 | 1.0000 | 1.0000 | 0.0000 | no | 0 |

_Showing first 12 of 22 rows._
