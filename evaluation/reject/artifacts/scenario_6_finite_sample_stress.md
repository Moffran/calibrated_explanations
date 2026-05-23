# Scenario 6 — Finite-sample stress tests

Rows: 51

## Key findings

- Stress tests focus on finite-sample behavior and extreme confidence settings.
- The suite reports coverage violations empirically rather than claiming new guarantees.
- violation is computed from actual coverage, not hard-coded.
- extreme_confidence probe uses the same violation logic as small_calibration.

## Outcome snapshot

- **rows**: 51
- **violations**: 28
- **max_reject_rate**: 1.0000
- **small_cal_violations**: 27
- **extreme_conf_violations**: 1

## Result table

| dataset | probe | n_cal | epsilon | coverage | reject_rate | error_rate | violation | matched_count | singleton_precision | singleton_recall | singleton_correct_count | singleton_count | singleton_precision_recall_defined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | small_calibration | 10 | 0.0500 | 1.0000 | 1.0000 | 0.0000 | no | 0 | nan | 0.0000 | 0 | 0 | no |
| breast_cancer | small_calibration | 10 | 0.1000 | 1.0000 | 0.2368 | 0.1310 | no | 87 | 1.0000 | 0.7632 | 87 | 87 | yes |
| breast_cancer | small_calibration | 10 | 0.2500 | 0.7632 | 0.2368 | 0.0172 | no | 87 | 1.0000 | 0.7632 | 87 | 87 | yes |
| breast_cancer | small_calibration | 20 | 0.0500 | 0.9737 | 0.0526 | 0.0528 | no | 108 | 0.9722 | 0.9211 | 105 | 108 | yes |
| breast_cancer | small_calibration | 20 | 0.1000 | 0.9211 | 0.0526 | 0.0500 | no | 108 | 0.9722 | 0.9211 | 105 | 108 | yes |
| breast_cancer | small_calibration | 20 | 0.2500 | 0.8070 | 0.1930 | 0.0707 | no | 92 | 1.0000 | 0.8070 | 92 | 92 | yes |
| breast_cancer | small_calibration | 50 | 0.0500 | 0.9561 | 0.0088 | 0.0504 | no | 113 | 0.9558 | 0.9474 | 108 | 113 | yes |
| breast_cancer | small_calibration | 50 | 0.1000 | 0.8860 | 0.1053 | 0.0000 | yes | 102 | 0.9902 | 0.8860 | 101 | 102 | yes |
| breast_cancer | small_calibration | 50 | 0.2500 | 0.7018 | 0.2982 | 0.0000 | yes | 80 | 1.0000 | 0.7018 | 80 | 80 | yes |
| breast_cancer | small_calibration | 100 | 0.0500 | 0.9386 | 0.0175 | 0.0330 | yes | 112 | 0.9554 | 0.9386 | 107 | 112 | yes |
| breast_cancer | small_calibration | 100 | 0.1000 | 0.9211 | 0.0439 | 0.0587 | no | 109 | 0.9633 | 0.9211 | 105 | 109 | yes |
| breast_cancer | small_calibration | 100 | 0.2500 | 0.7632 | 0.2368 | 0.0172 | no | 87 | 1.0000 | 0.7632 | 87 | 87 | yes |

_Showing first 12 of 51 rows._
