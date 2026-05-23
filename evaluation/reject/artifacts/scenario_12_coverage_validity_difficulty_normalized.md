# Scenario 12 — Coverage validity: arm A vs arm C

Rows: 520

## Key findings

- Mirrors Scenario 1 but runs arm A (builtin.default) and arm C (experimental.difficulty_normalized) side by side.
- Coverage is standard label-set coverage from conformal prediction sets.
- A structural violation means CI upper bound < 1-epsilon; finite-sample noise cannot explain the shortfall.
- Arm A violations: 115/260; structural: 13/260.
- Arm C violations: 135/260; structural: 23/260.
- Arm A mean coverage: 0.9238; Arm C mean coverage: 0.9146.

## Outcome snapshot

- **rows**: 520
- **datasets**: 26
- **seeds**: 5
- **arm_A_violations**: 115
- **arm_A_structural_violations**: 13
- **arm_A_total_rows**: 260
- **arm_A_mean_coverage**: 0.9238
- **arm_C_violations**: 135
- **arm_C_structural_violations**: 23
- **arm_C_total_rows**: 260
- **arm_C_mean_coverage**: 0.9146

## Result table

| arm | strategy | dataset | seed | epsilon | confidence | n_cal | n_test | coverage | lower_ci | upper_ci | violation | structural_violation | reject_rate | singleton_precision | singleton_recall | singleton_correct_count | singleton_count | singleton_precision_recall_defined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | builtin.default | breast_cancer | 42 | 0.0500 | 0.9500 | 114 | 114 | 0.9386 | 0.8776 | 0.9750 | yes | no | 0.0263 | 0.9640 | 0.9386 | 107 | 111 | yes |
| C | experimental.difficulty_normalized | breast_cancer | 42 | 0.0500 | 0.9500 | 114 | 114 | 0.9298 | 0.8664 | 0.9692 | yes | no | 0.0439 | 0.9725 | 0.9298 | 106 | 109 | yes |
| A | builtin.default | breast_cancer | 42 | 0.1000 | 0.9000 | 114 | 114 | 0.9211 | 0.8554 | 0.9633 | no | no | 0.0526 | 0.9722 | 0.9211 | 105 | 108 | yes |
| C | experimental.difficulty_normalized | breast_cancer | 42 | 0.1000 | 0.9000 | 114 | 114 | 0.9298 | 0.8664 | 0.9692 | no | no | 0.0439 | 0.9725 | 0.9298 | 106 | 109 | yes |
| A | builtin.default | breast_cancer | 43 | 0.0500 | 0.9500 | 114 | 114 | 0.9211 | 0.8554 | 0.9633 | yes | no | 0.0263 | 0.9459 | 0.9211 | 105 | 111 | yes |
| C | experimental.difficulty_normalized | breast_cancer | 43 | 0.0500 | 0.9500 | 114 | 114 | 0.9123 | 0.8446 | 0.9571 | yes | no | 0.0351 | 0.9455 | 0.9123 | 104 | 110 | yes |
| A | builtin.default | breast_cancer | 43 | 0.1000 | 0.9000 | 114 | 114 | 0.9035 | 0.8339 | 0.9508 | no | no | 0.0526 | 0.9537 | 0.9035 | 103 | 108 | yes |
| C | experimental.difficulty_normalized | breast_cancer | 43 | 0.1000 | 0.9000 | 114 | 114 | 0.9035 | 0.8339 | 0.9508 | no | no | 0.0526 | 0.9537 | 0.9035 | 103 | 108 | yes |
| A | builtin.default | breast_cancer | 44 | 0.0500 | 0.9500 | 114 | 114 | 0.9825 | 0.9381 | 0.9979 | no | no | 0.0000 | 0.9825 | 0.9825 | 112 | 114 | yes |
| C | experimental.difficulty_normalized | breast_cancer | 44 | 0.0500 | 0.9500 | 114 | 114 | 0.9737 | 0.9250 | 0.9945 | no | no | 0.0351 | 0.9818 | 0.9474 | 108 | 110 | yes |
| A | builtin.default | breast_cancer | 44 | 0.1000 | 0.9000 | 114 | 114 | 0.9386 | 0.8776 | 0.9750 | no | no | 0.0526 | 0.9907 | 0.9386 | 107 | 108 | yes |
| C | experimental.difficulty_normalized | breast_cancer | 44 | 0.1000 | 0.9000 | 114 | 114 | 0.9386 | 0.8776 | 0.9750 | no | no | 0.0526 | 0.9907 | 0.9386 | 107 | 108 | yes |

_Showing first 12 of 520 rows._
