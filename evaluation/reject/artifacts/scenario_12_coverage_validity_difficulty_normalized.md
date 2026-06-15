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

## Coverage validity by arm and epsilon

| arm | epsilon | violations | structural_violations | violation_rate | structural_violation_rate | mean_coverage |
|---|---|---|---|---|---|---|
| A | 0.0500 | 54 | 6 | 0.4154 | 0.0462 | 0.9497 |
| A | 0.1000 | 61 | 7 | 0.4692 | 0.0538 | 0.8980 |
| C | 0.0500 | 66 | 12 | 0.5077 | 0.0923 | 0.9428 |
| C | 0.1000 | 69 | 11 | 0.5308 | 0.0846 | 0.8865 |

## Structural violations by arm and dataset (all datasets)

| arm | dataset | structural_violations | violations | mean_coverage |
|---|---|---|---|---|
| A | creditA | 3 | 7 | 0.9036 |
| A | heartS | 3 | 7 | 0.8778 |
| A | je4243 | 2 | 8 | 0.8726 |
| A | liver | 2 | 4 | 0.9159 |
| A | colic | 1 | 7 | 0.9028 |
| A | kc3 | 1 | 4 | 0.9292 |
| A | pc1req | 1 | 6 | 0.8762 |
| A | breast_cancer | 0 | 4 | 0.9325 |
| A | diabetes | 0 | 1 | 0.9455 |
| A | german | 0 | 3 | 0.9377 |
| A | haberman | 0 | 5 | 0.9246 |
| A | heartC | 0 | 4 | 0.9377 |
| A | heartH | 0 | 7 | 0.9102 |
| A | hepati | 0 | 0 | 0.9645 |
| A | iono | 0 | 4 | 0.9314 |
| A | je4042 | 0 | 4 | 0.9333 |
| A | kc1 | 0 | 2 | 0.9469 |
| A | kc2 | 0 | 2 | 0.9459 |
| A | pc4 | 0 | 6 | 0.9238 |
| A | sonar | 0 | 4 | 0.9357 |
| A | spect | 0 | 3 | 0.9341 |
| A | spectf | 0 | 3 | 0.9296 |
| A | transfusion | 0 | 7 | 0.9129 |
| A | ttt | 0 | 4 | 0.9323 |
| A | vote | 0 | 5 | 0.9308 |
| A | wbc | 0 | 4 | 0.9323 |
| C | creditA | 3 | 9 | 0.8891 |
| C | haberman | 2 | 5 | 0.9193 |
| C | heartS | 2 | 6 | 0.8815 |
| C | je4243 | 2 | 5 | 0.8945 |
| C | liver | 2 | 5 | 0.9203 |
| C | pc1req | 2 | 7 | 0.8619 |
| C | spectf | 2 | 5 | 0.9093 |
| C | transfusion | 2 | 6 | 0.9040 |
| C | colic | 1 | 7 | 0.9000 |
| C | diabetes | 1 | 3 | 0.9260 |
| C | heartC | 1 | 4 | 0.9115 |
| C | iono | 1 | 7 | 0.9086 |
| C | je4042 | 1 | 5 | 0.9111 |
| C | kc2 | 1 | 5 | 0.9243 |
| C | breast_cancer | 0 | 5 | 0.9254 |
| C | german | 0 | 5 | 0.9241 |
| C | heartH | 0 | 7 | 0.9102 |
| C | hepati | 0 | 3 | 0.9452 |
| C | kc1 | 0 | 4 | 0.9285 |
| C | kc3 | 0 | 3 | 0.9338 |
| C | pc4 | 0 | 5 | 0.9204 |
| C | sonar | 0 | 3 | 0.9452 |
| C | spect | 0 | 6 | 0.9182 |
| C | ttt | 0 | 4 | 0.9255 |
| C | vote | 0 | 6 | 0.9221 |
| C | wbc | 0 | 5 | 0.9204 |
