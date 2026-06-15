# Scenario 13 — n_cal sweep: arm A vs arm C structural violations

Rows: 1960

## Key findings

- Sweeps n_cal in {50, 100, 200, 400} to test variance-inflation hypothesis for arm C.
- Hypothesis: arm C structural violation rate decreases as n_cal grows (finite-sample effect).
- Counter-evidence: if rate does not decrease, a genuine exchangeability violation is indicated.
- Arm A structural violations: 51/1960.
- Arm C structural violations: 82/1960.
- Hypothesis (arm C violations decrease with n_cal): SUPPORTED.

## Outcome snapshot

- **rows**: 1960
- **datasets**: 26
- **seeds**: 5
- **n_cal_targets**: [50, 100, 200, 400]
- **arm_A_structural_violations**: 51
- **arm_A_total_rows**: 1960
- **arm_C_structural_violations**: 82
- **arm_C_total_rows**: 1960
- **hypothesis_supported**: yes
- **sv_by_arm_ncal**: [{'arm': 'A', 'n_cal_target': 50, 'structural_violation_rate': 0.07692307692307693, 'structural_violations': 20, 'total_rows': 260, 'mean_coverage': 0.9263892228718063}, {'arm': 'A', 'n_cal_target': 100, 'structural_violation_rate': 0.06923076923076923, 'structural_violations': 18, 'total_rows': 260, 'mean_coverage': 0.9280714491648994}, {'arm': 'A', 'n_cal_target': 200, 'structural_violation_rate': 0.041666666666666664, 'structural_violations': 10, 'total_rows': 240, 'mean_coverage': 0.931443033247339}, {'arm': 'A', 'n_cal_target': 400, 'structural_violation_rate': 0.013636363636363636, 'structural_violations': 3, 'total_rows': 220, 'mean_coverage': 0.9425159734903532}, {'arm': 'C', 'n_cal_target': 50, 'structural_violation_rate': 0.11538461538461539, 'structural_violations': 30, 'total_rows': 260, 'mean_coverage': 0.9138973962159901}, {'arm': 'C', 'n_cal_target': 100, 'structural_violation_rate': 0.11538461538461539, 'structural_violations': 30, 'total_rows': 260, 'mean_coverage': 0.9126868078157021}, {'arm': 'C', 'n_cal_target': 200, 'structural_violation_rate': 0.06666666666666667, 'structural_violations': 16, 'total_rows': 240, 'mean_coverage': 0.9166840625664082}, {'arm': 'C', 'n_cal_target': 400, 'structural_violation_rate': 0.02727272727272727, 'structural_violations': 6, 'total_rows': 220, 'mean_coverage': 0.9242094755605108}]

## Structural violation rate by arm and n_cal

| arm | n_cal_target | structural_violation_rate | structural_violations | total_rows | mean_coverage |
|---|---|---|---|---|---|
| A | 50 | 0.0769 | 20 | 260 | 0.9264 |
| A | 100 | 0.0692 | 18 | 260 | 0.9281 |
| A | 200 | 0.0417 | 10 | 240 | 0.9314 |
| A | 400 | 0.0136 | 3 | 220 | 0.9425 |
| C | 50 | 0.1154 | 30 | 260 | 0.9139 |
| C | 100 | 0.1154 | 30 | 260 | 0.9127 |
| C | 200 | 0.0667 | 16 | 240 | 0.9167 |
| C | 400 | 0.0273 | 6 | 220 | 0.9242 |

## Hypothesis verdict

| arm_C_hypothesis_supported | arm_A_total_sv | arm_C_total_sv |
|---|---|---|
| yes | 51 | 82 |
