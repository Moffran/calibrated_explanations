# Scenario 7 — NCF coverage validity sweep (supplementary)

Rows: 416

## Key findings

- SUPPLEMENTARY — may yield misleading results before the RT-2 fix.
- Empirical companion to Proposition 1: coverage >= 1-epsilon across (NCF, w, epsilon) grid.
- structural_violation: CI upper bound < 1-epsilon; cannot be attributed to finite-sample noise.

## Outcome snapshot

- **datasets**: 26
- **total_violations**: 0
- **structural_violations**: 0
- **violations_by_ncf_w**: {"('default', 0.3)": 0, "('default', 0.5)": 0, "('default', 0.7)": 0, "('default', 1.0)": 0, "('ensured', 0.3)": 0, "('ensured', 0.5)": 0, "('ensured', 0.7)": 0, "('ensured', 1.0)": 0}

## Result table

| dataset | ncf | w | epsilon | n_cal | n_test | coverage | lower_ci | upper_ci | violation | structural_violation | accept_rate | singleton_precision | singleton_recall | singleton_correct_count | singleton_count | singleton_precision_recall_defined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | default | 0.3000 | 0.0500 | 114 | 114 | nan | nan | nan | no | no | 0.9737 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 0.3000 | 0.1000 | 114 | 114 | nan | nan | nan | no | no | 0.9474 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 0.5000 | 0.0500 | 114 | 114 | nan | nan | nan | no | no | 0.9737 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 0.5000 | 0.1000 | 114 | 114 | nan | nan | nan | no | no | 0.9474 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 0.7000 | 0.0500 | 114 | 114 | nan | nan | nan | no | no | 0.9737 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 0.7000 | 0.1000 | 114 | 114 | nan | nan | nan | no | no | 0.9474 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 1.0000 | 0.0500 | 114 | 114 | nan | nan | nan | no | no | 0.9737 | nan | nan | 0 | 0 | no |
| breast_cancer | default | 1.0000 | 0.1000 | 114 | 114 | nan | nan | nan | no | no | 0.9474 | nan | nan | 0 | 0 | no |
| breast_cancer | ensured | 0.3000 | 0.0500 | 114 | 114 | nan | nan | nan | no | no | 0.5877 | nan | nan | 0 | 0 | no |
| breast_cancer | ensured | 0.3000 | 0.1000 | 114 | 114 | nan | nan | nan | no | no | 0.5877 | nan | nan | 0 | 0 | no |
| breast_cancer | ensured | 0.5000 | 0.0500 | 114 | 114 | nan | nan | nan | no | no | 0.6316 | nan | nan | 0 | 0 | no |
| breast_cancer | ensured | 0.5000 | 0.1000 | 114 | 114 | nan | nan | nan | no | no | 0.6316 | nan | nan | 0 | 0 | no |

_Showing first 12 of 416 rows._
