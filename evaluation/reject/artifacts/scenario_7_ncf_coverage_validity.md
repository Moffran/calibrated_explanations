# Scenario 7 - NCF coverage validity sweep (supplementary)

Rows: 2080

## Key findings

- SUPPLEMENTARY empirical diagnostic; not a standalone proof of conformal validity.
- Coverage is measured from prediction sets stored in result.metadata['prediction_set'].
- Observed row-level coverage violations: 841/2080.
- Observed row-level structural violations: 100/2080.
- The dominant tendency is singleton collapse on harder datasets: when accept_rate/singleton_rate is high, prediction-set coverage tracks ordinary baseline accuracy rather than gaining much from ambiguity sets.
- High-accept structural rows (accept_rate >= 0.95): 28/100.
- Collapsed by (dataset, seed, ncf, epsilon), structural violations are 38/520; this avoids over-reading repeated w rows for default NCF.
- structural_violation means the Clopper-Pearson upper bound is below 1-epsilon in this finite test batch; it is strong diagnostic evidence, not a separate theorem.

## Outcome snapshot

- **datasets**: 26
- **seeds**: 5
- **rows**: 2080
- **coverage_defined_count**: 2080
- **coverage_undefined_count**: 0
- **total_violations**: 841
- **structural_violations**: 100
- **independent_condition_groups**: 520
- **independent_total_violations**: 277
- **independent_structural_violations**: 38
- **high_accept_structural_violations**: 28
- **mean_by_ncf_epsilon**: [{'ncf': 'default', 'epsilon': 0.05, 'mean_coverage': 0.9496592720489462, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.5476097607730878, 'mean_singleton_rate': 0.5476097607730878, 'structural_violations': 24}, {'ncf': 'default', 'epsilon': 0.1, 'mean_coverage': 0.8980092196086135, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.7126330193726184, 'mean_singleton_rate': 0.7126330193726184, 'structural_violations': 28}, {'ncf': 'ensured', 'epsilon': 0.05, 'mean_coverage': 0.9554412237323779, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.35266214164592086, 'mean_singleton_rate': 0.35266214164592086, 'structural_violations': 23}, {'ncf': 'ensured', 'epsilon': 0.1, 'mean_coverage': 0.9088091546268486, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.4566215221436756, 'mean_singleton_rate': 0.4566215221436756, 'structural_violations': 25}]
- **top_structural_datasets**: [{'dataset': 'je4243', 'structural': 19, 'mean_coverage': 0.884931506849315, 'mean_accept_rate': 0.3184931506849315, 'mean_singleton_rate': 0.3184931506849315}, {'dataset': 'heartS', 'structural': 18, 'mean_coverage': 0.8935185185185185, 'mean_accept_rate': 0.5474537037037037, 'mean_singleton_rate': 0.5474537037037037}, {'dataset': 'creditA', 'structural': 18, 'mean_coverage': 0.911322463768116, 'mean_accept_rate': 0.7323369565217391, 'mean_singleton_rate': 0.7323369565217391}, {'dataset': 'liver', 'structural': 15, 'mean_coverage': 0.9233695652173914, 'mean_accept_rate': 0.32753623188405795, 'mean_singleton_rate': 0.32753623188405795}, {'dataset': 'kc3', 'structural': 11, 'mean_coverage': 0.9225, 'mean_accept_rate': 0.5630769230769231, 'mean_singleton_rate': 0.5630769230769231}, {'dataset': 'colic', 'structural': 8, 'mean_coverage': 0.9069444444444444, 'mean_accept_rate': 0.6546875, 'mean_singleton_rate': 0.6546875}, {'dataset': 'pc1req', 'structural': 7, 'mean_coverage': 0.9, 'mean_accept_rate': 0.24761904761904763, 'mean_singleton_rate': 0.24761904761904763}, {'dataset': 'spectf', 'structural': 1, 'mean_coverage': 0.924074074074074, 'mean_accept_rate': 0.4810185185185185, 'mean_singleton_rate': 0.4810185185185185}]
- **structural_violations_by_ncf_w**: {"('default', 0.3)": 13, "('default', 0.5)": 13, "('default', 0.7)": 13, "('default', 1.0)": 13, "('ensured', 0.3)": 7, "('ensured', 0.5)": 14, "('ensured', 0.7)": 14, "('ensured', 1.0)": 13}
- **violations_by_ncf_w**: {"('default', 0.3)": 115, "('default', 0.5)": 115, "('default', 0.7)": 115, "('default', 1.0)": 115, "('ensured', 0.3)": 75, "('ensured', 0.5)": 85, "('ensured', 0.7)": 106, "('ensured', 1.0)": 115}

## Coverage by NCF and epsilon

| ncf | epsilon | mean_coverage | violation_rate | structural_violation_rate | mean_accept_rate |
|---|---|---|---|---|---|
| default | 0.0500 | 0.9497 | 0.4154 | 0.0462 | 0.5476 |
| default | 0.1000 | 0.8980 | 0.4692 | 0.0538 | 0.7126 |
| ensured | 0.0500 | 0.9554 | 0.3442 | 0.0442 | 0.3527 |
| ensured | 0.1000 | 0.9088 | 0.3885 | 0.0481 | 0.4566 |

## All datasets — structural violations

| dataset | structural_violations | violations | mean_coverage | mean_accept_rate |
|---|---|---|---|---|
| je4243 | 19 | 57 | 0.8849 | 0.3185 |
| heartS | 18 | 49 | 0.8935 | 0.5475 |
| creditA | 18 | 49 | 0.9113 | 0.7323 |
| liver | 15 | 29 | 0.9234 | 0.3275 |
| kc3 | 11 | 34 | 0.9225 | 0.5631 |
| colic | 8 | 49 | 0.9069 | 0.6547 |
| pc1req | 7 | 37 | 0.9000 | 0.2476 |
| spectf | 1 | 29 | 0.9241 | 0.4810 |
| ttt | 1 | 29 | 0.9332 | 0.7680 |
| iono | 1 | 28 | 0.9363 | 0.7332 |
| sonar | 1 | 30 | 0.9351 | 0.6554 |
| hepati | 0 | 5 | 0.9589 | 0.4617 |
| heartC | 0 | 27 | 0.9408 | 0.4793 |
| heartH | 0 | 44 | 0.9261 | 0.5186 |
| diabetes | 0 | 12 | 0.9446 | 0.4375 |
| german | 0 | 23 | 0.9385 | 0.2416 |
| breast_cancer | 0 | 25 | 0.9441 | 0.8656 |
| haberman | 0 | 39 | 0.9276 | 0.2906 |
| kc2 | 0 | 19 | 0.9441 | 0.4127 |
| kc1 | 0 | 16 | 0.9446 | 0.2443 |
| je4042 | 0 | 35 | 0.9299 | 0.3375 |
| pc4 | 0 | 44 | 0.9282 | 0.7042 |
| spect | 0 | 19 | 0.9384 | 0.4651 |
| transfusion | 0 | 53 | 0.9165 | 0.4460 |
| vote | 0 | 37 | 0.9329 | 0.7115 |
| wbc | 0 | 23 | 0.9413 | 0.8070 |
