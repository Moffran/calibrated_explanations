# Scenario 5 — Explanation quality on accepted instances

Rows: 46

## Key findings

- Explanation quality is evaluated only empirically; no conformal claim is attached.
- Regime boundaries: low (<=15%), moderate (15%–40%), high (>40%) reject rate.
- Paper finding: accuracy_delta is most reliable in the low regime.
- mean_feature_weight_variance is not included — it is not a paper metric.

## Outcome snapshot

- **datasets**: 46
- **mean_accuracy_delta**: 0.0934
- **mean_ece_delta**: -0.0593
- **regime_summary**: {'low': {'n': 12, 'mean_accuracy_delta': 0.021164084270489786, 'mean_ece_delta': -0.005245520481660015, 'mean_reject_rate': 0.05843210797399068}, 'moderate': {'n': 9, 'mean_accuracy_delta': 0.061159300207046456, 'mean_ece_delta': -0.02685838297914815, 'mean_reject_rate': 0.27568393232999416}, 'high': {'n': 25, 'mean_accuracy_delta': 0.14361366739484643, 'mean_ece_delta': -0.10028716516834928, 'mean_reject_rate': 0.6550079657533209}}

## Result table

| dataset | task_type | n_test | confidence | reject_rate | regime | baseline_accuracy | accepted_accuracy | accuracy_delta | baseline_ece | accepted_ece | ece_delta | singleton_precision | singleton_recall | singleton_correct_count | singleton_count | singleton_precision_recall_defined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | binary | 114 | 0.9500 | 0.0263 | low | 0.9561 | 0.9640 | 0.0078 | 0.1851 | 0.1874 | -0.0023 | 0.9640 | 0.9386 | 107 | 111 | yes |
| colic | binary | 72 | 0.9500 | 0.3056 | moderate | 0.8333 | 0.9000 | 0.0667 | 0.2388 | 0.2880 | -0.0492 | 0.9000 | 0.6250 | 45 | 50 | yes |
| creditA | binary | 138 | 0.9500 | 0.0942 | low | 0.8478 | 0.8720 | 0.0242 | 0.3082 | 0.3295 | -0.0213 | 0.8720 | 0.7899 | 109 | 125 | yes |
| diabetes | binary | 154 | 0.9500 | 0.5455 | high | 0.7403 | 0.9000 | 0.1597 | 0.1184 | 0.1107 | 0.0077 | 0.9000 | 0.4091 | 63 | 70 | yes |
| german | binary | 191 | 0.9500 | 0.7382 | high | 0.6492 | 0.8400 | 0.1908 | 0.4400 | 0.7415 | -0.3015 | 0.8400 | 0.2199 | 42 | 50 | yes |
| haberman | binary | 57 | 0.9500 | 0.7193 | high | 0.6667 | 0.8125 | 0.1458 | 0.4504 | 0.6234 | -0.1730 | 0.8125 | 0.2281 | 13 | 16 | yes |
| heartC | binary | 61 | 0.9500 | 0.6066 | high | 0.7705 | 0.9583 | 0.1878 | 0.3542 | 0.4597 | -0.1055 | 0.9583 | 0.3770 | 23 | 24 | yes |
| heartH | binary | 59 | 0.9500 | 0.1695 | moderate | 0.7966 | 0.8571 | 0.0605 | 0.4641 | 0.5267 | -0.0626 | 0.8571 | 0.7119 | 42 | 49 | yes |
| heartS | binary | 54 | 0.9500 | 0.2778 | moderate | 0.7407 | 0.7692 | 0.0285 | 0.2648 | 0.3250 | -0.0602 | 0.7692 | 0.5556 | 30 | 39 | yes |
| hepati | binary | 31 | 0.9500 | 0.6774 | high | 0.9032 | 1.0000 | 0.0968 | 0.6172 | 0.8500 | -0.2328 | 1.0000 | 0.3226 | 10 | 10 | yes |
| iono | binary | 70 | 0.9500 | 0.1571 | moderate | 0.9571 | 1.0000 | 0.0429 | 0.5501 | 0.5342 | 0.0159 | 1.0000 | 0.8429 | 59 | 59 | yes |
| je4042 | binary | 54 | 0.9500 | 0.4444 | high | 0.7407 | 0.8000 | 0.0593 | 0.4049 | 0.5569 | -0.1520 | 0.8000 | 0.4444 | 24 | 30 | yes |

_Showing first 12 of 46 rows._
