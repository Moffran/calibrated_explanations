# Scenario I — Explanation quality

Rows: 46

## Key findings

- Explanation quality is evaluated only empirically; no conformal claim is attached to these metrics.
- Feature-weight stability is computed from per-instance explanation weight vectors.

## Outcome snapshot

- **datasets**: 46
- **mean_accuracy_delta**: 0.0933
- **mean_ece_delta**: -0.0594

## Result table

| dataset | n_test | confidence | baseline_accuracy | accepted_accuracy | accuracy_delta | baseline_ece | accepted_ece | ece_delta | reject_rate | weight_variance_all | weight_variance_accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 114 | 0.9500 | 0.9561 | 0.9640 | 0.0078 | 0.1851 | 0.1874 | -0.0023 | 0.0263 | 0.0029 | 0.0023 |
| colic | 72 | 0.9500 | 0.8333 | 0.9000 | 0.0667 | 0.2388 | 0.2880 | -0.0492 | 0.3056 | 0.0059 | 0.0034 |
| creditA | 138 | 0.9500 | 0.8478 | 0.8720 | 0.0242 | 0.3082 | 0.3295 | -0.0213 | 0.0942 | 0.0088 | 0.0082 |
| diabetes | 154 | 0.9500 | 0.7403 | 0.9000 | 0.1597 | 0.1184 | 0.1107 | 0.0077 | 0.5455 | 0.0170 | 0.0171 |
| german | 191 | 0.9500 | 0.6492 | 0.8400 | 0.1908 | 0.4400 | 0.7415 | -0.3015 | 0.7382 | 0.0029 | 0.0033 |
| haberman | 57 | 0.9500 | 0.6667 | 0.8125 | 0.1458 | 0.4504 | 0.6234 | -0.1730 | 0.7193 | 0.0225 | 0.0124 |
| heartC | 61 | 0.9500 | 0.7705 | 0.9583 | 0.1878 | 0.3542 | 0.4597 | -0.1055 | 0.6066 | 0.0054 | 0.0039 |
| heartH | 59 | 0.9500 | 0.7966 | 0.8571 | 0.0605 | 0.4641 | 0.5267 | -0.0626 | 0.1695 | 0.0102 | 0.0086 |
| heartS | 54 | 0.9500 | 0.7407 | 0.7692 | 0.0285 | 0.2648 | 0.3250 | -0.0602 | 0.2778 | 0.0196 | 0.0157 |
| hepati | 31 | 0.9500 | 0.9032 | 1.0000 | 0.0968 | 0.6172 | 0.8500 | -0.2328 | 0.6774 | 0.0065 | 0.0005 |
| iono | 70 | 0.9500 | 0.9571 | 1.0000 | 0.0429 | 0.5501 | 0.5342 | 0.0159 | 0.1571 | 0.0189 | 0.0185 |
| je4042 | 54 | 0.9500 | 0.7407 | 0.8000 | 0.0593 | 0.4049 | 0.5569 | -0.1520 | 0.4444 | 0.0273 | 0.0416 |

_Showing first 12 of 46 rows._
