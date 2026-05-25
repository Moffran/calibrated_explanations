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

## By reject-rate regime

| regime | n | mean_reject_rate | mean_accuracy_delta | mean_ece_delta |
|---|---|---|---|---|
| low | 12 | 0.0584 | 0.0212 | -0.0052 |
| moderate | 9 | 0.2757 | 0.0612 | -0.0269 |
| high | 25 | 0.6550 | 0.1436 | -0.1003 |

## Per-dataset results

| dataset | task_type | regime | reject_rate | accuracy_delta | ece_delta | accepted_accuracy | baseline_accuracy |
|---|---|---|---|---|---|---|---|
| pc1req | binary | high | 0.9048 | 0.3810 | -0.2710 | 1.0000 | 0.6190 |
| cmc | multiclass | high | 0.8169 | 0.2625 | 0.0240 | 0.7778 | 0.5153 |
| wineW | multiclass | high | 0.6398 | 0.2267 | -0.0271 | 0.8584 | 0.6316 |
| wineR | multiclass | high | 0.7125 | 0.1992 | -0.0157 | 0.8804 | 0.6813 |
| liver | binary | high | 0.5507 | 0.1931 | -0.0884 | 0.9032 | 0.7101 |
| german | binary | high | 0.7382 | 0.1908 | -0.3015 | 0.8400 | 0.6492 |
| spectf | binary | high | 0.4259 | 0.1900 | -0.2493 | 0.9677 | 0.7778 |
| heartC | binary | high | 0.6066 | 0.1878 | -0.1055 | 0.9583 | 0.7705 |
| whole | multiclass | high | 0.8864 | 0.1841 | 0.0356 | 0.9000 | 0.7159 |
| vehicle | multiclass | moderate | 0.3882 | 0.1648 | -0.0074 | 0.8942 | 0.7294 |
| steel | multiclass | high | 0.4165 | 0.1616 | 0.0012 | 0.9251 | 0.7635 |
| diabetes | binary | high | 0.5455 | 0.1597 | 0.0077 | 0.9000 | 0.7403 |
| haberman | binary | high | 0.7193 | 0.1458 | -0.1730 | 0.8125 | 0.6667 |
| yeast | multiclass | high | 0.7306 | 0.1367 | -0.0337 | 0.7125 | 0.5758 |
| glass | multiclass | high | 0.6047 | 0.1149 | -0.1157 | 0.8824 | 0.7674 |
| hepati | binary | high | 0.6774 | 0.0968 | -0.2328 | 1.0000 | 0.9032 |
| ecoli | multiclass | high | 0.4265 | 0.0920 | -0.0312 | 0.9744 | 0.8824 |
| balance | multiclass | low | 0.1360 | 0.0872 | 0.0024 | 0.9352 | 0.8480 |
| transfusion | binary | high | 0.5050 | 0.0871 | 0.0152 | 0.8000 | 0.7129 |
| wave | multiclass | moderate | 0.2770 | 0.0764 | -0.0093 | 0.9364 | 0.8600 |
| kc2 | binary | high | 0.4459 | 0.0725 | -0.0971 | 0.8293 | 0.7568 |
| colic | binary | moderate | 0.3056 | 0.0667 | -0.0492 | 0.9000 | 0.8333 |
| vote | binary | moderate | 0.3558 | 0.0643 | 0.0143 | 0.9104 | 0.8462 |
| pc4 | binary | low | 0.1413 | 0.0639 | -0.0439 | 0.9524 | 0.8885 |
| heartH | binary | moderate | 0.1695 | 0.0605 | -0.0626 | 0.8571 | 0.7966 |
| je4042 | binary | high | 0.4444 | 0.0593 | -0.1520 | 0.8000 | 0.7407 |
| kc1 | binary | high | 0.7238 | 0.0525 | -0.1737 | 0.8182 | 0.7657 |
| je4243 | binary | high | 0.7123 | 0.0502 | -0.1082 | 0.6667 | 0.6164 |
| iono | binary | moderate | 0.1571 | 0.0429 | 0.0159 | 1.0000 | 0.9571 |
| kc3 | binary | moderate | 0.1692 | 0.0427 | -0.0478 | 0.8889 | 0.8462 |
| spect | binary | high | 0.4318 | 0.0336 | -0.0363 | 0.9200 | 0.8864 |
| heartS | binary | moderate | 0.2778 | 0.0285 | -0.0602 | 0.7692 | 0.7407 |
| tae | multiclass | high | 0.7097 | 0.0251 | -0.1780 | 0.4444 | 0.4194 |
| creditA | binary | low | 0.0942 | 0.0242 | -0.0213 | 0.8720 | 0.8478 |
| user | multiclass | low | 0.1235 | 0.0160 | 0.0117 | 0.9296 | 0.9136 |
| ttt | binary | low | 0.0312 | 0.0156 | -0.0189 | 1.0000 | 0.9844 |
| cars | multiclass | low | 0.0462 | 0.0132 | 0.0088 | 0.9727 | 0.9595 |
| cool | multiclass | low | 0.0260 | 0.0123 | -0.0072 | 0.9733 | 0.9610 |
| image | multiclass | low | 0.0238 | 0.0104 | 0.0013 | 0.9823 | 0.9719 |
| breast_cancer | binary | low | 0.0263 | 0.0078 | -0.0023 | 0.9640 | 0.9561 |
| sonar | binary | moderate | 0.3810 | 0.0037 | -0.0355 | 0.8846 | 0.8810 |
| vowel | multiclass | low | 0.0202 | 0.0036 | 0.0013 | 0.9278 | 0.9242 |
| wbc | binary | low | 0.0000 | 0.0000 | 0.0000 | 0.9355 | 0.9355 |
| heat | multiclass | low | 0.0325 | -0.0002 | 0.0052 | 0.9933 | 0.9935 |
| iris | multiclass | high | 1.0000 | nan | nan | nan | 0.9333 |
| wine | multiclass | high | 1.0000 | nan | nan | nan | 1.0000 |
