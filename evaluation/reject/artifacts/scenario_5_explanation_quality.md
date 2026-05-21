# Scenario 5 — Explanation quality on accepted instances

Rows: 5

## Key findings

- Explanation quality is evaluated only empirically; no conformal claim is attached.
- Regime boundaries: low (<=15%), moderate (15%–40%), high (>40%) reject rate.
- Paper finding: accuracy_delta is most reliable in the low regime.
- mean_feature_weight_variance is not included — it is not a paper metric.

## Outcome snapshot

- **datasets**: 5
- **mean_accuracy_delta**: 0.0798
- **mean_ece_delta**: -0.0029
- **regime_summary**: {'low': {'n': 1, 'mean_accuracy_delta': -0.0006265664160400863, 'mean_ece_delta': 0.005283148347058175, 'mean_reject_rate': 0.017543859649122806}, 'moderate': {'n': 1, 'mean_accuracy_delta': 0.033333333333333326, 'mean_ece_delta': 0.0022613027939449304, 'mean_reject_rate': 0.16666666666666666}, 'high': {'n': 3, 'mean_accuracy_delta': 0.20673512533977656, 'mean_ece_delta': -0.016209138594082476, 'mean_reject_rate': 0.906926406926407}}

## Result table

| dataset | task_type | n_test | confidence | reject_rate | regime | baseline_accuracy | accepted_accuracy | accuracy_delta | baseline_ece | accepted_ece | ece_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | binary | 114 | 0.9500 | 0.0175 | low | 0.9649 | 0.9643 | -0.0006 | 0.1958 | 0.1905 | 0.0053 |
| colic | binary | 72 | 0.9500 | 0.1667 | moderate | 0.8333 | 0.8667 | 0.0333 | 0.2821 | 0.2798 | 0.0023 |
| diabetes | binary | 154 | 0.9500 | 0.7208 | high | 0.7468 | 0.9535 | 0.2067 | 0.1111 | 0.1273 | -0.0162 |
| balance | multiclass | 125 | 0.9500 | 1.0000 | high | 0.8640 | nan | nan | 0.0624 | nan | nan |
| iris | multiclass | 30 | 0.9500 | 1.0000 | high | 0.9333 | nan | nan | 0.0617 | nan | nan |
