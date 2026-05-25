# Scenario 2 - Multiclass correctness proxy

Rows: 80

## Key findings

- Primary empirical accuracy is computed in the binary proxy space: singleton {1}/{0} is compared with 1[top-1 prediction is correct].
- Accepted top-1 accuracy remains a precision-style diagnostic on {1} rows only; it is not the proxy classifier accuracy.
- This scenario opts into the multiclass-only experimental.multiclass_top1_correctness strategy.
- It evaluates CE multiclass reject as a binary correctness proxy, not a default rule and not a K-class prediction-set method.
- Accepted instances are restricted to {1} positive correctness-proxy singletons.
- {0} singletons are proxy-negative singletons: the aggregate non-top1 event is conforming, but no specific alternative class is selected.
- reject_rate is retained as a compatibility alias for non_accepted_rate in this proxy scenario.
- Hinge NCF is used for both 'default' and 'ensured' paths. Margin NCF was removed (it produced identical scores for both columns, making singletons impossible).

## Outcome snapshot

- **datasets**: 20
- **mean_proxy_singleton_accuracy**: 0.8841
- **mean_singleton_precision**: 0.8841
- **mean_singleton_recall**: 0.5031
- **mean_accepted_top1_accuracy**: 0.9006
- **mean_proxy_negative_singleton_accuracy**: 0.5289
- **mean_non_accepted_rate**: 0.4711
- **mean_reject_rate**: 0.4711
- **mean_positive_singleton_rate**: 0.5289
- **mean_correct_singleton_rate**: 0.5289
- **mean_proxy_negative_singleton_rate**: 0.0200
- **mean_error_singleton_rate**: 0.0200
- **collapse_events**: 8

## Per-dataset proxy accuracy

| dataset | ncf | n_classes | proxy_singleton_accuracy | singleton_precision | singleton_recall | non_accepted_rate |
|---|---|---|---|---|---|---|
| balance | default | 3 | 0.9371 | 0.9371 | 0.8680 | 0.1440 |
| balance | ensured | 3 | 0.9778 | 0.9778 | 0.7040 | 0.2800 |
| cars | default | 4 | 0.9785 | 0.9785 | 0.9133 | 0.0665 |
| cars | ensured | 4 | 0.9698 | 0.9698 | 0.9277 | 0.0434 |
| cmc | default | 3 | 0.7203 | 0.7203 | 0.1746 | 0.8305 |
| cmc | ensured | 3 | 0.7178 | 0.7178 | 0.0864 | 0.8814 |
| cool | default | 3 | 0.9829 | 0.9829 | 0.9026 | 0.0812 |
| cool | ensured | 3 | 0.9732 | 0.9732 | 0.9416 | 0.0325 |
| ecoli | default | 6 | 0.8120 | 0.8120 | 0.5000 | 0.5074 |
| ecoli | ensured | 6 | 1.0000 | 1.0000 | 0.4559 | 0.5441 |
| glass | default | 6 | 0.8368 | 0.8368 | 0.3605 | 0.6977 |
| glass | ensured | 6 | 1.0000 | 1.0000 | 0.3256 | 0.6744 |
| heat | default | 3 | 0.9966 | 0.9966 | 0.9156 | 0.0812 |
| heat | ensured | 3 | 0.9933 | 0.9933 | 0.9610 | 0.0325 |
| image | default | 7 | 0.9899 | 0.9899 | 0.9502 | 0.0465 |
| image | ensured | 7 | 0.9951 | 0.9951 | 0.8810 | 0.1147 |
| iris | default | 3 | nan | nan | 0.0000 | 1.0000 |
| iris | ensured | 3 | nan | nan | 0.0000 | 1.0000 |
| steel | default | 7 | 0.8867 | 0.8867 | 0.5720 | 0.3817 |
| steel | ensured | 7 | 0.9283 | 0.9283 | 0.5321 | 0.4267 |
| tae | default | 3 | 0.5778 | 0.5778 | 0.2258 | 0.6774 |
| tae | ensured | 3 | 0.5682 | 0.5682 | 0.1774 | 0.6935 |
| user | default | 5 | 0.8758 | 0.8758 | 0.7840 | 0.1543 |
| user | ensured | 5 | 0.9212 | 0.9212 | 0.7222 | 0.2160 |
| vehicle | default | 4 | 0.8788 | 0.8788 | 0.6029 | 0.4000 |
| vehicle | ensured | 4 | 0.9858 | 0.9858 | 0.4118 | 0.5824 |
| vowel | default | 11 | 0.9654 | 0.9654 | 0.9015 | 0.1061 |
| vowel | ensured | 11 | 0.9966 | 0.9966 | 0.7626 | 0.2348 |
| wave | default | 3 | 0.9148 | 0.9148 | 0.7360 | 0.1935 |
| wave | ensured | 3 | 0.9255 | 0.9255 | 0.6870 | 0.2575 |
| whole | default | 3 | 0.8429 | 0.8429 | 0.1761 | 0.7841 |
| whole | ensured | 3 | 0.8429 | 0.8429 | 0.1761 | 0.7841 |
| wine | default | 3 | nan | nan | 0.0000 | 1.0000 |
| wine | ensured | 3 | nan | nan | 0.0000 | 1.0000 |
| wineR | default | 6 | 0.8276 | 0.8276 | 0.3141 | 0.6578 |
| wineR | ensured | 6 | 0.8616 | 0.8616 | 0.2859 | 0.6672 |
| wineW | default | 7 | 0.8359 | 0.8359 | 0.3551 | 0.5903 |
| wineW | ensured | 7 | 0.8332 | 0.8332 | 0.3122 | 0.6250 |
| yeast | default | 10 | 0.7521 | 0.7521 | 0.2811 | 0.6852 |
| yeast | ensured | 10 | 0.7254 | 0.7254 | 0.2391 | 0.6684 |

## NCF comparison

| ncf | mean_proxy_singleton_accuracy | mean_singleton_precision | mean_singleton_recall | mean_non_accepted_rate | mean_ambiguity_rate | collapse_events |
|---|---|---|---|---|---|---|
| default | 0.8673 | 0.8673 | 0.5267 | 0.4543 | 0.3973 | 4 |
| ensured | 0.9009 | 0.9009 | 0.4795 | 0.4879 | 0.4650 | 4 |
