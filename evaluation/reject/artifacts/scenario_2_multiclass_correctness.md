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

## Result table

| dataset | epsilon | ncf | n_cal | n_test | n_classes | proxy_singleton_accuracy | proxy_singleton_accuracy_defined | proxy_singleton_count | singleton_precision | singleton_recall | singleton_correct_count | singleton_count | singleton_precision_recall_defined | accepted_top1_accuracy | proxy_negative_singleton_accuracy | non_accepted_rate | reject_rate | positive_singleton_rate | correct_singleton_rate | proxy_negative_singleton_rate | error_singleton_rate | ambiguity_rate | novelty_rate | expected_collapse | guarantee_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.0500 | default | 125 | 125 | 3 | 0.9630 | yes | 108 | 0.9630 | 0.8320 | 104 | 108 | yes | 0.9709 | 0.8000 | 0.1760 | 0.1760 | 0.8240 | 0.8240 | 0.0400 | 0.0400 | 0.1360 | 0.0000 | no | empirical |
| balance | 0.0500 | ensured | 125 | 125 | 3 | 0.9778 | yes | 90 | 0.9778 | 0.7040 | 88 | 90 | yes | 0.9778 | nan | 0.2800 | 0.2800 | 0.7200 | 0.7200 | 0.0000 | 0.0000 | 0.2800 | 0.0000 | no | empirical |
| balance | 0.1000 | default | 125 | 125 | 3 | 0.9113 | yes | 124 | 0.9113 | 0.9040 | 113 | 124 | yes | 0.9279 | 0.7692 | 0.1120 | 0.1120 | 0.8880 | 0.8880 | 0.1040 | 0.1040 | 0.0000 | 0.0080 | no | empirical |
| balance | 0.1000 | ensured | 125 | 125 | 3 | 0.9778 | yes | 90 | 0.9778 | 0.7040 | 88 | 90 | yes | 0.9778 | nan | 0.2800 | 0.2800 | 0.7200 | 0.7200 | 0.0000 | 0.0000 | 0.2800 | 0.0000 | no | empirical |
| cars | 0.0500 | default | 346 | 346 | 4 | 0.9727 | yes | 330 | 0.9727 | 0.9277 | 321 | 330 | yes | 0.9727 | nan | 0.0462 | 0.0462 | 0.9538 | 0.9538 | 0.0000 | 0.0000 | 0.0000 | 0.0462 | no | empirical |
| cars | 0.0500 | ensured | 346 | 346 | 4 | 0.9698 | yes | 331 | 0.9698 | 0.9277 | 321 | 331 | yes | 0.9698 | nan | 0.0434 | 0.0434 | 0.9566 | 0.9566 | 0.0000 | 0.0000 | 0.0434 | 0.0000 | no | empirical |
| cars | 0.1000 | default | 346 | 346 | 4 | 0.9842 | yes | 316 | 0.9842 | 0.8988 | 311 | 316 | yes | 0.9842 | nan | 0.0867 | 0.0867 | 0.9133 | 0.9133 | 0.0000 | 0.0000 | 0.0000 | 0.0867 | no | empirical |
| cars | 0.1000 | ensured | 346 | 346 | 4 | 0.9698 | yes | 331 | 0.9698 | 0.9277 | 321 | 331 | yes | 0.9698 | nan | 0.0434 | 0.0434 | 0.9566 | 0.9566 | 0.0000 | 0.0000 | 0.0434 | 0.0000 | no | empirical |
| cmc | 0.0500 | default | 295 | 295 | 3 | 0.7593 | yes | 54 | 0.7593 | 0.1390 | 41 | 54 | yes | 0.8085 | 0.4286 | 0.8407 | 0.8407 | 0.1593 | 0.1593 | 0.0237 | 0.0237 | 0.8169 | 0.0000 | no | empirical |
| cmc | 0.0500 | ensured | 295 | 295 | 3 | 0.7805 | yes | 41 | 0.7805 | 0.1085 | 32 | 41 | yes | 0.7805 | nan | 0.8610 | 0.8610 | 0.1390 | 0.1390 | 0.0000 | 0.0000 | 0.8203 | 0.0407 | no | empirical |
| cmc | 0.1000 | default | 295 | 295 | 3 | 0.6813 | yes | 91 | 0.6813 | 0.2102 | 62 | 91 | yes | 0.8302 | 0.4737 | 0.8203 | 0.8203 | 0.1797 | 0.1797 | 0.1288 | 0.1288 | 0.6915 | 0.0000 | no | empirical |
| cmc | 0.1000 | ensured | 295 | 295 | 3 | 0.6552 | yes | 29 | 0.6552 | 0.0644 | 19 | 29 | yes | 0.6552 | nan | 0.9017 | 0.9017 | 0.0983 | 0.0983 | 0.0000 | 0.0000 | 0.8000 | 0.1017 | no | empirical |

_Showing first 12 of 80 rows._
