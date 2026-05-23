# Scenario 2 — Multiclass correctness classifier

Rows: 80

## Key findings

- Accepted top-1 accuracy is reported empirically; the formal guarantee remains a proof obligation.
- This scenario evaluates CE multiclass reject as a correctness classifier, not a K-class prediction-set method.
- Accepted instances are restricted to {1} singletons (confident correct); {0} singletons (confident wrong) are error-rejected.
- Hinge NCF is used for both 'default' and 'ensured' paths. Margin NCF was removed (it produced identical scores for both columns, making singletons impossible).

## Outcome snapshot

- **datasets**: 20
- **mean_accepted_top1_accuracy**: 0.9006
- **mean_reject_rate**: 0.4711
- **mean_correct_singleton_rate**: 0.5289
- **mean_error_singleton_rate**: 0.0200
- **collapse_events**: 8

## Result table

| dataset | epsilon | ncf | n_cal | n_test | n_classes | accepted_top1_accuracy | reject_rate | correct_singleton_rate | error_singleton_rate | ambiguity_rate | novelty_rate | expected_collapse | guarantee_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.0500 | default | 125 | 125 | 3 | 0.9709 | 0.1760 | 0.8240 | 0.0400 | 0.1360 | 0.0000 | no | empirical |
| balance | 0.0500 | ensured | 125 | 125 | 3 | 0.9778 | 0.2800 | 0.7200 | 0.0000 | 0.2800 | 0.0000 | no | empirical |
| balance | 0.1000 | default | 125 | 125 | 3 | 0.9279 | 0.1120 | 0.8880 | 0.1040 | 0.0000 | 0.0080 | no | empirical |
| balance | 0.1000 | ensured | 125 | 125 | 3 | 0.9778 | 0.2800 | 0.7200 | 0.0000 | 0.2800 | 0.0000 | no | empirical |
| cars | 0.0500 | default | 346 | 346 | 4 | 0.9727 | 0.0462 | 0.9538 | 0.0000 | 0.0000 | 0.0462 | no | empirical |
| cars | 0.0500 | ensured | 346 | 346 | 4 | 0.9698 | 0.0434 | 0.9566 | 0.0000 | 0.0434 | 0.0000 | no | empirical |
| cars | 0.1000 | default | 346 | 346 | 4 | 0.9842 | 0.0867 | 0.9133 | 0.0000 | 0.0000 | 0.0867 | no | empirical |
| cars | 0.1000 | ensured | 346 | 346 | 4 | 0.9698 | 0.0434 | 0.9566 | 0.0000 | 0.0434 | 0.0000 | no | empirical |
| cmc | 0.0500 | default | 295 | 295 | 3 | 0.8085 | 0.8407 | 0.1593 | 0.0237 | 0.8169 | 0.0000 | no | empirical |
| cmc | 0.0500 | ensured | 295 | 295 | 3 | 0.7805 | 0.8610 | 0.1390 | 0.0000 | 0.8203 | 0.0407 | no | empirical |
| cmc | 0.1000 | default | 295 | 295 | 3 | 0.8302 | 0.8203 | 0.1797 | 0.1288 | 0.6915 | 0.0000 | no | empirical |
| cmc | 0.1000 | ensured | 295 | 295 | 3 | 0.6552 | 0.9017 | 0.0983 | 0.0000 | 0.8000 | 0.1017 | no | empirical |

_Showing first 12 of 80 rows._
