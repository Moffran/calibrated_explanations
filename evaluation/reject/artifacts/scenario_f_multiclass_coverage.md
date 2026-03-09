# Scenario F — Multiclass correctness evaluation

Rows: 80

## Key findings

- Accepted top-1 accuracy is reported empirically while the formal guarantee remains a proof obligation.
- The artifact explicitly marks `guarantee_status=empirical` to avoid over-claiming.
- This scenario evaluates CE multiclass reject as a correctness classifier, not as a K-class prediction-set method.

## Outcome snapshot

- **datasets**: 20
- **mean_accepted_top1_accuracy**: 0.7620
- **mean_reject_rate**: 0.6034

## Result table

| dataset | epsilon | ncf | n_cal | n_test | accepted_top1_accuracy | reject_rate | ambiguity_rate | guarantee_status |
|---|---|---|---|---|---|---|---|---|
| balance | 0.0500 | hinge | 125 | 125 | 0.9352 | 0.1360 | 0.1360 | empirical |
| balance | 0.0500 | margin | 125 | 125 | 0.2857 | 0.9440 | 0.9360 | empirical |
| balance | 0.1000 | hinge | 125 | 125 | 0.8548 | 0.0080 | 0.0000 | empirical |
| balance | 0.1000 | margin | 125 | 125 | 0.0000 | 0.9840 | 0.9280 | empirical |
| cars | 0.0500 | hinge | 346 | 346 | 0.9727 | 0.0462 | 0.0000 | empirical |
| cars | 0.0500 | margin | 346 | 346 | 0.9727 | 0.0462 | 0.0000 | empirical |
| cars | 0.1000 | hinge | 346 | 346 | 0.9842 | 0.0867 | 0.0000 | empirical |
| cars | 0.1000 | margin | 346 | 346 | 0.9842 | 0.0867 | 0.0000 | empirical |
| cmc | 0.0500 | hinge | 295 | 295 | 0.7778 | 0.8169 | 0.8169 | empirical |
| cmc | 0.0500 | margin | 295 | 295 | 0.5714 | 0.9288 | 0.9288 | empirical |
| cmc | 0.1000 | hinge | 295 | 295 | 0.7033 | 0.6915 | 0.6915 | empirical |
| cmc | 0.1000 | margin | 295 | 295 | 0.4419 | 0.8542 | 0.8508 | empirical |

_Showing first 12 of 80 rows._
