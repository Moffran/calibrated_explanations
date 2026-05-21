# Scenario 2 — Multiclass correctness classifier

Rows: 80

## Key findings

- Accepted top-1 accuracy is reported empirically; the formal guarantee remains a proof obligation.
- This scenario evaluates CE multiclass reject as a correctness classifier, not a K-class prediction-set method.

## Outcome snapshot

- **datasets**: 20
- **mean_accepted_top1_accuracy**: nan
- **mean_reject_rate**: 1.0000
- **hinge_collapse_events**: 0

## Result table

| dataset | epsilon | ncf | n_cal | n_test | n_classes | accepted_top1_accuracy | reject_rate | ambiguity_rate | expected_collapse | guarantee_status |
|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.0500 | default | 125 | 125 | 3 | nan | 1.0000 | 0.9920 | no | empirical |
| balance | 0.0500 | ensured | 125 | 125 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
| balance | 0.1000 | default | 125 | 125 | 3 | nan | 1.0000 | 0.9440 | no | empirical |
| balance | 0.1000 | ensured | 125 | 125 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
| cars | 0.0500 | default | 346 | 346 | 4 | nan | 1.0000 | 0.9480 | no | empirical |
| cars | 0.0500 | ensured | 346 | 346 | 4 | nan | 1.0000 | 1.0000 | no | empirical |
| cars | 0.1000 | default | 346 | 346 | 4 | nan | 1.0000 | 0.9104 | no | empirical |
| cars | 0.1000 | ensured | 346 | 346 | 4 | nan | 1.0000 | 1.0000 | no | empirical |
| cmc | 0.0500 | default | 295 | 295 | 3 | nan | 1.0000 | 0.9729 | no | empirical |
| cmc | 0.0500 | ensured | 295 | 295 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
| cmc | 0.1000 | default | 295 | 295 | 3 | nan | 1.0000 | 0.9017 | no | empirical |
| cmc | 0.1000 | ensured | 295 | 295 | 3 | nan | 1.0000 | 1.0000 | no | empirical |

_Showing first 12 of 80 rows._
