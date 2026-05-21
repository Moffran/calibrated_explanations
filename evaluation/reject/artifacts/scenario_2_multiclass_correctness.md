# Scenario 2 — Multiclass correctness classifier

Rows: 8

## Key findings

- Accepted top-1 accuracy is reported empirically; the formal guarantee remains a proof obligation.
- This scenario evaluates CE multiclass reject as a correctness classifier, not a K-class prediction-set method.

## Outcome snapshot

- **datasets**: 2
- **mean_accepted_top1_accuracy**: nan
- **mean_reject_rate**: 1.0000
- **hinge_collapse_events**: 0

## Result table

| dataset | epsilon | ncf | n_cal | n_test | n_classes | accepted_top1_accuracy | reject_rate | ambiguity_rate | expected_collapse | guarantee_status |
|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.0500 | default | 125 | 125 | 3 | nan | 1.0000 | 0.9600 | no | empirical |
| balance | 0.0500 | ensured | 125 | 125 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
| balance | 0.1000 | default | 125 | 125 | 3 | nan | 1.0000 | 0.9040 | no | empirical |
| balance | 0.1000 | ensured | 125 | 125 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
| iris | 0.0500 | default | 30 | 30 | 3 | nan | 1.0000 | 0.8000 | no | empirical |
| iris | 0.0500 | ensured | 30 | 30 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
| iris | 0.1000 | default | 30 | 30 | 3 | nan | 1.0000 | 0.8000 | no | empirical |
| iris | 0.1000 | ensured | 30 | 30 | 3 | nan | 1.0000 | 1.0000 | no | empirical |
