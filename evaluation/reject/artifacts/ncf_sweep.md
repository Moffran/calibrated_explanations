# Scenario B — NCF sweep

Rows: 20

## Key findings

- Best mean accepted-accuracy uplift came from `hinge`, with delta 0.0494 at mean coverage 0.5800.
- This scenario uses `RejectPolicySpec.flag(...)` so the sweep exercises the public per-call policy+NCF contract rather than direct synthetic scoring.
- Prediction-set size highlights how aggressively each NCF concentrates accepted multiclass decisions.

## Outcome snapshot

- **best_ncf**: hinge
- **best_mean_accuracy_delta**: 0.0494
- **best_mean_coverage**: 0.5800
- **lowest_mean_error_rate**: 0.0000

## Plots

- ![ncf_sweep_comparison.png](ncf_sweep_comparison.png)

## Result table

| ncf | repeat | coverage | reject_rate | error_rate | ambiguity_rate | novelty_rate | accepted_accuracy | accepted_accuracy_delta | mean_prediction_set_size |
|---|---|---|---|---|---|---|---|---|---|
| hinge | 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | nan | nan | 2.0000 |
| entropy | 0 | 0.1333 | 0.8667 | 0.3750 | 0.8667 | 0.0000 | 1.0000 | 0.0667 | 1.8667 |
| margin | 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | nan | nan | 2.0000 |
| ensured | 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | nan | nan | 2.0000 |
| hinge | 1 | 0.7000 | 0.3000 | 0.0714 | 0.3000 | 0.0000 | 1.0000 | 0.0667 | 1.3000 |
| entropy | 1 | 0.0333 | 0.9667 | 1.0000 | 0.9667 | 0.0000 | 1.0000 | 0.0667 | 1.9667 |
| margin | 1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | nan | nan | 2.0000 |
| ensured | 1 | 0.7000 | 0.3000 | 0.0714 | 0.3000 | 0.0000 | 1.0000 | 0.0667 | 1.3000 |
| hinge | 2 | 0.6333 | 0.3667 | 0.0789 | 0.3667 | 0.0000 | 1.0000 | 0.0333 | 1.3667 |
| entropy | 2 | 0.2000 | 0.8000 | 0.2500 | 0.8000 | 0.0000 | 0.8333 | -0.1333 | 1.8000 |
| margin | 2 | 0.0000 | 1.0000 | 0.0000 | 0.9667 | 0.0333 | nan | nan | 1.9333 |
| ensured | 2 | 0.6333 | 0.3667 | 0.0789 | 0.3667 | 0.0000 | 1.0000 | 0.0333 | 1.3667 |

_Showing first 12 of 20 rows._
