# Scenario A — Policy matrix

Rows: 27

## Key findings

- Best accuracy uplift came from `flag` at confidence 0.80, improving accepted accuracy by 0.0439 at coverage 0.8333.
- Baseline full-sample accuracy was 0.9561 and baseline ECE was 0.0342.
- `flag` explains every instance, while `only_rejected` and `only_accepted` shift explanation volume to the rejected or accepted subset respectively.

## Outcome snapshot

- **best_policy**: flag
- **best_confidence**: 0.8000
- **best_accuracy_delta**: 0.0439
- **best_coverage**: 0.8333
- **baseline_accuracy**: 0.9561
- **baseline_ece**: 0.0342

## Plots

- ![policy_matrix_tradeoffs.png](policy_matrix_tradeoffs.png)

## Result table

| policy | confidence | coverage | reject_rate | error_rate | ambiguity_rate | novelty_rate | accepted_accuracy | accepted_accuracy_delta | accepted_ece | accepted_ece_delta |
|---|---|---|---|---|---|---|---|---|---|---|
| flag | 0.8000 | 0.8333 | 0.1667 | 0.0400 | 0.0000 | 0.1667 | 1.0000 | 0.0439 | 0.0242 | 0.0100 |
| flag | 0.8237 | 0.8684 | 0.1316 | 0.0514 | 0.0000 | 0.1316 | 0.9899 | 0.0338 | 0.0275 | 0.0067 |
| flag | 0.8475 | 0.9035 | 0.0965 | 0.0620 | 0.0000 | 0.0965 | 0.9903 | 0.0342 | 0.0305 | 0.0037 |
| flag | 0.8713 | 0.9211 | 0.0789 | 0.0541 | 0.0000 | 0.0789 | 0.9810 | 0.0248 | 0.0306 | 0.0036 |
| flag | 0.8950 | 0.9298 | 0.0702 | 0.0375 | 0.0000 | 0.0702 | 0.9811 | 0.0250 | 0.0325 | 0.0017 |
| flag | 0.9187 | 0.9474 | 0.0526 | 0.0302 | 0.0000 | 0.0526 | 0.9722 | 0.0161 | 0.0316 | 0.0026 |
| flag | 0.9425 | 0.9474 | 0.0526 | 0.0051 | 0.0000 | 0.0526 | 0.9722 | 0.0161 | 0.0316 | 0.0026 |
| flag | 0.9663 | 0.9912 | 0.0088 | 0.0252 | 0.0000 | 0.0088 | 0.9646 | 0.0085 | 0.0393 | -0.0051 |
| flag | 0.9900 | 0.9474 | 0.0526 | 0.0106 | 0.0526 | 0.0000 | 0.9722 | 0.0161 | 0.0316 | 0.0026 |
| only_rejected | 0.8000 | 0.8333 | 0.1667 | 0.0400 | 0.0000 | 0.1667 | 1.0000 | 0.0439 | 0.0242 | 0.0100 |
| only_rejected | 0.8237 | 0.8684 | 0.1316 | 0.0514 | 0.0000 | 0.1316 | 0.9899 | 0.0338 | 0.0275 | 0.0067 |
| only_rejected | 0.8475 | 0.9035 | 0.0965 | 0.0620 | 0.0000 | 0.0965 | 0.9903 | 0.0342 | 0.0305 | 0.0037 |

_Showing first 12 of 27 rows._
