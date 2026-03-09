# Scenario K — Regression reject comparison

Rows: 264

## Key findings

- The primary method follows the 2024 conformal regression with reject option paper via difficulty-based Mondrian categories.
- Threshold and value-bin methods are retained only as heuristic baselines.
- Accepted-subset metrics are explicitly empirical for the heuristic baselines.

## Outcome snapshot

- **datasets**: 22
- **methods**: paper_difficulty_mondrian, threshold_baseline, value_bin_width_baseline
- **best_accepted_mae**: 0.0000

## Result table

| dataset | confidence | method | difficulty_estimator | requested_reject_rate | empirical_reject_rate | accepted_coverage | accepted_interval_width | accepted_mae | accepted_mse | interval_coverage_all | mse_all | guarantee_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diabetes_reg | 0.9500 | paper_difficulty_mondrian | default | 0.1000 | 0.0337 | 0.9419 | 0.6257 | 0.1429 | 0.0298 | 0.9438 | 0.0307 | target_formal_result |
| diabetes_reg | 0.9500 | paper_difficulty_mondrian | default | 0.2000 | 0.2472 | 0.9254 | 0.5480 | 0.1303 | 0.0241 | 0.9438 | 0.0307 | target_formal_result |
| diabetes_reg | 0.9500 | paper_difficulty_mondrian | default | 0.3000 | 0.2697 | 0.9231 | 0.5413 | 0.1293 | 0.0241 | 0.9438 | 0.0307 | target_formal_result |
| diabetes_reg | 0.9500 | paper_difficulty_mondrian | default | 0.4000 | 0.3258 | 0.9167 | 0.5270 | 0.1252 | 0.0231 | 0.9438 | 0.0307 | target_formal_result |
| diabetes_reg | 0.9500 | paper_difficulty_mondrian | default | 0.5000 | 0.3933 | 0.9259 | 0.5087 | 0.1204 | 0.0207 | 0.9438 | 0.0307 | target_formal_result |
| diabetes_reg | 0.9500 | threshold_baseline | none | 0.1000 | 0.2809 | 0.9688 | 0.7275 | 0.1537 | 0.0328 | 0.9775 | 0.0275 | heuristic |
| diabetes_reg | 0.9500 | threshold_baseline | none | 0.2000 | 0.5506 | 0.9000 | 0.6528 | 0.1418 | 0.0311 | 0.9438 | 0.0276 | heuristic |
| diabetes_reg | 0.9500 | threshold_baseline | none | 0.3000 | 0.5618 | 0.8462 | 0.6937 | 0.1588 | 0.0426 | 0.9101 | 0.0373 | heuristic |
| diabetes_reg | 0.9500 | threshold_baseline | none | 0.4000 | 0.5843 | 0.9459 | 0.6897 | 0.1191 | 0.0227 | 0.9775 | 0.0284 | heuristic |
| diabetes_reg | 0.9500 | threshold_baseline | none | 0.5000 | 0.5618 | 0.8718 | 0.6028 | 0.1436 | 0.0326 | 0.8876 | 0.0342 | heuristic |
| diabetes_reg | 0.9500 | value_bin_width_baseline | predicted_value_bins | nan | 0.0000 | 0.9438 | 0.6898 | 0.1505 | 0.0343 | 0.9438 | 0.0343 | heuristic |
| diabetes_reg | 0.9500 | value_bin_width_baseline | predicted_value_bins | nan | 0.0000 | 1.0000 | inf | 0.1501 | 0.0344 | 1.0000 | 0.0344 | heuristic |

_Showing first 12 of 264 rows._
