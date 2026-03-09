# Scenario G — Threshold regression empirical coverage

Rows: 220

## Key findings

- Accepted-subset coverage is reported as an empirical quantity only.
- Threshold-based regression reject remains explicitly heuristic in this suite.
- Both interval width and MSE are tracked on the accepted subset to capture the trade-off.

## Outcome snapshot

- **datasets**: 22
- **mean_reject_rate**: 0.2083
- **mean_accepted_mse_empirical**: 0.0092

## Result table

| dataset | confidence | threshold_quantile | n_cal | n_test | interval_coverage_all | accepted_coverage_empirical | interval_width_all | accepted_interval_width_empirical | mse_all | accepted_mse_empirical | reject_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diabetes_reg | 0.9000 | 0.1000 | 89 | 89 | 0.9775 | 0.9750 | 0.7275 | 0.7275 | 0.0275 | 0.0292 | 0.1011 |
| diabetes_reg | 0.9500 | 0.1000 | 89 | 89 | 0.9775 | 0.9688 | 0.7275 | 0.7275 | 0.0275 | 0.0328 | 0.2809 |
| diabetes_reg | 0.9000 | 0.2000 | 89 | 89 | 0.9438 | 0.9130 | 0.6528 | 0.6528 | 0.0276 | 0.0308 | 0.4831 |
| diabetes_reg | 0.9500 | 0.2000 | 89 | 89 | 0.9438 | 0.9000 | 0.6528 | 0.6528 | 0.0276 | 0.0311 | 0.5506 |
| diabetes_reg | 0.9000 | 0.3000 | 89 | 89 | 0.9101 | 0.8571 | 0.6937 | 0.6937 | 0.0373 | 0.0441 | 0.4494 |
| diabetes_reg | 0.9500 | 0.3000 | 89 | 89 | 0.9101 | 0.8462 | 0.6937 | 0.6937 | 0.0373 | 0.0426 | 0.5618 |
| diabetes_reg | 0.9000 | 0.4000 | 89 | 89 | 0.9775 | 0.9459 | 0.6897 | 0.6897 | 0.0284 | 0.0227 | 0.5843 |
| diabetes_reg | 0.9500 | 0.4000 | 89 | 89 | 0.9775 | 0.9459 | 0.6897 | 0.6897 | 0.0284 | 0.0227 | 0.5843 |
| diabetes_reg | 0.9000 | 0.5000 | 89 | 89 | 0.8876 | 0.8846 | 0.6028 | 0.6028 | 0.0342 | 0.0306 | 0.4157 |
| diabetes_reg | 0.9500 | 0.5000 | 89 | 89 | 0.8876 | 0.8718 | 0.6028 | 0.6028 | 0.0342 | 0.0326 | 0.5618 |
| abalone | 0.9000 | 0.1000 | 836 | 836 | 0.9031 | 0.9024 | 0.3258 | 0.3258 | 0.0107 | 0.0107 | 0.0072 |
| abalone | 0.9500 | 0.1000 | 836 | 836 | 0.9031 | 0.9024 | 0.3258 | 0.3258 | 0.0107 | 0.0107 | 0.0072 |

_Showing first 12 of 220 rows._
