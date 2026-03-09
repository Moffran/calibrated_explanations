# Scenario D — Regression threshold sweep

Rows: 5

## Key findings

- Best threshold quantile was 0.20, improving thresholded binary accuracy by 0.2059.
- Accepted-set interval miss fraction changed by -0.0438 at that setting.
- This scenario evaluates real reject learner thresholds for probabilistic regression rather than synthetic interval scaling.

## Outcome snapshot

- **best_threshold_quantile**: 0.2000
- **best_threshold_value**: 0.1759
- **best_binary_accuracy_delta**: 0.2059
- **best_outside_fraction_delta**: -0.0438

## Plots

- ![regression_threshold_tradeoffs.png](regression_threshold_tradeoffs.png)

## Result table

| threshold_quantile | threshold_value | coverage | reject_rate | error_rate | outside_fraction_all | outside_fraction_accepted | binary_accuracy_all | binary_accuracy_accepted | mse_all | mse_accepted | binary_accuracy_delta | outside_fraction_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.1190 | 0.3146 | 0.6854 | 0.1589 | 0.0225 | 0.0357 | 0.8764 | 1.0000 | 0.0275 | 0.0314 | 0.1236 | -0.0132 |
| 0.2000 | 0.1759 | 0.4494 | 0.5506 | 0.1113 | 0.0562 | 0.1000 | 0.7191 | 0.9250 | 0.0276 | 0.0311 | 0.2059 | -0.0438 |
| 0.3000 | 0.2500 | 0.4944 | 0.5056 | 0.1011 | 0.0899 | 0.0682 | 0.7640 | 0.8182 | 0.0373 | 0.0277 | 0.0541 | 0.0217 |
| 0.4000 | 0.3304 | 0.3708 | 0.6292 | 0.1348 | 0.0225 | 0.0606 | 0.8315 | 0.9394 | 0.0284 | 0.0227 | 0.1079 | -0.0381 |
| 0.5000 | 0.3797 | 0.3146 | 0.6854 | 0.1589 | 0.1124 | 0.1786 | 0.6629 | 0.7500 | 0.0342 | 0.0368 | 0.0871 | -0.0662 |
