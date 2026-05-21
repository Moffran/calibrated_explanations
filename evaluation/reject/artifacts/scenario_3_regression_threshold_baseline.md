# Scenario 3 — Threshold regression heuristic baseline

Rows: 12

## Key findings

- Headline finding: threshold reject does NOT select by uncertainty — accepted-subset interval width equals full-set interval width (~0 delta).
- Mean interval_width_delta across all rows: -0.0000 (near zero confirms the null result).
- Threshold-based regression reject remains explicitly heuristic in this suite.
- Both interval width and MSE are tracked on the accepted subset to capture the trade-off.
- The difficulty-normalised approach (C3) is deferred to a standalone scenario post-RT2.

## Outcome snapshot

- **datasets**: 2
- **mean_reject_rate**: 0.3275
- **mean_accepted_mse_empirical**: 0.0217
- **mean_interval_width_delta**: -0.0000

## Result table

| dataset | confidence | effective_confidence | threshold_quantile | effective_threshold | threshold_source | n_cal | n_test | interval_coverage_all | accepted_coverage_empirical | interval_width_all | accepted_interval_width_empirical | interval_width_delta | mse_all | accepted_mse_empirical | reject_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diabetes_reg | 0.9000 | 0.9000 | 0.1000 | 0.1079 | call_reinitialized | 89 | 89 | 0.9775 | 0.9750 | 0.7036 | 0.7036 | 0.0000 | 0.0272 | 0.0288 | 0.1011 |
| diabetes_reg | 0.9500 | 0.9500 | 0.1000 | 0.1079 | call | 89 | 89 | 0.9775 | 0.9677 | 0.7036 | 0.7036 | 0.0000 | 0.0272 | 0.0329 | 0.3034 |
| diabetes_reg | 0.9000 | 0.9000 | 0.3000 | 0.1750 | call_reinitialized | 89 | 89 | 0.9101 | 0.8750 | 0.7077 | 0.7077 | 0.0000 | 0.0383 | 0.0390 | 0.4607 |
| diabetes_reg | 0.9500 | 0.9500 | 0.3000 | 0.1750 | call | 89 | 89 | 0.9101 | 0.8696 | 0.7077 | 0.7077 | 0.0000 | 0.0383 | 0.0406 | 0.4831 |
| diabetes_reg | 0.9000 | 0.9000 | 0.5000 | 0.3587 | call_reinitialized | 89 | 89 | 0.8876 | 0.8824 | 0.6313 | 0.6313 | 0.0000 | 0.0346 | 0.0364 | 0.4270 |
| diabetes_reg | 0.9500 | 0.9500 | 0.5000 | 0.3587 | call | 89 | 89 | 0.8876 | 0.8378 | 0.6313 | 0.6313 | -0.0000 | 0.0346 | 0.0389 | 0.5843 |
| abalone | 0.9000 | 0.9000 | 0.1000 | 0.1818 | call_reinitialized | 836 | 836 | 0.8947 | 0.8912 | 0.3109 | 0.3109 | -0.0000 | 0.0105 | 0.0109 | 0.0431 |
| abalone | 0.9500 | 0.9500 | 0.1000 | 0.1818 | call | 836 | 836 | 0.8947 | 0.8877 | 0.3109 | 0.3109 | -0.0000 | 0.0105 | 0.0111 | 0.0730 |
| abalone | 0.9000 | 0.9000 | 0.3000 | 0.2500 | call_reinitialized | 836 | 836 | 0.9450 | 0.9404 | 0.2875 | 0.2875 | -0.0000 | 0.0051 | 0.0054 | 0.0969 |
| abalone | 0.9500 | 0.9500 | 0.3000 | 0.2500 | call | 836 | 836 | 0.9450 | 0.9252 | 0.2875 | 0.2875 | -0.0000 | 0.0051 | 0.0062 | 0.3600 |
| abalone | 0.9000 | 0.9000 | 0.5000 | 0.3462 | call_reinitialized | 836 | 836 | 0.9246 | 0.9254 | 0.2889 | 0.2889 | 0.0000 | 0.0065 | 0.0059 | 0.4067 |
| abalone | 0.9500 | 0.9500 | 0.5000 | 0.3462 | call | 836 | 836 | 0.9246 | 0.9415 | 0.2889 | 0.2889 | 0.0000 | 0.0065 | 0.0044 | 0.5909 |
