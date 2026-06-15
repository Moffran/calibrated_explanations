# Scenario 6 — Finite-sample stress tests

Rows: 51

## Key findings

- Stress tests focus on finite-sample behavior and extreme confidence settings.
- The suite reports coverage violations empirically rather than claiming new guarantees.
- violation is computed from actual coverage, not hard-coded.
- extreme_confidence probe uses the same violation logic as small_calibration.

## Outcome snapshot

- **rows**: 51
- **violations**: 28
- **max_reject_rate**: 1.0000
- **small_cal_violations**: 27
- **extreme_conf_violations**: 1

## Violation rates by n_cal (small_calibration probe)

| n_cal | total_rows | violations | violation_rate | mean_reject_rate |
|---|---|---|---|---|
| 10.0000 | 9.0000 | 2.0000 | 0.2222 | 0.5066 |
| 20.0000 | 9.0000 | 5.0000 | 0.5556 | 0.2254 |
| 50.0000 | 9.0000 | 7.0000 | 0.7778 | 0.1628 |
| 100.0000 | 9.0000 | 6.0000 | 0.6667 | 0.1591 |
| 200.0000 | 9.0000 | 7.0000 | 0.7778 | 0.1716 |

## Violation rates by epsilon

| epsilon | violations | violation_rate | mean_coverage |
|---|---|---|---|
| 0.0050 | 0.0000 | 0.0000 | 1.0000 |
| 0.0100 | 1.0000 | 0.3333 | 0.9888 |
| 0.0500 | 9.0000 | 0.6000 | 0.9481 |
| 0.1000 | 10.0000 | 0.6667 | 0.8750 |
| 0.2500 | 8.0000 | 0.5333 | 0.7174 |
