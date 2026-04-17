# Perspective 2: Signal Retention

## Setup
- Seeds: 2
- Models: logreg, rf
- Features: 4 (informative: 0, 1; noise: 2, 3)
- Significance levels: [0.05, 0.1, 0.2]
- Bootstrap draws: 3
- Stability significance: 0.1

## Driver Recovery

Fraction of instances where the top-ranked feature(s) by |weight|
are informative (features 0 and 1).  Standard multi-bin uses the same
discretiser as guarded but without the conformal guard.

| method            |   significance | model   |   top1_recovery |   top2_recovery |   noise_rule_frac |   mean_rules |
|:------------------|---------------:|:--------|----------------:|----------------:|------------------:|-------------:|
| guarded           |           0.05 | logreg  |         1       |         1       |          0.348958 |      3.28125 |
| guarded           |           0.05 | rf      |         1       |         0.96875 |          0.447917 |      3.6875  |
| guarded           |           0.1  | logreg  |         1       |         1       |          0.322917 |      3.15625 |
| guarded           |           0.1  | rf      |         1       |         0.9375  |          0.427083 |      3.5     |
| guarded           |           0.2  | logreg  |         0.96875 |         0.9375  |          0.270833 |      2.8125  |
| guarded           |           0.2  | rf      |         0.96875 |         0.90625 |          0.359375 |      3.0625  |
| standard_multibin |         nan    | logreg  |         0.90625 |         0.8125  |          0.328125 |      5.125   |
| standard_multibin |         nan    | rf      |         0.84375 |         0.75    |          0.479911 |      6.1875  |

## Stability Under Calibration Resampling

Mean Jaccard overlap of emitted (feature, condition) sets between a
reference calibration and 3 bootstrap resamples.

| method            | model   |   mean_jaccard |   std_jaccard |
|:------------------|:--------|---------------:|--------------:|
| guarded           | logreg  |      0.0668155 |      0.142825 |
| guarded           | rf      |      0.0715774 |      0.174396 |
| standard_multibin | logreg  |      0.16087   |      0.176011 |
| standard_multibin | rf      |      0.112574  |      0.107665 |
