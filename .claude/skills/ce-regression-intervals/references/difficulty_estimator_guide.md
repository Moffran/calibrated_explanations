# DifficultyEstimator Guide for Per-Instance Adaptive Intervals

For standard regression, a global interval width treats all test instances
identically. Adding a `DifficultyEstimator` from `crepes.extras` makes the
CPS (Conformal Predictive System) scale each interval by the predicted
difficulty sigma_i — easy instances get tighter intervals, hard ones get wider.

**Effect**: `set_difficulty_estimator` triggers a **full refit** of the
internal `ConformalPredictiveSystem` with `sigmas=`. The previous calibrator
is invalidated and rebuilt with the new difficulty values.

## Method 1 — pass at calibrate time (recommended for fresh setups)

```python
from crepes.extras import DifficultyEstimator

# Option A: variance normalization using the fitted learner's predictions
de = DifficultyEstimator().fit(X=X_prop_train, learner=regressor.learner, scaler=True)

# Option B: residual-based using OOB for RandomForest
oob_preds = regressor.learner.oob_prediction_   # needs oob_score=True
residuals = y_prop_train - oob_preds
de = DifficultyEstimator().fit(X=X_prop_train, residuals=residuals, scaler=True)

# Option C: fit directly on calibration targets (simpler, less accurate)
de = DifficultyEstimator().fit(X=X_prop_train, y=y_prop_train, scaler=True)

regressor.calibrate(
    x_cal, y_cal,
    feature_names=feature_names,
    difficulty_estimator=de,
)
```

## Method 2 — set (or change) after calibration

```python
from crepes.extras import DifficultyEstimator

# Add difficulty estimation post-calibration (rebuilds the CPS internally)
de = DifficultyEstimator().fit(X=X_prop_train, learner=regressor.learner, scaler=True)
regressor.set_difficulty_estimator(de)
```

**Requires**: `WrapCalibratedExplainer` must already be fitted and calibrated.

## Removing the DifficultyEstimator

```python
# Remove: resets to uniform sigma=1 (flat intervals)
regressor.set_difficulty_estimator(None)

# Alternative: recalibrate without it
regressor.calibrate(x_cal, y_cal, feature_names=feature_names)
```

## Validation rules

- `DifficultyEstimator` must have `fitted=True` before passing to CE.
- Passing an unfitted instance raises `NotFittedError`.
- The estimator must expose an `apply(X) -> np.ndarray` method.
- `scaler=True` is recommended (stabilises sigma values to a reasonable range).

## Effect on interval semantics

| Setup | Interval source |
|---|---|
| No DifficultyEstimator | Global CPS — same width for all instances |
| With DifficultyEstimator | Normalized CPS scaled by sigma_i — wider for hard, tighter for easy |

The difficulty estimator does **not** change how `low_high_percentiles` or
`threshold` are interpreted — it only scales the underlying conformal scores.
