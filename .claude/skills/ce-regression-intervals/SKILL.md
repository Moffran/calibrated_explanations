---
name: ce-regression-intervals
description: >
  Configure and interpret regression prediction intervals in calibrated_explanations.
  Use when working with regression tasks, asking about 'prediction interval',
  'conformal regression', 'low_high_percentiles', 'threshold regression', 'CPS',
  'calibrated probability for regression', 'probabilistic regression',
  'interval width', 'DifficultyEstimator', 'crepes.extras', 'per-instance
  difficulty', 'difficulty estimation', 'variance normalization',
  'adaptive intervals', 'set_difficulty_estimator', 'sigma'. Covers ADR-021
  three-mode semantics: percentile conformal intervals, thresholded probabilistic
  regression, and the distinction between them; and how per-instance difficulty
  estimation narrows intervals for easy instances and widens them for hard ones.
---

# CE Regression Intervals

You are configuring or interpreting regression intervals in CE. This skill
covers the three distinct regression modes defined in ADR-021. A fully
fit + calibrated `WrapCalibratedExplainer` (regression model) is required.

## Three Modes — Decision Tree

```
Is threshold= provided?
├── NO  → Mode 2: Percentile conformal intervals (CPS)
│         Use low_high_percentiles=(low%, high%)
│         Result: interval is a percentile band (e.g., 5th–95th percentile)
│
└── YES → Mode 3: Thresholded probabilistic regression (CPS + Venn-Abers)
          Result: calibrated P(y ≤ threshold) with probability interval
          Interval semantics = probability mass (NOT a percentile band)
```

Mode 1 (classification) is handled by `ce-pipeline-builder` and `ce-factual-explain`.

---

## Mode 2: Percentile Conformal Intervals

**Use when:** you want a confidence band around the regression prediction.

```python
# Default: 5th–95th percentile interval
explanations = explainer.explain_factual(X_query)

# Custom width: 10th–90th percentile (tighter)
explanations = explainer.explain_factual(X_query, low_high_percentiles=(10, 90))

# Very wide: 1st–99th percentile
explanations = explainer.explain_factual(X_query, low_high_percentiles=(1, 99))
```

**How to read the output:**
```python
exp = explanations[0]
median = exp.prediction['predict']    # point estimate (CPS median)
low    = exp.prediction['low']        # e.g., 5th percentile
high   = exp.prediction['high']       # e.g., 95th percentile
# Invariant (ADR-021 §4): low <= median <= high
```

**What drives the interval width:**
- Calibration set size: larger → tighter intervals.
- Input difficulty (Mondrian bins): harder instances → wider intervals.
- Percentile choice: (5, 95) is the default; (10, 90) is tighter.

---

## Mode 3: Thresholded Probabilistic Regression

**Use when:** you want the calibrated probability that the output exceeds (or
falls below) a specific threshold value.

```python
# Scalar threshold: P(y ≤ 50.0)
explanations = explainer.explain_factual(X_query, threshold=50.0)

# Two-sided threshold: P(40.0 < y ≤ 60.0) — probability of being in a window
explanations = explainer.explain_factual(X_query, threshold=(40.0, 60.0))

# Per-instance thresholds (one per row in X_query)
per_instance_thresholds = np.array([45.0, 55.0, 50.0])
explanations = explainer.explain_factual(X_query, threshold=per_instance_thresholds)
```

**How to read the output:**
```python
exp = explanations[0]
prob       = exp.prediction['predict']   # calibrated P(y ≤ threshold) — a probability
prob_low   = exp.prediction['low']       # lower bound of probability interval
prob_high  = exp.prediction['high']      # upper bound of probability interval
# Interval semantics = PROBABILITY MASS, not percentile band
# Invariant (ADR-021 §4): prob_low <= prob <= prob_high
```

**Key difference from Mode 2:**
- The `predict` value in Mode 3 is a **probability** (0–1), not a value in y-space.
- The interval bounds are probability bounds, not regression value bounds.
- Do NOT mix `low_high_percentiles` and `threshold` in the same call — they
  activate mutually exclusive code paths.

---

## Calibration Engine (Context — ADR-021)

| Mode | Engine | What is calibrated |
|---|---|---|
| Classification | Venn-Abers only | P(class=k) with interval |
| Regression (percentile) | CPS only | Percentile bands |
| Regression (threshold) | CPS → Venn-Abers | P(y ≤ t) with interval |

The threshold mode reuses one half of the calibration set for CPS, the other
half for fitting the Venn-Abers step. The result caches under
`explainer._interval_learner.current_y_threshold`.

---

## Choosing Interval Width

| Goal | Setting |
|---|---|
| Standard 90% interval | `low_high_percentiles=(5, 95)` (default) |
| Tighter 80% interval | `low_high_percentiles=(10, 90)` |
| Tight 50% IQR | `low_high_percentiles=(25, 75)` |
| Wider 98% interval | `low_high_percentiles=(1, 99)` |

Tighter intervals → more instances fall outside → use when you have large
calibration sets. Wider intervals → safer coverage guarantee.

---

## Per-Instance Adaptive Intervals (DifficultyEstimator)

For standard regression, a global interval width treats all test instances
identically. Adding a `DifficultyEstimator` from `crepes.extras` makes the
CPS (Conformal Predictive System) scale each interval by the predicted
difficulty σᵢ — easy instances get tighter intervals, hard ones get wider.

**Effect**: `set_difficulty_estimator` triggers a **full refit** of the
internal `ConformalPredictiveSystem` with `sigmas=`. The previous calibrator
is invalidated and rebuilt with the new difficulty values.

### Method 1 — pass at calibrate time (recommended for fresh setups)

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

### Method 2 — set (or change) after calibration

```python
from crepes.extras import DifficultyEstimator

# Add difficulty estimation post-calibration (rebuilds the CPS internally)
de = DifficultyEstimator().fit(X=X_prop_train, learner=regressor.learner, scaler=True)
regressor.set_difficulty_estimator(de)
```

**Requires**: `WrapCalibratedExplainer` must already be fitted and calibrated.

### Removing the DifficultyEstimator

```python
# Remove: resets to uniform σ=1 (flat intervals)
regressor.set_difficulty_estimator(None)

# Alternative: recalibrate without it
regressor.calibrate(x_cal, y_cal, feature_names=feature_names)
```

### Validation rules

- `DifficultyEstimator` must have `fitted=True` before passing to CE.
- Passing an unfitted instance raises `NotFittedError`.
- The estimator must expose an `apply(X) → np.ndarray` method.
- `scaler=True` is recommended (stabilises σ values to a reasonable range).

### Effect on interval semantics

| Setup | Interval source |
|---|---|
| No DifficultyEstimator | Global CPS — same width for all instances |
| With DifficultyEstimator | Mondrian-like CPS scaled by σᵢ — wider for hard, tighter for easy |

The difficulty estimator does **not** change how `low_high_percentiles` or
`threshold` are interpreted — it only scales the underlying conformal scores.

---

## Alternatives in Regression

```python
# Conformal alternatives: what changes push y beyond a threshold?
alternatives = explainer.explore_alternatives(X_query, threshold=50.0)

# Percentile alternatives: what changes move y until the interval is outside target?
alternatives = explainer.explore_alternatives(X_query, low_high_percentiles=(10, 90))
```

---

## Out of Scope

- Classification interval semantics (see `ce-factual-explain`).
- Building the pipeline (see `ce-pipeline-builder`).
- Persisting the calibrated regression model (see `ce-serializer-impl`).

## Evaluation Checklist

- [ ] Mode correctly identified (percentile vs threshold).
- [ ] `low_high_percentiles` and `threshold` not used simultaneously.
- [ ] Interval invariant `low ≤ predict ≤ high` verified.
- [ ] Correct semantic interpretation of `predict` stated (probability vs y-value).
- [ ] Calibration set size and expected interval width proportional.
- [ ] If DifficultyEstimator used: estimator is `fitted=True` before assignment.
- [ ] If DifficultyEstimator used: interval widths vary per instance (not uniform).
- [ ] `set_difficulty_estimator(None)` tested if removal path is needed.
