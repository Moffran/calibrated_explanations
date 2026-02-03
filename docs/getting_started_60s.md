# 60-second decision tree

Use this quick path to get calibrated explanations running fast.

## 1) Do you have a scikit-learn compatible model?
- **Yes** → continue.
- **No** → wrap or convert your model so it exposes `fit`, `predict`, and/or `predict_proba`.

## 2) Do you have a calibration split?
- **Yes** → continue.
- **No** → split your training data: `x_proper, x_cal, y_proper, y_cal = train_test_split(...)`.

## 3) Pick your task (and the control knob)

All tasks support:
- `predict(x[, ...])` and `predict(x, uq_interval=True[, ...])`
- `explain_factual(x[, ...])` and `explore_alternatives(x[, ...])`

### Classification (binary / multiclass)
- Calibrated probabilities: `predict_proba(x[, ...])`
- Probabilities + UQ: `predict_proba(x, uq_interval=True[, ...])`

### Conformal interval regression (CPS)  ← CE "regression"
Regression in Calibrated Explanations is **conformal interval regression** via **Conformal Predictive Systems (CPS)**:
point regression + calibrated uncertainty intervals = (conformal) interval regression.
- Point + CPS interval: `predict(x, uq_interval=True, low_high_percentiles=(5, 95)[, ...])`
- You can also request CPS-controlled intervals from explanations:
	`explain_factual(x, low_high_percentiles=(5, 95)[, ...])`

### Probabilistic regression (thresholded probabilities for y)
Probabilistic regression answers probability queries for a real-valued target by assigning a `threshold`:
- Exceedance probability: `predict_proba(x, threshold=t[, ...])`
- Exceedance probability + UQ: `predict_proba(x, uq_interval=True, threshold=t[, ...])`
- Interval event probability: `predict_proba(x, threshold=(low, high)[, ...])` → P(true value ∈ [low, high])

Reject policy note: You can opt into reject-aware behavior per-call via the `reject_policy` parameter (e.g., `explain_factual(X, reject_policy=RejectPolicy.PREDICT_AND_FLAG)`). When a non-`NONE` policy is active the API returns a `RejectResult` envelope whose `prediction` field mirrors the legacy return shape (including regression UQ tuples like `(proba, (low, high))`). Per-instance breakdowns are available in `RejectResult.metadata` under `ambiguity_mask`, `novelty_mask`, `prediction_set_size`, and `epsilon`. The `explanation` field contains the explanation object or `None` if no explanation was produced.

## 4) Run the minimal CE-first flow

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)
explanations = explainer.explain_factual(X_query)
```

## 5) Need alternatives or uncertainty intervals?
- Alternatives: `explainer.explore_alternatives(X_query)`
- Probabilities + intervals: `explainer.predict_proba(X_query, uq_interval=True)`

Return both point estimates and intervals, and include the rule table for traceability.
