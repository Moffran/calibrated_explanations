# Calibrated Explanations — Minimal Working Examples (use `WrapCalibratedExplainer`)

## Core methods (all tasks)

```python
# All tasks support:
pred = explainer.predict(X_query)                         # point prediction (sklearn-like)
pred, (lo, hi) = explainer.predict(X_query, uq_interval=True)  # point + uncertainty interval
factual = explainer.explain_factual(X_query)              # factual explanations (rules + bounds)
alts = explainer.explore_alternatives(X_query)            # alternative explanations (what would change it?)
```

Reject integration (policy-first): When a `RejectPolicy` other than `NONE` is passed to prediction/explanation APIs (e.g., `explain_factual(..., reject_policy=...)`), the call returns a `RejectResult` envelope. The envelope's `prediction` field mirrors the invoked method's legacy payload (including regression UQ tuples `(proba, (low, high))`), and `metadata` exposes per-instance breakdown keys: `ambiguity_mask`, `uncertainty_mask`, `prediction_set_size`, and `epsilon`. The envelope's `explanation` field contains the explanation object or `None` if no explanation was produced.

Optional parameters (`[, ...]`) across methods:
- `bins=...` for conditional calibration
- `low_high_percentiles=(a, b)` for **CPS conformal interval regression**
- `threshold=...` for **probabilistic regression**

---

## Classification (binary / multiclass)
```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer
explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, feature_names=feature_names)
factual = explainer.explain_factual(X_test[:1])
print(factual[0])
```

### Calibrated probability + interval (classification)
```python
probs = explainer.predict_proba(X_sample)  # calibrated probabilities (sklearn-like)
probs, (low, high) = explainer.predict_proba(X_sample, uq_interval=True)  # + uncertainty bounds
print("P(class=1) =", probs[0, 1], "interval =", low[0, 1], high[0, 1])
```

---

## Conformal interval regression (CPS)  ← CE "regression"

Regression in Calibrated Explanations is **conformal interval regression** via **Conformal Predictive Systems (CPS)**:
point regression + calibrated uncertainty intervals = (conformal) interval regression. The CPS interval is
controlled using `low_high_percentiles`.

```python
from sklearn.linear_model import BayesianRidge
from calibrated_explanations import WrapCalibratedExplainer
explainer = WrapCalibratedExplainer(BayesianRidge())
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, mode='regression', feature_names=feature_names)

# point + CPS interval (arbitrary intervals via percentiles)
pred, (low, high) = explainer.predict(
    X_sample,
    uq_interval=True,
    low_high_percentiles=(5, 95),
)
print(pred[0], low[0], high[0])

# you can also request CPS-controlled intervals inside explanations
factual = explainer.explain_factual(X_sample, low_high_percentiles=(5, 95))
```

---

## Probabilistic regression (thresholded probabilities for y)

Probabilistic regression requires a `threshold`:
- `threshold=t` queries exceedance probability (e.g., P(y ≥ t) depending on formulation)
- `threshold=(low, high)` queries interval event probability P(true y ∈ [low, high])

```python
# exceedance probability (+ optional uncertainty bounds)
p = explainer.predict_proba(X_sample, threshold=120.0)
p, (plo, phi) = explainer.predict_proba(X_sample, uq_interval=True, threshold=120.0)
print("P(y exceeds 120) =", p[0], "interval =", plo[0], phi[0])

# probability that true value lies inside a user-defined interval
p_in = explainer.predict_proba(X_sample, threshold=(100.0, 140.0))
p_in, (ilo, ihi) = explainer.predict_proba(X_sample, uq_interval=True, threshold=(100.0, 140.0))
print("P(100 <= y <= 140) =", p_in[0], "interval =", ilo[0], ihi[0])
```

---

## Alternative explanations (core capability)

```python
# returns alternative rule table(s) showing what would change the decision
alternatives = explainer.explore_alternatives(X_test[:3])
```

Notes: Keep examples deterministic by setting `random_state` where possible. All quickstarts use
`WrapCalibratedExplainer` and assume `fit(...)` followed by `calibrate(...)`. Examples in
`examples/use_cases/` emit JSON summaries validated by CI.
# Calibrated Explanations — Minimal Working Examples (use `WrapCalibratedExplainer`)

## Classification (3–6 lines)
```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer
explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, feature_names=feature_names)
factual = explainer.explain_factual(X_test[:1])
print(factual[0])
```

## Regression

```python
from sklearn.linear_model import BayesianRidge
from calibrated_explanations import WrapCalibratedExplainer
explainer = WrapCalibratedExplainer(BayesianRidge())
explainer.fit(X_proper, y_proper)
explainer.calibrate(X_cal, y_cal, mode='regression', feature_names=feature_names)
explanation = explainer.explain_factual(X_test)
```

## Alternative explanations

```python
# returns alternative rule table(s) showing what would change the decision
alternatives = explainer.explore_alternatives(X_test[:3])
```

## Get calibrated probability + interval

```python
probs, (low, high) = explainer.predict_proba(X_sample, uq_interval=True)
print("P(class=1) =", probs[0,1], "interval =", low[0,1], high[0,1])
```

Notes: Keep examples deterministic by setting `random_state` where possible. All quickstarts use
`WrapCalibratedExplainer` and assume `fit(...)` followed by `calibrate(...)`. Examples in
`examples/use_cases/` emit JSON summaries validated by CI.
