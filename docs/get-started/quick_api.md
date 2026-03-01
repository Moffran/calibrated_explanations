# Quick API (cheat sheet)

This page is a minimal reference for the most common `WrapCalibratedExplainer` workflows.
For the full repository cheat sheet, see
[QUICK_API.md on GitHub](https://github.com/Moffran/calibrated_explanations/blob/main/QUICK_API.md).

```{admonition} Guarantees & assumptions
:class: important
You should `fit(...)` on a **proper training set** and `calibrate(...)` on a **held-out calibration set**.
Using the same data for both steps can invalidate calibration guarantees.
```

## Core methods (all tasks)

```python
# point prediction (sklearn-like)
pred = explainer.predict(X_query)

# point + calibrated uncertainty interval (task-dependent)
pred, (low, high) = explainer.predict(X_query, uq_interval=True)

# probabilities (sklearn-like)
probs = explainer.predict_proba(X_query)

# probabilities + calibrated uncertainty interval (task-dependent)
probs, (low, high) = explainer.predict_proba(X_query, uq_interval=True)

# factual explanations (rules + bounds)
factual = explainer.explain_factual(X_query)

# alternative explanations ("what would change it?")
alternatives = explainer.explore_alternatives(X_query)

# preprocessor mapping snapshot (JSON-safe primitives only)
mapping = explainer.export_preprocessor_mapping()
if mapping is not None:
    explainer.import_preprocessor_mapping(mapping)
```

## Classification (binary / multiclass)

```python
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(RandomForestClassifier(random_state=0))
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

factual = explainer.explain_factual(x_test[:1])
print(factual[0])
```

For a true multiclass explanation (one explanation per class), enable multi-label mode:

```python
multi = explainer.explain_factual(x_test[:1], multi_labels_enabled=True)
```

`multi_labels_enabled=True` is intended for all-classes analysis. In this mode:
- collection operations and reject/feature-filter options are supported;
- multiclass JSON export/import is supported;
- binary datasets are accepted but emit a compatibility warning (default mode is usually preferable).

### Calibrated probability + interval

```python
probs = explainer.predict_proba(X_sample)
probs, (low, high) = explainer.predict_proba(X_sample, uq_interval=True)
print("P(class=1) =", probs[0, 1], "interval =", low[0, 1], high[0, 1])
```

## Regression (conformal interval regression via CPS)

Regression in Calibrated Explanations is conformal interval regression via CPS.
Use `low_high_percentiles=(a, b)` to control which CPS interval is returned.

```python
from sklearn.linear_model import BayesianRidge
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(BayesianRidge())
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, mode="regression", feature_names=feature_names)

pred, (low, high) = explainer.predict(
    X_sample,
    uq_interval=True,
    low_high_percentiles=(5, 95),
)
print(pred[0], low[0], high[0])
```

## Probabilistic regression (thresholded probability for y)

Probabilistic regression requires a `threshold` and operate directly on the regression model's output, allowing you to query calibrated probabilities for any threshold of interest without retraining. The returned probability reflects the model's confidence that the true value of `y` falls below the specified threshold, while the accompanying interval quantifies the uncertainty around this estimate:

- `threshold=t` queries $P(y \le t)$
- `threshold=(low, high)` queries $P(low < y \le high)$

```python
p = explainer.predict(X_sample, threshold=120.0) # returns the labels `y <= 120.0` or `y > 120.0`
p = explainer.predict_proba(X_sample, threshold=120.0)
p, (plo, phi) = explainer.predict_proba(X_sample, uq_interval=True, threshold=120.0)
print("P(y <= 120) =", p[0], "interval =", plo[0], phi[0])
```
