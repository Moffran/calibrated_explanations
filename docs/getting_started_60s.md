# 60-second decision tree

Use this page to pick the correct first quickstart.

## 1. Model compatibility

Need a scikit-learn compatible estimator with `fit` and prediction methods.

## 2. Calibration split

Need a held-out calibration split: `x_cal, y_cal`.

## 3. Choose mode

- Classification: {doc}`get-started/quickstart_classification`
- Percentile or interval regression: {doc}`get-started/quickstart_regression`
- Probabilistic or thresholded regression: {doc}`get-started/quickstart_regression`
- Guarded explanations: {doc}`get-started/quickstart_guarded`

Semantics are mode-specific. Use
{doc}`foundations/concepts/calibrated_interval_semantics`.

## 4. Minimal flow

```python
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)
explanations = explainer.explain_factual(X_query)
```

Entry-point tier: Tier 1.

