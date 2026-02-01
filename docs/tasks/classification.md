# Classification (Binary & Multiclass)

Calibrated Explanations wraps standard scikit-learn classifiers to produce uncertainty-aware predictions and explanations. 

## Supported signatures

| Method | Description |
| :--- | :--- |
| `predict(x)` | Point prediction (class label) |
| `predict_proba(x)` | Calibrated probabilities (n_samples, n_classes) |
| `predict_proba(x, uq_interval=True)` | Calibrated probabilities **plus** uncertainty intervals |
| `explain_factual(x)` | Factual explanation (rules + uncertainty) |
| `explore_alternatives(x)` | Alternative explanations (counterfactuals) |

> ℹ️ **Note:** `predict()` and `predict_proba()` match the scikit-learn API but return **calibrated** values.

## Examples

### 1. Calibrated probabilities with uncertainty

```python
# Predict probabilities for the positive class (binary) or all classes (multiclass)
probs, (low, high) = explainer.predict_proba(X_test, uq_interval=True)

print(f"Probability: {probs[0, 1]:.3f} (Interval: {low[0, 1]:.3f} – {high[0, 1]:.3f})")
```

### 2. Factual explanation

Returns the rules explaining *why* the model made this prediction, with epistemic and aleatoric uncertainty bounds on the feature weights.

```python
explanation = explainer.explain_factual(X_test)
```

### 3. Explore alternatives

Finds what feature changes would be necessary to flip the prediction or increase confidence.

```python
alternatives = explainer.explore_alternatives(X_test)
```

## Key parameters

- **`bins`**: Supply Mondrian categories/bins for conditional calibration.
- **`uq_interval`**: Set to `True` to receive uncertainty intervals (tuples of lower/upper bounds).
