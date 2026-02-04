# Probabilistic (Thresholded) Regression

Probabilistic regression allows you to ask probability questions about a real-valued target (e.g., "What is the probability the house price <= 5 million?").

This is technically implemented as **Thresholded Regression**: you supply a `threshold`, and CE returns the calibrated probability that the outcome satisfies that threshold condition.

## Supported signatures

| Method | Description |
| :--- | :--- |
| `predict_proba(x, threshold=t)` | Threshold probability P(y <= t) |
| `predict_proba(x, threshold=(low, high))` | Interval event probability P(low < y <= high) |
| `predict_proba(x, uq_interval=True, ...)` | Probability + uncertainty interval |
| `explain_factual(x, threshold=...)` | Explains why the specific threshold condition is met |

> Note: Returns P(y <= t) for scalar thresholds and P(low < y <= high) for interval thresholds.

## Examples

### 1. Threshold probability (Scalar threshold)

```python
# Probability that y is at or below 150
probs, (low_p, high_p) = explainer.predict_proba(
    x_test,
    threshold=150,
    uq_interval=True
)
print(f"P(y <= 150): {probs[0]}  Confidence: [{low_p[0]}, {high_p[0]}]")
```

### 2. Interval event probability (Range threshold)

Calculate the probability that the true value lies **inside** a specific user-defined range.

```python
# Probability that y is between 100 and 200
probs, (low_p, high_p) = explainer.predict_proba(
    x_test,
    threshold=(100, 200),
    uq_interval=True
)
print(f"P(100 < y <= 200): {probs[0]}")
```

### 3. Explaining the probability

You can generate feature rules explaining exactly why the probability is high or low for your chosen threshold.

```python
# Why is P(y <= 150) so high (or low)?
explanation = explainer.explain_factual(
    x_test,
    threshold=150,
)
```

## Key parameters

- **`threshold`**:
  - Scalar `t`: treated as a binary classification boundary.
  - Tuple `(low, high)`: treated as an interval containment query.
- **`uq_interval`**: Returns the uncertainty bound on the **probability estimate** itself (aleatoric + epistemic uncertainty on the score).
