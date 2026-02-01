# Conformal Interval Regression (CPS)

Regression in Calibrated Explanations is **conformal interval regression** implemented via **Conformal Predictive Systems (CPS)**. 

- **Canonical semantics:** Point regression + calibrated uncertainty intervals = conformal regression.
- **Interval control:** The specific interval width is controlled by `low_high_percentiles`.

## Supported signatures

| Method | Description |
| :--- | :--- |
| `predict(x)` | Point regression estimate |
| `predict(x, uq_interval=True, ...)` | Point estimate + CPS parameterised interval |
| `explain_factual(x, ...)` | Factual explanation with CPS intervals |
| `explore_alternatives(x, ...)` | Alternative explanations with CPS intervals |

## Controlling the interval: `low_high_percentiles`

The `low_high_percentiles` parameter (tuple `(low, high)`) governs the CPS interval.

* Default: `(5, 95)` → 90% central interval.
* One-sided: `(-np.inf, 95)` or `(5, np.inf)`.

## Examples

### 1. Point prediction + 90% conformal interval

```python
# Returns median, low (5th percentile), and high (95th percentile)
prediction, (low, high) = explainer.predict(
    X_test, 
    uq_interval=True, 
    low_high_percentiles=(5, 95)
)
print(f"Prediction: {prediction[0]} Interval: {low[0]} – {high[0]}")
```

### 2. Explanation with specific interval settings

You can request explanations with arbitrary confidence levels by strictly passing the percentiles:

```python
# Explain with a 50% central interval (25th - 75th percentiles)
explanation = explainer.explain_factual(
    X_test, 
    low_high_percentiles=(25, 75)
)
```

## Key semantics

* **Prediction Interval:** The interval returned by `predict(..., uq_interval=True)` is the conformal interval derived from the CPS.
* **Rule Intervals:** The uncertainty bounds on feature weights in `explain_factual` rules are also derived from the underlying CPS calibration.
