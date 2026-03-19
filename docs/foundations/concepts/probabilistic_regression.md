# Probabilistic and interval regression

This page explains the two regression modes and how they differ.

For full guarantees, assumptions, and non-guarantees, use
{doc}`calibrated_interval_semantics`.

## Mode routing

| Mode | API signal | Output | Primary question |
| --- | --- | --- | --- |
| Percentile or interval regression | `predict(..., uq_interval=True, low_high_percentiles=...)` | Numeric prediction with CPS percentile interval | Where will `y` fall? |
| Probabilistic or thresholded regression | `predict_proba(..., threshold=...)` | Calibrated event probability with interval | How likely is an event on `y`? |

## Examples

```python
pred, (low, high) = explainer.predict(
    X,
    uq_interval=True,
    low_high_percentiles=(5, 95),
)

p, (plo, phi) = explainer.predict_proba(
    X,
    threshold=150,
    uq_interval=True,
)
```

Interval mode and probabilistic mode can be used on the same calibrated
explainer. The `threshold` argument selects probabilistic mode.

## Related pages

- {doc}`../../get-started/quickstart_regression`
- {doc}`../how-to/interpret_explanations`
- {doc}`../../tasks/probabilistic_regression`

Entry-point tier: Tier 3.

