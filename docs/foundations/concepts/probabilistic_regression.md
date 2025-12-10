# Probabilistic & interval regression

Probabilistic regression extends calibrated explanations beyond point estimates.
It pairs calibrated probabilities ("what is the chance the outcome exceeds my
threshold?") with calibrated intervals that describe where the numeric target is
likely to fall. Use this guide alongside the practitioner quickstarts and
notebooks to keep probabilistic and interval narratives aligned.

> ℹ️ **Terminology note:** You may encounter both "probabilistic regression" and
> "thresholded regression" in documentation and code. These terms are synonymous
> and refer to the same feature: regression with calibrated probability predictions
> and threshold-based decision boundaries. See the [ADR-021 Terminology section](https://github.com/Moffran/calibrated_explanations/blob/main/docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md)
> for terminology guidance.

## Calibrated probabilities and intervals

1. Start with the {doc}`../../get-started/quickstart_regression` flow or the
   companion {download}`probabilistic regression notebook
   <https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/demo_probabilistic_regression.ipynb>`.
2. Call :meth:`calibrated_explanations.WrapCalibratedExplainer.predict` with
   ``threshold`` and ``uq_interval=True`` to obtain the calibrated probability
   and its `(low, high)` interval bounds.
3. Treat the interval as the decision boundary for interval regression use
   cases. The probability tells you how likely it is that the target lands above
   (or within) the threshold.

```python
probabilities, probability_interval = explainer.predict_proba(
    X_test[:1],
    threshold=150,
    uq_interval=True,
)
low, high = probability_interval
print(f"Calibrated probability: {probabilities[0, 1]:.3f}")
print(f"Interval bounds: {low[0]:.3f} – {high[0]:.3f}")
```

> 🧭 **Interpretation shortcut:** Revisit the
> {doc}`../how-to/interpret_explanations` guide for screenshots that map
> calibrated values, threshold metadata, and interval semantics back to the
> quickstarts and notebooks.

## Alternative scenarios with triangular plots

Probabilistic regression shares the same alternative exploration workflow as the
classification quickstarts.

```python
alternatives = explainer.explore_alternatives(
    X_test[:2],
    threshold=150,
)
```

Use the triangular view to compare the base prediction (red) against calibrated
alternatives (blue). The rhombus overlay highlights scenarios whose calibrated
interval still crosses the decision boundary, signalling that more calibration
data may be required before deployment.

## Research and citations

- {doc}`/citing` aggregates the peer-reviewed papers covering binary,
  multiclass, probabilistic, and interval regression.
- The probabilistic regression journal article (Machine Learning, 2025) describes
  the conformal uncertainty intervals referenced throughout this guide.
- {doc}`../../researcher/advanced/theory_and_literature` lists benchmarks and evaluation
  harnesses so you can compare your own workloads against the published results.
