# Probabilistic & interval regression

Probabilistic regression extends calibrated explanations beyond point estimates.
It pairs calibrated probabilities ("what is the chance the outcome exceeds my
threshold?") with calibrated intervals that describe where the numeric target is
likely to fall. Use this guide alongside the practitioner quickstarts and
notebooks to keep probabilistic and interval narratives aligned.

> ℹ️ **Terminology note:** You may encounter both "probabilistic regression" and
> "thresholded regression" in documentation and code. These terms are synonymous
> and refer to the same feature: regression with calibrated probability predictions
> and threshold-based decision boundaries. See the [ADR-021 Terminology section](../../improvement/adrs/ADR-021-calibrated-interval-semantics.md)
> for terminology guidance.

## Two Regression Modes

Calibrated Explanations supports **two distinct regression modes** that answer different questions:

| Mode | API Signal | Returns | Question Answered |
|------|-----------|---------|-------------------|
| **Conformal Interval Regression** | `predict(x, uq_interval=True, low_high_percentiles=...)` | Point estimate + CPS interval | "Where will y fall?" |
| **Probabilistic Regression** | `predict_proba(x, threshold=...)` | Probability P(y ≤ t) or P(low < y ≤ high) | "What's the probability y meets my threshold?" |

### Key Semantic Difference

* **Interval regression** answers: "Give me a range where y is likely to be" (conformal percentiles)
* **Probabilistic regression** answers: "What's the probability y satisfies my condition?" (calibrated probability via CPS + Venn-Abers)

Both modes can be used on the same explainer; the `threshold` parameter activates probabilistic mode.

```{admonition} Common Confusion
:class: warning

These are **not** the same thing:
* Interval regression returns a **numeric range** (e.g., [120, 180])
* Probabilistic regression returns a **probability** (e.g., P(y ≤ 150) = 0.73)

The interval tells you *where* the value will be; the probability tells you *how likely* a condition is met.
```

## Calibrated probabilities and intervals

1. Start with the {doc}`../../get-started/quickstart_regression` flow or the
   companion {download}`probabilistic regression notebook
   <https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/core_demos/demo_probabilistic_regression.ipynb>`.
2. Call :meth:`calibrated_explanations.WrapCalibratedExplainer.predict` with
   ``threshold`` and ``uq_interval=True`` to obtain the calibrated probability
   and its `(low, high)` interval bounds.
3. Treat the interval as the decision boundary for interval regression use
   cases. The probability tells you how likely it is that the target is at or below
   the threshold (for scalar thresholds) or within the specified interval (for tuple thresholds).

```python
probabilities, probability_interval = explainer.predict_proba(
    x_test[:1],
    threshold=150,
    uq_interval=True,
)
low, high = probability_interval
print(f"Calibrated probability: {probabilities[0, 1]:.3f}")
print(f"Interval bounds: {low[0]:.3f} – {high[0]:.3f}")
```

You can also ask for the probability that the true value lies inside a **user-defined interval** by passing a tuple threshold:

```python
probabilities, probability_interval = explainer.predict_proba(
   x_test[:1],
   threshold=(120, 180),
   uq_interval=True,
)
low, high = probability_interval
print(f"P(120 < y <= 180): {probabilities[0]:.3f}")
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
    x_test[:2],
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
