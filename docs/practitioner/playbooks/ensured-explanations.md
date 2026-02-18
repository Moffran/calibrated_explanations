# Ensured Explanations Playbook

Ensured explanations actively reduce epistemic uncertainty in alternative explanations, helping you find more reliable intervention recommendations.

> **Audience:** Practitioner

```{admonition} Guarantees & Assumptions
:class: important

- Ensured explanations are a *filtering and ranking framework* over existing
    alternative rules.
- Ensured filtering prefers alternatives with **narrower uncertainty
    intervals** than the base prediction interval.
- The calibration guarantees behind uncertainty intervals require that the
    calibration set and your deployment queries are exchangeable / distribution
    matched.
```

## What are Ensured Explanations?

Standard alternative explanations may have wide uncertainty intervals when the calibration data doesn't cover the proposed scenario well. **Ensured explanations** use strategies to:

1. Prioritize alternatives with lower epistemic uncertainty
2. Flag alternatives where confidence is insufficient
3. Guide users toward more reliable recommendations

## When to Use

* **High-stakes decisions**: Medical diagnosis, loan approval, safety-critical systems
* **Audit requirements**: Need to justify confidence in recommendations
* **Uncertainty-sensitive domains**: When acting on uncertain alternatives is risky
* **Resource-constrained interventions**: When you can only pursue a few alternatives

## Understanding Epistemic vs Aleatoric Uncertainty

| Type | Source | Can Be Reduced? | Example |
| :--- | :--- | :--- | :--- |
| **Epistemic** | Limited data/knowledge | Yes (more data) | "We haven't seen this scenario before" |
| **Aleatoric** | Inherent randomness | No | "This outcome varies naturally" |

Ensured explanations focus on **reducing epistemic uncertainty** by identifying alternatives where the calibration data provides strong support.

## Usage Pattern

### Core filtering API

The ensured framework is exposed directly on the alternative explanation objects.

```python
alternatives = explainer.explore_alternatives(x_test)
alt0 = alternatives[0]

# Ensured: alternatives with narrower uncertainty intervals than the base
ensured = alt0.ensured_explanations()

# Super: alternatives reinforcing the base prediction
super_alts = alt0.super_explanations()

# Semi/Counter: alternatives opposing the prediction
semi_alts = alt0.semi_explanations()
counter_alts = alt0.counter_explanations()

# Pareto: output-envelope frontier over (output, uncertainty width)
pareto_alts = alt0.pareto_explanations()

# Combine: ensured super-explanations
ensured_super = alt0.super_explanations(only_ensured=True)
```

### Ranking controls (rnk_metric / rnk_weight / filter_top)

Triangular plots are commonly used together with ensured ranking:

```python
# Use ensured-style triangular view (alias for style="triangular")
alt0.plot(
    style="ensured",
    rnk_metric="ensured",
    rnk_weight=0.7,
    filter_top=10,
    show=False,
)
```

Notes:

- ``rnk_metric="ensured"`` combines output and uncertainty interval width.
- ``rnk_weight`` shifts the trade-off between output and uncertainty.
- Use ``show=False`` in headless environments.

## Setting Uncertainty Thresholds

Choose thresholds based on your domain requirements:

| Domain | Suggested Threshold | Rationale |
| :--- | :--- | :--- |
| Medical diagnosis | 0.05–0.10 | High stakes require narrow intervals |
| Financial decisions | 0.10–0.15 | Moderate risk tolerance |
| Exploratory analysis | 0.20–0.30 | Wider intervals acceptable for insights |

```{admonition} Threshold Selection
:class: tip

Start with a conservative threshold (e.g., 0.10) and observe how many alternatives pass. If too few pass:

1. Consider whether you need more calibration data
2. Relax the threshold slightly
3. Accept that some scenarios genuinely have high uncertainty
```

## Best Practices

### 1. Document Your Confidence Criteria

```python
# Make your filtering criteria explicit and reproducible
ENSURED_CONFIG = {
    "max_interval_width": 0.12,
    "min_calibration_support": 10,  # if available
    "domain": "credit_approval"
}

def is_ensured(alternative, config=ENSURED_CONFIG):
    width = alternative.uncertainty_high - alternative.uncertainty_low
    return width < config["max_interval_width"]
```

### 2. Report Both Ensured and All Alternatives

```python
all_alternatives = alternatives[0]
ensured_alternatives = [a for a in all_alternatives if is_ensured(a)]

print(f"Total alternatives: {len(all_alternatives)}")
print(f"Ensured alternatives: {len(ensured_alternatives)}")
print(f"Ensured ratio: {len(ensured_alternatives)/len(all_alternatives):.1%}")
```

### 3. Consider Augmenting Calibration Data

If important scenarios consistently fail the ensured threshold:

* Collect more calibration data for those regions
* Use Mondrian calibration to improve group-specific estimates
* Document which scenarios remain uncertain

## Interpreting Ensured Results

| Outcome | Interpretation | Action |
| :--- | :--- | :--- |
| Many ensured alternatives | Good calibration coverage | Proceed with confidence |
| Few ensured alternatives | Sparse calibration regions | Collect more data or accept uncertainty |
| No ensured alternatives | Novel scenario | Defer to expert or flag for review |

## Research Background

This capability is documented in:

> Löfström, H., Löfström, T., and Hallberg Szabadvary, J. (2024).
> [Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions](https://arxiv.org/abs/2410.05479).
> arXiv preprint arXiv:2410.05479.

## Cross-References

* {doc}`../../tasks/capabilities` - Full capability manifest
* {doc}`../../foundations/concepts/alternatives` - Alternative explanations concept
* {doc}`decision-policies` - Decision policy patterns using uncertainty
* {doc}`../../citing` - Citation information

## Regression support notes

Ensured explanations support:

- **Classification** (binary/multiclass) via calibrated probabilities.
- **Probabilistic regression** via calibrated probabilities for events like
    $\Pr(y \le t)$ (pass ``threshold=...`` to ``explore_alternatives``).
- **Plain regression** (numeric outputs):

    - ``super_explanations`` keeps higher predicted outputs than the base.
    - ``semi_explanations`` and ``counter_explanations`` keep lower predicted
        outputs than the base.
    - ``ensured_explanations`` and ``pareto_explanations`` work the same way as
        in classification, filtering by uncertainty interval width and trade-offs.

Use the global plot to sanity-check calibration and uncertainty behaviour over
a dataset:

```python
explainer.plot(x_test, y_test, show=False)
explainer.plot(x_test, y_test, threshold=150.0, show=False)
```
