# Alternative explanations and the ensured framework

Alternative explanations answer "what would need to change for the model output
to move?" They are returned as **rule candidates** with calibrated predictions
and **uncertainty intervals**, letting you compare different interventions while
keeping uncertainty explicit.

> **Audience:** Practitioner

```{admonition} Guarantees & Assumptions
:class: important

- Alternative rules are evaluated through the same calibrated pipeline as the
   base prediction.
- Uncertainty intervals are only meaningful under the usual calibration
   assumptions (notably: calibration and deployment data are exchangeable / drawn
   from matching distributions).
- Ensured filtering reduces *epistemic uncertainty* in the sense of preferring
   alternatives with **narrower uncertainty intervals**, but it does not make
   uncertainty disappear.

See {doc}`../../tasks/capabilities` for the capability overview and the formal
interval semantics in ADR-021.
```

## Getting alternatives

Use :meth:`calibrated_explanations.WrapCalibratedExplainer.explore_alternatives`
to generate alternatives for one or more instances.

```python
alternatives = explainer.explore_alternatives(X_query)

# Per-instance alternative explanation
alt0 = alternatives[0]
```

For **probabilistic regression**, pass a threshold to define the event
probability (for example, $\Pr(y \le t)$):

```python
alternatives = explainer.explore_alternatives(X_query, threshold=150.0)
```

## The ensured framework (five filters)

The ensured framework provides five user-facing filters over a set of
alternatives:

- :meth:`calibrated_explanations.explanations.explanation.AlternativeExplanation.super_explanations`
- :meth:`calibrated_explanations.explanations.explanation.AlternativeExplanation.semi_explanations`
- :meth:`calibrated_explanations.explanations.explanation.AlternativeExplanation.counter_explanations`
- :meth:`calibrated_explanations.explanations.explanation.AlternativeExplanation.ensured_explanations`
- :meth:`calibrated_explanations.explanations.explanation.AlternativeExplanation.pareto_explanations`

Each method returns a filtered view of the same underlying alternative rules.

### Potential alternatives

All five methods accept ``include_potential=...``. *Potential* alternatives are
those that sit on/near the decision boundary, meaning the uncertainty interval
still covers a different decision than the point estimate.

## Semantics across tasks

The meaning of "super / semi / counter / potential" depends on the output type.

| Filter | Classification (boundary at 0.5) | Probabilistic regression (boundary at 0.5 for $\Pr(y \le t)$) | Plain regression (reference is base prediction) |
|---|---|---|---|
| **super** | Moves further into the predicted class (away from 0.5) | Moves further into the predicted event vs non-event (away from 0.5) | Higher predicted output than the base prediction |
| **semi** | Same side of 0.5 as the base, but closer to 0.5 | Same as classification | Lower predicted output than the base prediction |
| **counter** | Crosses 0.5 (opposes the base decision) | Crosses 0.5 (opposes the base decision) | Lower predicted output than the base prediction |
| **potential** | Uncertainty interval spans 0.5 | Uncertainty interval spans 0.5 | Uncertainty interval covers the base prediction |
| **ensured** | Uncertainty interval width is no wider than the base interval | Same as classification | Same as classification |
| **pareto** | Output/uncertainty frontier (non-dominated by lower uncertainty at same output) | Same as classification | Same as classification |

Notes:

- In plain regression, ``semi`` and ``counter`` currently have the same
   directionality (both select lower outputs). Use ``ensured`` and ``pareto`` to
   control uncertainty and trade-offs.

## Minimal API examples

```python
alternatives = explainer.explore_alternatives(X_query)
alt0 = alternatives[0]

ensured = alt0.ensured_explanations()
super_alts = alt0.super_explanations()
semi_alts = alt0.semi_explanations()
counter_alts = alt0.counter_explanations()
pareto = alt0.pareto_explanations() # by uncertainty

# Pareto by rule complexity (useful when conjunctions are present)
pareto_simple = alt0.pareto_explanations(pareto_cost="rule_size")

# Combine: ensured super-explanations
ensured_super = alt0.super_explanations(only_ensured=True)
```

### Ranking controls

When plotting alternatives, you can control how rules are ranked:

- ``rnk_metric="ensured"`` ranks by a combined score of predicted output and
   uncertainty interval width.
- ``rnk_weight`` controls the trade-off between output and uncertainty in the
   ensured ranking.
- ``filter_top`` limits how many rules are shown.

## Conjunctions

If you need multi-feature interventions, call
:meth:`calibrated_explanations.explanations.explanation.AlternativeExplanation.add_conjunctions`
to create conjunctive rules.

```python
alt0 = alternatives[0].add_conjunctions(n_top_features=5, max_rule_size=2)
```

## Triangular plots ("ensured" view)

Use the triangular visualization to compare the base prediction (red) against
selected alternatives (blue) in terms of **prediction vs uncertainty interval
width**.

```python
# Equivalent styles
alt0.plot(style="triangular", show=False)
alt0.plot(style="ensured", show=False)
```

## Global plots

The global plot shows calibration and uncertainty behaviour over a batch of
instances:

```python
explainer.plot(X_test, y_test, show=False)

# Probabilistic regression global view
explainer.plot(X_test, y_test, threshold=150.0, show=False)
```

## See also

```{admonition} Guarded alternatives
:class: seealso

For an in-distribution-aware variant that filters out implausible perturbations
before generating alternatives, see {doc}`guarded_explanations`. Guarded
alternatives use conformal anomaly detection to ensure every suggested
intervention is plausible given the calibration data.
```

## Cross-references

- Playbook: {doc}`../../practitioner/playbooks/ensured-explanations`
- Interpretation guide: {doc}`../how-to/interpret_explanations`
- Capability manifest: {doc}`../../tasks/capabilities`
- Guarded explanations: {doc}`guarded_explanations`
