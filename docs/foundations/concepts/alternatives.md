# Alternative explanations and ensured filters

Alternative explanations show how feature changes may change predictions.

Semantics and non-guarantees are defined in
{doc}`calibrated_interval_semantics`.

## Generate alternatives

```python
alternatives = explainer.explore_alternatives(X_query)
alt0 = alternatives[0]
```

For probabilistic regression:

```python
alternatives = explainer.explore_alternatives(X_query, threshold=150.0)
```

## Ensured filters

The alternative object supports these filters:

- `super_explanations()`
- `semi_explanations()`
- `counter_explanations()`
- `ensured_explanations()`
- `pareto_explanations()`

## Plotting and conjunctions

```python
alt0.add_conjunctions(n_top_features=5, max_rule_size=2)
alt0.plot(style="triangular", show=False)
```

## Related pages

- {doc}`../../practitioner/playbooks/ensured-explanations`
- {doc}`../how-to/interpret_explanations`
- {doc}`guarded_explanations`

Entry-point tier: Tier 3.

