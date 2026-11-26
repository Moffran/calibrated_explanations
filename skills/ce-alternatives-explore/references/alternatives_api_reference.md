# Alternatives API Reference

## The Ensured Framework — Five Filters

Each filter returns a new (or mutated) view of the same alternatives,
selecting rules by their relationship to the base prediction.

### Semantic table

| Filter | Classification / probabilistic regression | Plain regression |
|---|---|---|
| **super** | Further into the predicted class (away from 0.5) | Higher output than base |
| **semi** | Same side of 0.5, but closer to 0.5 | Lower output than base |
| **counter** | Crosses 0.5 (opposes base decision) | Lower output than base |
| **ensured** | Uncertainty interval is no wider than the base interval | Same |
| **pareto** | Output/uncertainty Pareto frontier (non-dominated) | Same |
| **potential** | `include_potential=True`: interval spans the decision boundary | Interval covers base prediction |

> In plain regression, `semi` and `counter` currently select the same direction (lower outputs).

### API — instance level (AlternativeExplanation)

```python
alt0 = alternatives[0]

# Five filters — all return AlternativeExplanation
super_alts   = alt0.super_explanations(only_ensured=False, include_potential=True, copy=True)
semi_alts    = alt0.semi_explanations(only_ensured=False, include_potential=True, copy=True)
counter_alts = alt0.counter_explanations(only_ensured=False, include_potential=True, copy=True)
ensured_alts = alt0.ensured_explanations(include_potential=True, copy=True)
pareto_alts  = alt0.pareto_explanations(include_potential=True, copy=True)

# Combine: ensured super-explanations only
ensured_super = alt0.super_explanations(only_ensured=True)

# Shorthand aliases (.super, .semi, .counter, .ensured, .pareto)
alt0.ensured()
alt0.pareto()
alt0.super()
alt0.semi()
alt0.counter()
```

### API — collection level (AlternativeExplanations)

```python
super_view   = alternatives.super_explanations(only_ensured=False, include_potential=True)
semi_view    = alternatives.semi_explanations()
counter_view = alternatives.counter_explanations()
ensured_view = alternatives.ensured_explanations()
pareto_view  = alternatives.pareto_explanations()
```

### Parameters shared by super / semi / counter

| Parameter | Default | Meaning |
|---|---|---|
| `only_ensured` | `False` | When `True`, additionally require that the uncertainty interval is no wider than the base |
| `include_potential` | `True` | Include potential alternatives (uncertainty spans the boundary) |
| `copy` | `True` | Return a new filtered object. `False` = mutate in place |

`ensured_explanations` and `pareto_explanations` do not take `only_ensured`
(they already imply the ensured condition).

---

## Alternatives-Specific Plot Styles

`AlternativeExplanation.plot()` supports two styles beyond `'regular'`:

```python
alt0.plot(style="triangular")     # highlights prediction vs uncertainty trade-off
alt0.plot(style="ensured")        # alias for "triangular"
alt0.plot(style="regular")        # standard bar chart
```

### Ranking controls (alternatives-specific defaults)

```python
alt0.plot(
    filter_top=10,           # max rules to show (None = show all)
    style="ensured",         # use triangular style to show uncertainty/prediction trade-off
    rnk_metric="ensured",    # default for AlternativeExplanation (vs "feature_weight" for factuals)
    rnk_weight=0.5,          # 0 = uncertainty only; +1 = higher prob ranked higher; -1 = lower prob ranked higher
)
```

| `rnk_metric` | Effect |
|---|---|
| `"ensured"` | Ranks by combined (prediction x uncertainty) score with `rnk_weight`, works with all styles|
| `"feature_weight"` | Ranks by raw feature impact (like factual default) |
| `"uncertainty"` | Alias: sets  `rnk_metric="ensured"` and `rnk_weight=0.0` and ranks by interval width |

`rnk_weight` range: -1 to 1.
- `rnk_weight = 0` -> uncertainty only
- `rnk_weight = 1` -> higher predicted class probability ranked higher
- `rnk_weight = -1` -> lower predicted class probability ranked higher

### Typical pattern: plot each ensured-framework view

```python
views = {
    "super":   alternatives.super_explanations(),
    "semi":    alternatives.semi_explanations(),
    "counter": alternatives.counter_explanations(),
    "ensured": alternatives.ensured_explanations(),
    "pareto":  alternatives.pareto_explanations(),
}
for name, view in views.items():
    print(f"View: {name}")
    view.plot(style="ensured", filter_top=None) # show all rules in triangular style
```

---

## Alternatives Conjunctions (Quick Reference)

```python
alternatives.add_conjunctions(max_rule_size=2)    # pairs (default)
alternatives.add_conjunctions(max_rule_size=3)    # triples
```

Plot conjunctions with triangular style to see prediction/uncertainty interplay:
```python
alternatives.add_conjunctions(max_rule_size=3).plot(style="ensured", filter_top=None)
```

For the full conjunction + filtering API, see `ce-explain-interact`.
