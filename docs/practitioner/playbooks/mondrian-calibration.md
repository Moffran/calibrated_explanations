# Mondrian (Conditional) Calibration Playbook

Mondrian calibration allows you to calibrate separately for different subgroups, addressing potential bias and providing group-specific uncertainty estimates.

## When to Use Mondrian Calibration

* **Fairness-aware deployments**: Calibrate separately by protected attributes to reveal group-specific prediction quality
* **Heterogeneous data**: When different subgroups have different prediction patterns or error distributions
* **Domain-specific groupings**: Industry segments, geographic regions, customer tiers, or other natural partitions

## Quick Start

```python
from calibrated_explanations import WrapCalibratedExplainer
import numpy as np

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)
explainer.calibrate(x_cal, y_cal, feature_names=feature_names)

# Define group labels for your test instances
# Example: group by a categorical feature or external attribute
group_labels = x_test[:, group_feature_idx]  # or any array of group assignments

# Use bins parameter for conditional calibration
factual = explainer.explain_factual(
    x_test,
    bins=group_labels
)
```

## Using MondrianCategorizer

For more control over bin definitions, use `crepes.extras.MondrianCategorizer`:

```python
from crepes.extras import MondrianCategorizer

# Create categorizer based on a continuous feature (auto-binning)
categorizer = MondrianCategorizer()
categorizer.fit(x_cal[:, feature_idx])

# Apply to calibration and explanation
factual = explainer.explain_factual(x_test, bins=categorizer)
```

### Custom Bin Boundaries

```python
# Define explicit bin boundaries
categorizer = MondrianCategorizer(bins=[0, 25, 50, 75, 100])
categorizer.fit(x_cal[:, age_feature_idx])
```

## Trade-offs

| Aspect | Without Mondrian | With Mondrian |
| :--- | :--- | :--- |
| Calibration data per group | Full set | Split by group |
| Interval width | May underestimate for minorities | Group-appropriate |
| Fairness visibility | Averaged across groups | Group-specific uncertainty |
| Sample requirements | Lower | Higher (need enough per group) |

```{admonition} Minimum Bin Size Warning
:class: warning

Each Mondrian bin needs sufficient calibration samples for reliable intervals. With too few samples per bin:

* Intervals may be overly wide or unstable
* Coverage guarantees may not hold

**Rule of thumb**: Aim for at least 30-50 samples per bin. If a bin has fewer samples, consider merging with adjacent bins or using fewer groups.
```

## Common Use Cases

### Fairness Analysis

```python
# Calibrate separately by protected attribute
protected_attr = x_test[:, gender_idx]  # e.g., 0 or 1

factual = explainer.explain_factual(x_test, bins=protected_attr)

# Compare uncertainty intervals across groups
# Wider intervals for a group may indicate less reliable predictions
```

### Domain-Specific Calibration

```python
# Calibrate by customer segment
segments = customer_data["segment"]  # e.g., "enterprise", "smb", "consumer"

factual = explainer.explain_factual(x_test, bins=segments)
```

### Geographic Regions

```python
# Calibrate by region where prediction patterns differ
regions = location_data["region"]  # e.g., "north", "south", "east", "west"

factual = explainer.explain_factual(x_test, bins=regions)
```

## Interpreting Mondrian Results

When using Mondrian calibration:

1. **Compare interval widths across groups**: Wider intervals indicate more uncertainty for that group
2. **Check coverage per group**: Verify that calibration quality holds for each subgroup
3. **Look for systematic differences**: If one group consistently has wider intervals, you may need more calibration data for that group

## Research Background

Conditional calibrated explanations are documented in:

> Löfström, H., et al. (2024). Conditional Calibrated Explanations.
> In: xAI 2024. Lecture Notes in Computer Science.
> [DOI: 10.1007/978-3-031-63787-2_17](https://link.springer.com/chapter/10.1007/978-3-031-63787-2_17)

## Cross-References

* {doc}`../../tasks/capabilities` - Full capability manifest
* {doc}`../../foundations/how-to/interpret_explanations` - Interpretation guide
* {doc}`../../citing` - Citation information
