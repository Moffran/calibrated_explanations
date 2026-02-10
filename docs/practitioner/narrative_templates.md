# Narrative Template Customization

This guide explains how to customize the narrative templates used by the `to_narrative()` method in Calibrated Explanations.

## Overview

Narrative templates define how explanations are rendered into human-readable text. They use a YAML or JSON format with placeholders that get filled with actual values.

## Template Structure

Templates are organized hierarchically:

```yaml
narrative_templates:
  <problem_type>:
    <explanation_type>:
      <expertise_level>: |
        Template text with {placeholders}...
```

### Problem Types

- `binary_classification` - Two-class classification
- `multiclass_classification` - Multi-class classification
- `regression` - Standard regression with intervals
- `probabilistic_regression` - Regression with probability thresholds

### Explanation Types

- `factual` - Explains why the current prediction was made
- `alternative` - Shows how to change the prediction

### Expertise Levels

- `beginner` - Simple language, minimal details
- `intermediate` - Includes weights and prediction intervals
- `advanced` - Full weight intervals and uncertainty bounds

## Available Placeholders

### Global Context

| Placeholder | Description |
|-------------|-------------|
| `{label}` | Predicted class label |
| `{positive_label}` | Positive class label (binary classification) |
| `{calibrated_pred}` | Calibrated prediction value |
| `{pred_interval_lower}` | Lower bound of prediction interval |
| `{pred_interval_upper}` | Upper bound of prediction interval |
| `{threshold}` | Threshold value (scalar) |
| `{threshold_condition}` | Full threshold condition (e.g., "target <= 150" or "100 < target <= 200") |
| `{threshold_low}` | Lower threshold (for tuple thresholds) |
| `{threshold_high}` | Upper threshold (for tuple thresholds) |
| `{runner_up_class}` | Second-most likely class (multiclass) |
| `{margin_value}` | Margin to runner-up class |
| `{interval_width}` | Width of prediction interval |

### Per-Feature Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{feature_name}` | Name of the feature |
| `{feature_actual_value}` | Actual value of the feature |
| `{condition}` | Feature condition (e.g., "> 5.0") |
| `{feature_weight}` | Feature's contribution weight |
| `{feature_weight_low}` | Lower bound of weight interval |
| `{feature_weight_high}` | Upper bound of weight interval |
| `{predict}` | Predicted value for this rule |
| `{predict_low}` | Lower bound of prediction |
| `{predict_high}` | Upper bound of prediction |

## Conjunctive Rules

When `add_conjunctions()` is called on explanations, multiple feature conditions are combined. In narratives, these appear with a configurable separator:

```python
# Default: " AND "
explanations.to_narrative(conjunction_separator=" AND ")
# Result: "(Glucose > 120 AND BMI > 28)"

# Custom separator
explanations.to_narrative(conjunction_separator=" & ")
# Result: "(Glucose > 120 & BMI > 28)"
```

Conjunctive rules are automatically wrapped in parentheses for clarity.

## Weight Alignment

By default, narratives vertically align the weight columns for better readability:

```python
# With alignment (default)
explanations.to_narrative(align_weights=True)
# Result:
# * Glucose (145.0) > 120      — weight ≈ 0.150
# * BMI (28.5) > 25            — weight ≈ 0.082
# * Age (55) > 40              — weight ≈ 0.045

# Without alignment
explanations.to_narrative(align_weights=False)
# Result:
# * Glucose (145.0) > 120 — weight ≈ 0.150
# * BMI (28.5) > 25 — weight ≈ 0.082
# * Age (55) > 40 — weight ≈ 0.045
```

## Interval Probabilistic Regression

For probabilistic regression with interval thresholds, use tuple thresholds:

```python
# Scalar threshold: P(y <= 150)
explanations = explainer.explain_factual(x_test, threshold=150)

# Interval threshold: P(100 < y <= 200)
explanations = explainer.explain_factual(x_test, threshold=(100, 200))
```

The `{threshold_condition}` placeholder automatically adapts:
- Scalar: "target <= 150.000"
- Interval: "100.000 < target <= 200.000"

## Output Formats

The `to_narrative()` method supports multiple output formats:

| Format | Description |
|--------|-------------|
| `"dataframe"` | pandas DataFrame (default) |
| `"text"` | Plain text with headers |
| `"html"` | HTML table |
| `"dict"` | List of dictionaries |
| `"markdown"` | Markdown with code blocks |

## Creating Custom Templates

1. Copy the default template from `src/calibrated_explanations/templates/explain_template.yaml`
2. Modify the templates as needed
3. Pass your template path to `to_narrative()`:

```python
narratives = explanations.to_narrative(
    template_path="my_template.yaml",
    expertise_level="advanced",
    output_format="text"
)
```

### Example Custom Template

```yaml
narrative_templates:
  binary_classification:
    factual:
      beginner: |
        The model predicts: {label}
        Confidence: {calibrated_pred}

        Main reasons:
        * {feature_name} = {feature_actual_value}
```

## Best Practices

1. **Use appropriate expertise levels**: Beginners need simple explanations; experts want uncertainty details.
2. **Test with real data**: Verify placeholders are filled correctly for your use case.
3. **Keep templates readable**: Use line breaks and clear structure.
4. **Handle edge cases**: Consider what happens when there are few features or missing values.

## See Also

- [API Reference: Calibrated Explainer](../api/calibrated_explainer.md)
- [Probabilistic Regression](../foundations/concepts/probabilistic_regression.md)
- [Explanation Types](../tasks/capabilities.md)
