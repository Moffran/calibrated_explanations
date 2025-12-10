# Narrative Explanations

Calibrated Explanations supports generating human-readable narrative explanations. This feature allows you to convert the technical output of the explainer into natural language text, making it easier for non-technical stakeholders to understand the model's predictions.

## Overview

Narrative explanations are generated using a template-based approach. You can customize the templates to suit your specific domain and audience. The system supports multiple expertise levels (beginner, intermediate, advanced) and output formats (text, dataframe, HTML, dictionary).

## Usage

There are two main ways to generate narrative explanations:

1.  **`explain_with_narrative`**: Directly generate narratives from the explainer.
2.  **`to_narrative`**: Convert existing `CalibratedExplanations` objects to narratives.

### 1. Using `explain_with_narrative`

This method is called directly on the `CalibratedExplainer` instance. It generates factual and alternative explanations and then converts them to narratives.

```python
from calibrated_explanations import CalibratedExplainer

# ... train model and create explainer ...

# Generate narratives
explanations_df = ce.explain_with_narrative(
    X_test[:5],
    expertise_level=("beginner", "intermediate", "advanced")
)
```

### 2. Using `to_narrative`

This method is called on a `CalibratedExplanations` object (returned by `explain_factual`). It allows you to generate narratives from explanations you've already computed.

```python
# Generate factual explanations first
factual_explanations = ce.explain_factual(X_test)

# Convert to narrative
narratives = factual_explanations.to_narrative(
    expertise_level="beginner",
    output_format="text"
)
```

## Parameters

-   **`template_path`** (str, optional): Path to the YAML or JSON template file. Defaults to `explain_template.yaml` in the current directory.
-   **`expertise_level`** (str or tuple, default=("beginner", "advanced")): The level of detail for the narrative. Options: "beginner", "intermediate", "advanced".
-   **`output_format`** (str, default="dataframe"): The format of the output. Options:
    -   `"dataframe"`: Returns a pandas DataFrame.
    -   `"text"`: Returns a formatted text string.
    -   `"html"`: Returns an HTML table.
    -   `"dict"`: Returns a list of dictionaries.

## Customizing Templates

You can customize the narratives by creating your own YAML template file. The default template is `explain_template.yaml`.

Example structure of `explain_template.yaml`:

```yaml
beginner:
  factual:
    prediction: "Prediction: {prediction}"
    probability: "Calibrated Probability: {probability}"
    # ...
  alternative:
    # ...

intermediate:
  # ...

advanced:
  # ...
```

To use a custom template:

```python
narratives = explanations.to_narrative(
    template_path="path/to/custom_template.yaml"
)
```

## Examples

See `notebooks/narrative_demo.ipynb` for a complete demonstration of the narrative explanations feature.
