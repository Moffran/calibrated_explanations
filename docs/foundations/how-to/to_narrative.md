# Generate narratives with `to_narrative`

Use `explanations.to_narrative()` to turn calibrated explanations into narrative
text, HTML, or data structures without changing your modeling flow.

## Basic usage

```python
narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe",
)
```

## Output formats

1) **DataFrame** (default) — returns a pandas DataFrame

```python
df_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe",
)
```

2) **Text** — returns formatted text string

```python
text_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level="beginner",
    output_format="text",
)
```

3) **HTML** — returns an HTML table

```python
html_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level=("beginner", "intermediate", "advanced"),
    output_format="html",
)
```

4) **Dictionary** — returns a list of dictionaries

```python
dict_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level="advanced",
    output_format="dict",
)
```

## Expertise levels

**Single level**

```python
beginner_only = explanations.to_narrative(
    expertise_level="beginner",
    output_format="text",
)
```

**Multiple levels**

```python
all_levels = explanations.to_narrative(
    expertise_level=("beginner", "intermediate", "advanced"),
    output_format="dataframe",
)
```

## Template path handling

If `exp.yaml` doesn't exist, the method automatically falls back to
`explain_template.yaml` and emits a `UserWarning`.

```python
narratives = explanations.to_narrative(
    template_path="exp.yaml",  # Will use default if not found
    expertise_level=("beginner", "advanced"),
    output_format="dataframe",
)
```

Use a custom template:

```python
narratives = explanations.to_narrative(
    template_path="/path/to/custom_template.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe",
)
```

Use the default template explicitly:

```python
narratives = explanations.to_narrative(
    template_path="explain_template.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe",
)
```

The `to_narrative` method provides a clean, intuitive API for generating
narratives from calibrated explanations.

## Narrative facts for your own text

`to_narrative()` renders the packaged templates offline. If you write your own
text, or check text produced elsewhere, start from the deterministic facts CE
has already computed instead of re-reading explanation objects:

```python
from calibrated_explanations.explanations import build_narrative_spec

payload = explanations.to_json()["explanations"][0]
spec = build_narrative_spec(payload)

spec.prediction            # predict, low, high, low_bounded, high_bounded
spec.rules[0].rule         # strongest rule first (ADR-042 canonical order)
spec.rules[0].source       # "/rules/<i>": where the fact came from in the payload
spec.context               # metadata.narrative_context: problem type, class, threshold, reject state
spec.qualifiers            # e.g. ("rejected",) or ("unbounded_interval",)
spec.to_json()             # canonical JSON; identical input gives identical text and spec_id
```

The spec copies values without rounding and raises `ValidationError` for
missing, `NaN` or contradictory data. An endpoint with `low_bounded=False` or
`high_bounded=False` is not a calibrated bound (for example the open side of
`low_high_percentiles=(-inf, 90)`) and must not be stated as a numeric limit.
For alternatives, `rnk_metric` and `rnk_weight` select the ranking as in the
alternative plots.

A correct spec does not make text built from it correct: check your text
against the spec. CE does not describe narrative prose as calibrated.
