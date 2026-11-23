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

## Which rules a narrative describes

A narrative describes the same rules, in the same order, as the plot of the
same explanation with the same ranking arguments. `to_narrative()` takes the
plot's `rnk_metric`, `rnk_weight` and `filter_top`, with the plot's defaults:

- **Factual explanations** rank rules by the absolute calibrated
  feature weight. Exact ties go to the narrower weight interval, then to rule
  order. `rnk_metric="feature_weight"` (or `None`) is the only accepted value.
- **Alternative explanations** rank by the `ensured` score
  `(1 - w)(1 - u) + w * p`, with `w = rnk_weight` (default `0.5`), `u` the
  alternative's interval width and `p` its prediction, oriented to the
  predicted class for classification and thresholded regression.
  `rnk_metric="uncertainty"` means `w = 0`, putting the most certain
  alternatives first. `rnk_metric="feature_weight"` ranks by how far the
  alternative's prediction moves from the current one, the v1.0.x order.
  For plain regression `p` and `u` are in target units and are not normalised,
  so choose `w` for your task.
- **`filter_top`** keeps the strongest rules: all of them for a single
  explanation (as `explanation.plot()`), ten per instance for a collection (as
  `explanations.plot()`). `None` keeps all, `0` none, and a negative value
  raises `ValidationError`.

```python
single = explanations[0]
text = single.to_narrative(output_format="text", filter_top=4)
single.plot(filter_top=4, show=False)  # the same four rules, in the same order

alternatives.to_narrative(output_format="text", rnk_metric="uncertainty")
```

Selection happens once, before grouping. The positive/negative groups and, at
the advanced level, the certain/uncertain groups only partition the selected
rules and keep their order. As in the alternative plot, alternatives identical
to the current prediction and interval are dropped after selection.

`add_conjunctions()` picks its candidates when it runs, using the same ranking.
If you will plot or narrate with a non-default `rnk_metric` or `rnk_weight`,
pass the same values to `add_conjunctions()`.

A ranking is a deterministic function of one explanation's estimates and
intervals. It carries no statistical guarantee, a top-k set is not "correct",
and a rank is not a measure of global feature importance.

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
