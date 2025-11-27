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
`explain_template.yaml`.

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
