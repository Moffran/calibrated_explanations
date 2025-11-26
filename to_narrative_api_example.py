"""
Example demonstrating the new to_narrative API.

This shows the clean API for generating narratives from calibrated explanations.
"""

# The new to_narrative method provides a clean API:

# Basic usage
narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe"
)

# Different output formats:

# 1. DataFrame (default) - returns pandas DataFrame
df_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe"
)

# 2. Text - returns formatted text string
text_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level="beginner",
    output_format="text"
)

# 3. HTML - returns HTML table
html_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level=("beginner", "intermediate", "advanced"),
    output_format="html"
)

# 4. Dictionary - returns list of dictionaries
dict_narratives = explanations.to_narrative(
    template_path="exp.yaml",
    expertise_level="advanced",
    output_format="dict"
)

# Different expertise levels:

# Single level
beginner_only = explanations.to_narrative(
    expertise_level="beginner",
    output_format="text"
)

# Multiple levels
all_levels = explanations.to_narrative(
    expertise_level=("beginner", "intermediate", "advanced"),
    output_format="dataframe"
)

# Template path handling:

# If exp.yaml doesn't exist, automatically falls back to explain_template.yaml
narratives = explanations.to_narrative(
    template_path="exp.yaml",  # Will use default if not found
    expertise_level=("beginner", "advanced"),
    output_format="dataframe"
)

# Use custom template
narratives = explanations.to_narrative(
    template_path="/path/to/custom_template.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe"
)

# Use default template explicitly
narratives = explanations.to_narrative(
    template_path="explain_template.yaml",
    expertise_level=("beginner", "advanced"),
    output_format="dataframe"
)

print("The to_narrative method provides a clean, intuitive API for generating narratives!")
