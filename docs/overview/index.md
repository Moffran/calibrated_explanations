# Overview

Calibrated Explanations delivers calibrated, uncertainty-aware feature importance
for classification and regression models. The library wraps existing estimators
so teams can surface factual, alternative, and fast explanations with calibrated
probabilities and reproducible rule payloads.

## Release highlights (v0.8.0)

- PlotSpec routing is now the default renderer, with fallbacks recorded in
  telemetry for auditability.
- Telemetry payloads capture interval, probability, preprocessing, and plotting
  sources so downstream services can inspect execution without custom hooks.
- Documentation adopts a role-based structure to guide practitioners, researchers,
  and maintainers to the right entry point.

See the [release notes](../governance/release_notes.md) for a curated summary of
changes and links to the full changelog.

## Quick links

- [Install the package](../get-started/installation.md)
- [Run the classification quickstart](../get-started/quickstart_classification.md)
- [Explore the regression quickstart](../get-started/quickstart_regression.md)
- [Browse the API reference](../reference/api.md)

## Optional extras

Some functionality is provided via optional extras to keep the core lean:

- Visualization (matplotlib): `pip install "calibrated_explanations[viz]"`
- LIME integration: `pip install "calibrated_explanations[lime]"`

Plotting raises a friendly error if matplotlib is not installed, with guidance
for enabling the `viz` extra.
