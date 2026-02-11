# Notebooks - Learning Path

This folder contains interactive notebooks demonstrating how to use calibrated-explanations.

## Quickstarts (start here)

- [quickstart_tiny.ipynb](./quickstart_tiny.ipynb) (2–5 minutes)
   - Tiny, dependency-light notebook using scikit-learn toy data.
- [quickstart.ipynb](./quickstart.ipynb) (5–10 minutes)
   - Full quickstart covering classification and regression with plotting and uncertainty.

## Core demos (practitioner workflows)

- Classification
   - [demo_binary_classification.ipynb](./core_demos/demo_binary_classification.ipynb) (10–15 minutes)
   - [demo_multiclass_glass.ipynb](./core_demos/demo_multiclass_glass.ipynb) (10–20 minutes)
- Regression
   - [demo_regression.ipynb](./core_demos/demo_regression.ipynb) (15–30 minutes)
   - [demo_probabilistic_regression.ipynb](./core_demos/demo_probabilistic_regression.ipynb) (15–30 minutes)

## Advanced topics

- [demo_conditional.ipynb](./advanced/demo_conditional.ipynb) (15–30 minutes)
- [demo_reject.ipynb](./advanced/demo_reject.ipynb) (10–20 minutes)
- [demo_narrative_explanations.ipynb](./advanced/demo_narrative_explanations.ipynb) (10–20 minutes)
- [fast_feature_filtering_demo.ipynb](./advanced/fast_feature_filtering_demo.ipynb)

## Developer / internals

- [demo_under_the_hood.ipynb](./advanced/demo_under_the_hood.ipynb) (developer, 10–20 minutes)
   - Internals: `_get_rules()`, telemetry payloads, and programmatic access to explanation fields.
- [demo_plugin_wiring.ipynb](./advanced/demo_plugin_wiring.ipynb) (developer, 5–10 minutes)
   - How to wire plugins in a controlled way.

## Supplementary notebooks

See [miscellaneous/](./miscellaneous/) for special topics, comparisons, and issue-focused reproductions.

## Notes & tips

- If you need a very fast run for CI or to demo to colleagues, start with [quickstart_tiny.ipynb](./quickstart_tiny.ipynb).
- Look for the small "Learning objectives" cell at the top of each main notebook to pick the right demo for your use case.
- To reproduce examples exactly, use the same versions listed in `pyproject.toml` / `requirements.txt`.

## Folder hygiene (why things look “messy”)

- [plots/](./plots/) contains generated images (not essential for understanding the notebooks). It can be treated as build output.
- Some supplementary notebooks may create tool-specific output folders (e.g., `catboost_info/`) under `miscellaneous/`.
