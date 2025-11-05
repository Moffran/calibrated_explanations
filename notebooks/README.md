# Notebooks - Learning Path

This folder contains interactive notebooks demonstrating how to use calibrated-explanations.

Quick learning path (recommended order):

1. quickstart_tiny.ipynb (2–5 minutes)  
   - Tiny, dependency-light notebook using scikit-learn toy data. Run this first if you want a fast local trial.
2. quickstart.ipynb (5–10 minutes)  
   - Full quickstart covering classification and regression with plotting and uncertainty.
3. quickstart_wrap.ipynb / quickstart_wrap_oob.ipynb (5–10 minutes)  
   - Wrapper API examples and out-of-bag calibration variants.
4. demo_binary_classification.ipynb (10–15 minutes)  
   - Binary classification practitioner examples (factual, alternatives, conjunctions).
5. demo_multiclass*.ipynb (10–20 minutes)  
   - Multiclass demos and VennAbers examples.
6. demo_regression.ipynb / demo_probabilistic_regression.ipynb (15–30 minutes)  
   - Interval and probabilistic regression, triangular plots, difficulty estimators.
7. demo_conditional.ipynb (15–30 minutes)  
   - Conditional/normalized explanations and bins.
8. demo_under_the_hood.ipynb (developer, 10–20 minutes)  
   - Internals: `_get_rules()`, telemetry payloads, and programmatic access to explanation fields.

Supplementary notebooks (additional_notebooks/)
- LIME_comparison.ipynb, demo_speeddating.ipynb, imbalanced_wrap.ipynb — special topics & comparisons.

Notes and tips
- If you need a very fast run for CI or to demo to colleagues, run `quickstart_tiny.ipynb` first.
- Look for the small "Learning objectives" cell at the top of each main notebook to pick the right demo for your use case.
- To reproduce examples exactly, use the same versions listed in `pyproject.toml` / `requirements.txt`.
