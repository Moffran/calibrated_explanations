# Code Hotspots (Extended Code-Focused Cycle)

Ranking heuristic: `risk_score = length + 8*args + 3*branches`.

| Rank | Score | Length | Args | Branches | Function | Location |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2601 | 1473 | 6 | 360 | `render` | `src/calibrated_explanations/viz/matplotlib_adapter.py:106` |
| 2 | 1190 | 682 | 2 | 164 | `_render_body` | `src/calibrated_explanations/viz/matplotlib_adapter.py:529` |
| 3 | 973 | 504 | 14 | 119 | `plot_alternative` | `src/calibrated_explanations/plotting.py:985` |
| 4 | 760 | 527 | 1 | 75 | `feature_task` | `src/calibrated_explanations/core/explain/feature_task.py:117` |
| 5 | 730 | 443 | 10 | 69 | `invoke` | `src/calibrated_explanations/core/explain/orchestrator.py:279` |
| 6 | 639 | 321 | 12 | 74 | `expand_template` | `src/calibrated_explanations/core/narrative_generator.py:509` |
| 7 | 587 | 294 | 16 | 55 | `plot_probabilistic` | `src/calibrated_explanations/plotting.py:378` |
| 8 | 566 | 253 | 20 | 51 | `build_alternative_probabilistic_spec` | `src/calibrated_explanations/viz/builders.py:433` |
| 9 | 552 | 414 | 3 | 38 | `explain_batch` | `src/calibrated_explanations/plugins/builtins.py:443` |
| 10 | 515 | 339 | 7 | 40 | `explain` | `src/calibrated_explanations/core/explain/_legacy_explain.py:29` |
| 11 | 515 | 307 | 8 | 48 | `explain_predict_step` | `src/calibrated_explanations/core/explain/_computation.py:320` |
| 12 | 497 | 286 | 2 | 65 | `build` | `src/calibrated_explanations/plugins/builtins.py:1221` |
| 13 | 454 | 194 | 19 | 36 | `build_alternative_regression_spec` | `src/calibrated_explanations/viz/builders.py:688` |
| 14 | 440 | 257 | 9 | 37 | `generate_narrative` | `src/calibrated_explanations/core/narrative_generator.py:187` |
| 15 | 439 | 278 | 4 | 43 | `add_conjunctions` | `src/calibrated_explanations/explanations/explanation.py:3085` |
| 16 | 434 | 276 | 4 | 42 | `add_conjunctions` | `src/calibrated_explanations/explanations/explanation.py:1931` |
| 17 | 427 | 163 | 21 | 32 | `build_probabilistic_bars_spec` | `src/calibrated_explanations/viz/builders.py:894` |
| 18 | 411 | 190 | 16 | 31 | `plot_regression` | `src/calibrated_explanations/plotting.py:675` |
| 19 | 409 | 247 | 12 | 22 | `__init__` | `src/calibrated_explanations/core/calibrated_explainer.py:74` |
| 20 | 405 | 277 | 4 | 32 | `compute_filtered_features_to_ignore` | `src/calibrated_explanations/core/explain/_feature_filter.py:137` |
| 21 | 391 | 220 | 9 | 33 | `predict_impl` | `src/calibrated_explanations/core/prediction/orchestrator.py:331` |
| 22 | 368 | 164 | 15 | 28 | `build_regression_bars_spec` | `src/calibrated_explanations/viz/builders.py:253` |
| 23 | 356 | 156 | 13 | 32 | `_plot_probabilistic` | `src/calibrated_explanations/legacy/plotting.py:86` |
| 24 | 338 | 148 | 11 | 34 | `plot_alternative` | `src/calibrated_explanations/legacy/plotting.py:470` |
| 25 | 318 | 185 | 2 | 39 | `build_lines` | `src/calibrated_explanations/core/narrative_generator.py:554` |
| 26 | 305 | 178 | 2 | 37 | `resolve_plugin` | `src/calibrated_explanations/core/explain/orchestrator.py:935` |
| 27 | 292 | 174 | 5 | 26 | `plot_global` | `src/calibrated_explanations/legacy/plotting.py:621` |
| 28 | 290 | 178 | 8 | 16 | `plot` | `src/calibrated_explanations/viz/narrative_plugin.py:87` |
| 29 | 287 | 117 | 13 | 22 | `plot_regression` | `src/calibrated_explanations/legacy/plotting.py:250` |
| 30 | 279 | 171 | 6 | 20 | `predict_proba` | `src/calibrated_explanations/core/calibrated_explainer.py:2483` |
