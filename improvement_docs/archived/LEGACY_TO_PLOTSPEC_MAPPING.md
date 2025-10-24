> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Archived as of v0.8.x delivery · Implementation window: Historical (≤v0.8.x).

Mapping: legacy functions -> PlotSpec builder
============================================

This document lists the main branches in the legacy v0.5.1 plotting code
(`_plots (1).py`) and the corresponding rules the PlotSpec builders must
follow to reproduce the exact semantics.

1) `_plot_probabilistic(explanation, instance, predict, feature_weights, features_to_plot, num_to_show, column_names, title, path, show, interval=False, idx=None, save_ext=None)`

- Header: `dual`. Two small header axes: `header.pos` and `header.neg`.
- Body: `weight`. Main axis id: `main`.
- Data in PlotSpec:
  - mode: 'classification' or 'regression' (choose via explanation.get_mode())
  - feature_entries: for each feature index in `features_to_plot`, include weight, low, high, name, instance_value.
  - uncertainty: True if `interval==True` and predict contains low/high arrays.
  - axis_hints.header.pos/neg: xlim [0,1], xticks as np.linspace(0,1,6) for negative, empty xticks for positive header.
  - x-labels: set as in legacy (class labels via explanation.get_class_labels())

- Legacy drawing rules to capture in builder:
  - If predict['low']==-np.inf or predict['high']==np.inf, replace by explanation.y_minmax[0/1].
  - For each feature: if `interval` True, create `uncertainty_area` primitive (fill_between) between low and high. Create a `feature_bar` primitive: draw solid rect from 0 to weight only if the interval does NOT cross zero; otherwise suppress the feature_bar primitive. The overlay primitive for uncertainty must have alpha ~0.2 (adapter can use role).
  - Colors: positive widths use role 'positive_fill' (legacy 'b'), negative widths use role 'negative_fill' (legacy 'r'). Venn/interval bands use role 'venn_abers_band' with alpha ~0.15.

2) `_plot_regression(...)`

- Header: `single`. Top panel `header.main` shows regression interval as fill_betweenx of [pl, ph] and median line.
- Body: `weight`. Main axis id: `main`.

This document lists the main branches in the legacy v0.5.1 plotting code
(`_plots (1).py`) and the corresponding rules the PlotSpec builders must
follow to reproduce the exact semantics.

_plot_probabilistic(...)
------------------------

Header: `dual`. Two small header axes: `header.pos` and `header.neg`.

Body: `weight`. Main axis id: `main`.

Data in PlotSpec:

- mode: 'classification' or 'regression' (choose via explanation.get_mode())
- feature_entries: for each feature index in `features_to_plot`, include weight, low, high, name, instance_value.
- uncertainty: True if `interval==True` and predict contains low/high arrays.
- axis_hints.header.pos/neg: xlim [0,1], xticks as np.linspace(0,1,6) for negative, empty xticks for positive header.
- x-labels: set as in legacy (class labels via explanation.get_class_labels())

Legacy drawing rules to capture in builder:

- If predict['low']==-np.inf or predict['high']==np.inf, replace by explanation.y_minmax[0/1].
- For each feature: if `interval` True, create `uncertainty_area` primitive (fill_between) between low and high. Create a `feature_bar` primitive: draw solid rect from 0 to weight only if the interval does NOT cross zero; otherwise suppress the feature_bar primitive. The overlay primitive for uncertainty must have alpha ~0.2 (adapter can use role).
- Colors: positive widths use role 'positive_fill' (legacy 'b'), negative widths use role 'negative_fill' (legacy 'r'). Venn/interval bands use role 'venn_abers_band' with alpha ~0.15 for the band.


_plot_regression(...)
----------------------

Header: `single`. Top panel `header.main` shows regression interval as fill_betweenx of [pl, ph] and median line.

Body: `weight`. Main axis id: `main`.

PlotSpec fields:

- mode: 'regression'
- threshold: explanation.y_threshold if present
- uncertainty: True if `interval` True.
- axis_hints.header.main.xlabel -> f'Prediction interval with {confidence}% confidence'

Legacy drawing rules:

- When interval True: compute gwl/gwh relative to predict median and draw uncertainty_area across the full width for features as in legacy code. Handle infinite bounds with y_minmax substitution.
- Suppress solid feature bar if interval crosses zero.


_plot_alternative(...)
-----------------------

Header: `none` (draw Venn-Abers band as background on main axis).

Body: `predict`.

PlotSpec fields:

- mode: classification | regression
- feature_entries: feature-level predict/low/high
- venn_abers: {low_high: [p_l,p_h], predict: p}

Legacy drawing rules:

- Draw a `venn_abers_band` (fill_betweenx) across the rows using p_l and p_h (or split into two bands when p crosses 0.5). Use alpha ~0.15 for the band.
- For each feature: draw either a regression-style band (red) or a class-colored band using `__get_fill_color(venn_abers, reduction=0.99)`; if p_l/p_h straddle 0.5, split the band into two halves using p_l->0.5 and 0.5->p_h with appropriate colors.
- Axis x-limits: [0,1] for probabilistic; [y_minmax[0], y_minmax[1]] for regression.


_plot_triangular(...)
----------------------

Kind: `triangular`.

PlotSpec fields:

- is_probabilistic: boolean (True when classification or thresholded regression)
- proba: scalar or array (original proba)
- uncertainty: scalar or array
- rule_proba, rule_uncertainty arrays

Legacy drawing rules:

- If is_probabilistic True: do the proba triangle helper (grid lines and three curves). Otherwise compute min/max on rule_proba/proba and rule_uncertainty/uncertainty and extend ranges with 10% padding.
- Export quiver primitive with x=proba (replicated), y=uncertainty, u=rule_proba-proba, v=rule_uncertainty-uncertainty and a scatter for alternative rules and original prediction. Axis labels: 'Probability' or 'Prediction' and ylabel 'Uncertainty'.


_plot_global(...)
------------------

Kind: `global_probabilistic` or `global_regression` depending on whether `predict_proba` exists or `threshold` is provided.

PlotSpec fields:

- mode: classification|regression
- is_regularized: boolean (when proba is returned by predict_proba call)
- arrays: proba/predict, low, high, uncertainty, y_test
- axis_hints: computed min_x/min_y/max_x/max_y with 10% padding as in legacy

Legacy drawing rules:

- If is_regularized True: include a primitive hint to call the proba_triangle renderer (adapter may draw via dedicated routine). Otherwise create `scatter` primitives for predictions vs uncertainty. For multiclass, select predicted or actual class proba and corresponding uncertainty slices as legacy does.
- For continuous y_test (regression plot branch) map colors via a colormap (tab10 for categorical, viridis for continuous) and include legend metadata in the primitives export.


General mapping notes
---------------------

- Builders must capture all branches that alter axis labels, x-limits, tick counts, and color roles.
- For any branch that uses `explanation._get_explainer().is_multiclass()` and array indexing, the builder must include `class_index` or `use_predicted_class` in PlotSpec so adapters select the correct slices.
- Builders must explicitly compute replacement values for infinite low/high using explanation.y_minmax.
- Builders must compute `feature_order` deterministically (use the provided `features_to_plot` ordering or sort by absolute weight if called for by the caller).


Canonical primitive examples
----------------------------

Factual probabilistic, one feature (uncertainty True, interval [0.2,0.5], weight=0.35)

plot_spec: {kind: 'factual_probabilistic', header:'dual', body:'weight', uncertainty: true, feature_order:[0], feature_entries:[{index:0,name:'f0',weight:0.35,low:0.2,high:0.5}]}

primitives:

- {id:'p1', axis_id:'header.pos', type:'fill_between', coords:{x:[0,1], y1:[0.2,0.2], y2:[0.5,0.5]}, style:{color:'role:prob_fill', alpha:0.2}, semantic:'probability_fill'}
- {id:'p2', axis_id:'main', type:'fill_between', coords:{x:[0.0], y1:[0.2], y2:[0.5]}, style:{color:'role:positive_fill', alpha:0.2}, semantic:'uncertainty_area'}
- {id:'p3', axis_id:'main', type:'rect', coords:{x0:0, y0:-0.1, x1:0.35, y1:0.1}, style:{color:'role:positive_fill', alpha:1.0}, semantic:'feature_bar'}
