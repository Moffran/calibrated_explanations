> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

PlotSpec JSON schema (informal)
================================

This document describes the PlotSpec contract that builders will emit and
adapters will consume. It is intentionally JSON-serializable and backend
agnostic.


Top-level
---------

- plot_spec: object with metadata
- primitives: list of primitive objects (usually created by the adapter
  renderer or by a test harness calling the adapter's `export_drawn_primitives`).


PlotSpec fields (required)
--------------------------

- kind: string, one of:
  - "factual_probabilistic"
  - "factual_regression"
  - "alternative_probabilistic"
  - "alternative_regression"
  - "triangular"
  - "global_probabilistic"
  - "global_regression"
- mode: "classification" | "regression"
- header: object | null (interval metadata; supports keys like `pred`, `low`, `high`, `xlabel`, `ylabel`, `xlim`, `dual`)
- body: object | null (body metadata; supports keys like `bars_count`, `xlabel`, `ylabel`)
- style: "regular" | "triangular"
- uncertainty: boolean
- feature_order: array of feature indices in display order (empty for global/triangular)


PlotSpec fields (optional but recommended)
-----------------------------------------

- rank_by: "ensured" | "feature_weight" | "uncertainty" | null
- rank_weight: number in [-1,1]
- threshold: null | number | [number, number]
- class_index: null | integer (for multiclass plots which specific class to show)
- feature_entries: array of objects {index, name, weight, low, high, instance_value}
- global_entries: object with arrays (proba/predict, low, high, uncertainty, y_test)
- axis_hints: object mapping axis ids -> {xlim:[min,max], xticks:[...], xlabel:string}
- color_roles: object mapping role names to color hex strings (optional override)


Design notes
------------

- Coordinates in primitives are in data-space.
- Logical axis ids (example): "header.pos", "header.neg", "main". Adapters map
  these to backend axes and must place primitives on the correct axis.
- feature_order guarantees deterministic rendering order across adapters.


Example minimal PlotSpec (factual_probabilistic)
------------------------------------------------

{
  "plot_spec": {
    "kind": "factual_probabilistic",
    "mode": "classification",
    "header": {"pred": 0.65, "low": 0.2, "high": 0.8, "dual": true},
    "body": {"bars_count": 3},
    "style": "regular",
    "uncertainty": true,
    "feature_order": [2, 0, 1],
    "feature_entries": [
      {"index":2, "name":"feat2", "weight":0.35, "low":0.2, "high":0.5, "instance_value":"A"},
      {"index":0, "name":"feat0", "weight":0.10, "low":0.05, "high":0.15, "instance_value":0.6},
      {"index":1, "name":"feat1", "weight":-0.05, "low":-0.12, "high":0.02, "instance_value":3}
    ]
  }
}
