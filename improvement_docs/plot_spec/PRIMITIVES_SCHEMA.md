
Exported primitives schema (informal JSON)
=========================================

This document describes the `primitives` structure returned by
`export_drawn_primitives()` and used in adapter tests.

Top-level structure
-------------------

{
  "plot_spec": { ... },
  "primitives": [ <primitive>, ... ]
}

Primitive object (required fields)
----------------------------------
- id: string (unique within this export)
- axis_id: string (logical axis id, e.g., "header.pos", "main")
- type: string, one of: "rect", "fill_between", "line", "scatter", "quiver", "text", "legend", "area", "image"
- coords: object -- contents depend on type (see examples)
- style: object { color: "#RRGGBB" | "role:positive_fill", alpha: number, linewidth: number, linestyle: string, marker: string }
- semantic: string tag describing role (e.g., "feature_bar", "uncertainty_area", "venn_abers_band", "probability_fill", "base_line", "quiver_arrow")

Examples: coords shapes
- rect: {x0, y0, x1, y1}
- fill_between: {x: [...], y1: [...], y2: [...]} (data-space arrays)
- line: {x: [...], y: [...]}
- scatter: {x: [...], y: [...], sizes: [...]}
- quiver: {x: [...], y: [...], u: [...], v: [...]}

Example primitive (uncertainty area)

{
  "id": "p1",
  "axis_id": "main",
  "type": "fill_between",
  "coords": {"x": [0,1], "y1": [0.2,0.2], "y2": [0.5,0.5]},
  "style": {"color": "#FF0000", "alpha": 0.2},
  "semantic": "uncertainty_area"
}

Design notes
------------
- All coordinates are in data-space. Adapters must not convert to pixels
  in the primitive export.
- style.color can be a literal hex or a role string; tests should prefer
  role-based assertions for robustness unless the adapter reproduces the
  legacy color mixing algorithm.
