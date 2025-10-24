> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-007: Visualization Abstraction Layer

Status: Accepted
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Visualization currently couples matplotlib-specific code with explanation data construction. Need separation so alternative backends (plotly, bokeh) or headless export (SVG/JSON) can be supported without rewriting explanation logic.

## Decision

Introduce `PlotSpec` intermediate representation (IR): a pure-Python (JSON-serializable) dict structure describing intended plot (kind, axes, data series, styles, annotations). Explanation strategies emit `PlotSpec` objects; backend renderers translate to concrete library calls.

Components:

- `PlotSpec` dataclass (kind, data, encodings, meta, version).
- Registry of renderer adapters (`matplotlib`, future `plotly`).
- Validation: `validate_plotspec(spec)` ensuring required fields by kind.
- Stable minimal schema version (1.0) independent from explanation schema (see ADR-005).

Rendering API:
`render(spec, backend="matplotlib", **opts) -> FigureLike` where `FigureLike` can be a matplotlib Figure, plotly Figure, or plain SVG/PNG bytes (if headless requested).

Extensibility:

- Plugins may register new `kind` with declared required fields + default renderer fallback.
- Convert `PlotSpec` to/from JSON for caching or remote rendering.

## Alternatives Considered

1. Keep direct matplotlib imperative code (fast now, blocks backend flexibility).
2. Adopt Vega-Lite fully (rich grammar but heavier dependency & learning curve; could map later).
3. Return raw pandas DataFrames and let users plot (less guidance, inconsistent visuals).

## Consequences

Positive:

- Backend-agnostic; easier to add interactive outputs later.
- Supports caching or diffing of visual specs for golden tests.
- Simplifies testing by asserting spec content without rendering.

Negative / Risks:

- Requires discipline to avoid backend leakage into spec (document constraints).
- Initial time investment to implement adapters.

## Adoption & Migration

Phase C (v0.6.x): Define `PlotSpec` + matplotlib adapter; convert 1–2 plots to emit spec then render; keep optional deps via extras.
Phase later (v0.7+): Add a second backend (plotly) to validate abstraction; extend examples and docs.

## Open Questions

- Do we version spec separately from rendering API? (Yes.)
- Provide headless export utilities early or defer? (Probably early for docs.)
- Need theming system (light/dark) embedded in spec or adapter-level? (Adapter-level initially.)

## Decision Notes

Evaluate spec churn after first two backends; freeze 1.x guidelines.
