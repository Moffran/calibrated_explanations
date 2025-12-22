> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-007: Visualization Abstraction Layer

Status: Accepted (scoped)
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Visualization currently couples matplotlib-specific code with explanation data construction. We need a modest separation so optional renderers (plotly, bokeh, headless export) can be introduced without rewriting explanation logic. At the same time, plotting is secondary to calibrated explanations and should not become a core architectural dependency.

## Decision

Introduce an **optional** `PlotSpec` intermediate representation (IR) as a minimal, JSON-serializable dict that describes the intended plot (kind, axes, data series, styles, annotations). Explanation strategies may emit `PlotSpec` objects; backends can render them when users explicitly opt in. The default plotting path remains the legacy matplotlib implementation.

Scope and constraints:

- `PlotSpec` is a lightweight, backend-agnostic contract intended for interoperability, not a full grammar-of-graphics.
- The rendering adapter layer is treated as a **secondary subsystem**. It must not impose new requirements on the core explanation API.
- The initial implementation can focus on a single adapter (matplotlib) and a small subset of plot kinds to validate the concept.
- Exact parity with legacy visuals is not a requirement of `PlotSpec`; the legacy renderer remains the canonical source for pixel-level behaviour.

Rendering API (optional):
`render(spec, backend="matplotlib", **opts) -> FigureLike` where `FigureLike` can be a matplotlib Figure, plotly Figure, or plain SVG/PNG bytes (if headless requested).

Extensibility:

- Plugins may register new `kind` identifiers, but this is optional and should not gate core explanation releases.
- Convert `PlotSpec` to/from JSON for caching or remote rendering when needed.

## Alternatives Considered

1. Keep direct matplotlib imperative code (fast now, blocks backend flexibility).
2. Adopt Vega-Lite fully (rich grammar but heavier dependency & learning curve; could map later).
3. Return raw pandas DataFrames and let users plot (less guidance, inconsistent visuals).

## Consequences

Positive:

- Enables optional backends without entangling the core explanation runtime.
- Supports caching or diffing of visual specs for targeted tests.
- Keeps plotting improvements incremental and opt-in.

Negative / Risks:

- Requires discipline to avoid backend leakage into spec.
- Multiple rendering paths increase test surface area if expanded too quickly.

## Adoption & Migration

Phase C (v0.6.x): Define `PlotSpec` + matplotlib adapter; convert 1–2 plots to emit spec then render; keep optional deps via extras.
Phase later (v0.7+): Add a second backend only if it delivers clear user value and does not expand the core contract.

## Open Questions

- Do we version spec separately from rendering API? (Yes.)
- Provide headless export utilities early or defer? (Probably early for docs.)
- Need theming system (light/dark) embedded in spec or adapter-level? (Adapter-level initially.)

## Decision Notes

Evaluate spec churn after first two backends; freeze 1.x guidelines.
