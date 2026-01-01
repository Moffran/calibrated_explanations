> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-014: Plot Plugin Strategy

Status: Accepted (scoped)
Date: 2025-09-16 (updated 2025-10-02)
Deciders: Core maintainers
Reviewers: TBD
Supersedes: ADR-013-interval-and-plot-plugin-strategy
Superseded-by: None
Related: ADR-006-plugin-registry-trust-model, ADR-007-visualization-abstraction-layer, ADR-016-plot-spec-separation

## Context
Explanation objects expose `.plot()` methods that currently delegate to helpers in `src/calibrated_explanations/viz/plots.py`. The helpers can either call the legacy matplotlib routines (`legacy/plotting.py`) or build a PlotSpec and render it with the `viz` adapter stack. Demand for alternative renderers exists, but plotting must remain an optional subsystem and must not expand the core API surface beyond what explanation users already rely on.

## Decision

Provide a **lightweight, opt-in** plugin mechanism focused on code-level extensibility rather than CLI-driven configuration. The default behaviour stays with the legacy matplotlib plots unless a caller explicitly opts into a different renderer.

1. **Minimal builder and renderer protocols (optional).**
   - `PlotRenderContext` encapsulates the explanation payload, instance metadata, plot intent, and caller-supplied options (`show`, `path`, `save_ext`).
   - `PlotBuilder` exposes `build(self, context) -> PlotArtifact`.
   - `PlotRenderer` exposes `render(self, artifact, *, context) -> PlotRenderResult`.
   - The legacy path remains callable directly, and the legacy renderer remains the default in `.plot()`.

2. **Registry is optional and thin.**
   - If implemented, registration is in-process (module import time), not a hard dependency for plotting.
   - Metadata should be minimal (identifier, dependencies, short description). Avoid strict schemas or configuration hierarchies that would require ongoing maintenance.
   - Entry points for third-party renderers are permitted but not required for core usage.

3. **Explicit opt-in for non-legacy renderers.**
   - `.plot()` uses the legacy path unless a caller explicitly passes a renderer/style or directly calls the PlotSpec-based helpers.
   - No CLI tooling, environment variables, or multi-step fallback chains are required to use or discover renderers.

4. **Legacy plotting remains canonical.**
   - The legacy builder/renderer pair is the primary path for parity behaviour.
   - PlotSpec-based renderers focus on semantic correctness, not pixel-perfect parity.

## Consequences
### Positive
- Keeps plotting extensibility available without increasing core complexity.
- Reduces operational surface area (no CLI tooling or configuration resolution order to support).
- Enables gradual adoption of PlotSpec without forcing a migration.

### Negative / Risks
- Less discoverability for third-party renderers without CLI tooling.
- In-process registration may be less flexible for dynamic environments.

## Status & Next Steps
- If needed, define the minimal protocols in `viz/plugins.py` and a small registry helper.
- Keep `.plot()` defaulting to the legacy renderer.
- Document PlotSpec-based renderers as experimental/optional until they provide clear user value.
