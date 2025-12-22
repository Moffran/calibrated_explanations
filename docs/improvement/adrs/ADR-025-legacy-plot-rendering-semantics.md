> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-025: Legacy Plot Rendering Semantics

Status: Deprecated (superseded by maintenance reference)
Date: 2025-10-18
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: docs/maintenance/legacy-plotting-reference.md
Related: ADR-014, ADR-016, ADR-021, ADR-024

## Context

This ADR previously documented pixel-level rendering behaviour for the legacy matplotlib plots. Maintaining that level of prescription in an ADR proved costly and does not align with the PlotSpec pathway, which only requires semantic consistency.

## Decision

- Pixel-level semantics for legacy plots are **removed from ADR scope**.
- The maintenance-only reference document records legacy rendering details for debugging and parity checks:
  `docs/maintenance/legacy-plotting-reference.md` (source of truth remains `src/calibrated_explanations/legacy/plotting.py`).
- New renderers are not required to replicate every legacy visual detail unless they explicitly aim for parity.

## Consequences

- Legacy parity information is preserved for maintainers without burdening architectural decisions.
- PlotSpec-related ADRs remain focused on intent and semantic guarantees.
