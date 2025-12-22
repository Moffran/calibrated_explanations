> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-024: Legacy Plot Input Contracts

Status: Deprecated (superseded by maintenance reference)
Date: 2025-10-18
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: docs/maintenance/legacy-plotting-reference.md
Related: ADR-014, ADR-016, ADR-020, ADR-021

## Context

Earlier drafts attempted to codify **every** positional argument and legacy payload detail to preserve perfect parity with historical plots. That level of detail is heavy to maintain and is not required for the PlotSpec pathway, which targets semantic interoperability rather than pixel-perfect parity.

## Decision

- The ADR no longer specifies exhaustive input contracts.
- Detailed legacy input behaviour lives in a **maintenance-only** reference:
  `docs/maintenance/legacy-plotting-reference.md` (source of truth remains the legacy code in `src/calibrated_explanations/legacy/plotting.py`).
- PlotSpec builders should target semantic correctness and rely on legacy plots only when strict parity is required.

## Consequences

- The legacy contracts remain documented for maintainers without constraining new renderers.
- ADRs stay focused on architectural scope rather than brittle implementation detail.
