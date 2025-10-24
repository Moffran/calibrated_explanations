> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-009: Input Preprocessing & Mapping Persistence Policy

Status: Accepted
Date: 2025-08-22
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Issue reported: The library only supports numeric input natively. Users must call
`utils.helper.transform_to_numeric` manually to encode DataFrames with text/categorical
features and manage mappings externally. Native support in the wrapper would improve
usability and reproducibility.

## Decision

- Keep the core numeric; add preprocessing in the wrapper layer:
  - `wrap_explainer.py` learns/applies preprocessing (either built-in `transform_to_numeric` or a user-supplied transformer/pipeline).
  - Add configuration: `auto_encode=True|False|'auto'`, `preprocessor: Optional[Transformer]`, and policy for unseen categories (`'ignore'|'error'`).
  - Persist mapping artifacts on the explainer; attach mapping metadata to Explanation provenance.
- Validation (`core/validation.py`) detects DataFrames/non-numeric columns and enforces NaN/dtype policies, returning actionable errors.

## Consequences

- Positive: greatly improved ergonomics; deterministic mappings in predict/online; clearer provenance.
- Negative: additional complexity in wrapper; need to document behavior and storage.

## Alternatives

- Enforce user-supplied preprocessing only (status quo), which is less friendly and harder to reproduce.
- Push preprocessing into core classes, which couples responsibilities and complicates testing/design.

## Adoption & Migration

- Phase 1B: extend validation to be DataFrame-aware.
- Phase 2: introduce wrapper preprocessing options, persist mappings, and round-trip tests; document configs.

### Adoption Progress (2025-09-02)

Implemented:

- User-supplied preprocessor wiring with private fit/transform helpers in `wrap_explainer.py`.
- Config fields in `ExplainerConfig` (`preprocessor`, `auto_encode`) and builder pass-through.
- Docs updated (Getting Started, API reference) to show config-driven preprocessing.
- Tests added for deterministic reuse of the same transformer across fit/inference.

Pending:

- Automatic encoding path (`auto_encode='auto'`) and mapping persistence helpers for that mode with deterministic mapping storage on the wrapper.
- Unseen-category policy behavior and documentation (default `'error'`, configurable `'ignore'`).

## Open Questions

- Where to store mappings (in-memory only vs. optional serialization helpers)? Start in-memory with API hooks for export/import.
- Default `auto_encode` value and policy for unseen categories; propose default `'auto'` and `'error'` respectively, reconsider after user feedback.
