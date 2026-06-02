> **Status note (2026-01-12):** Last edited 2026-01-12 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

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
  - `auto_encode='auto'` is the automatic encoding mode; it deterministically
    learns mappings during fit/init and applies them during inference.
  - Default unseen-category policy is `'error'`; `'ignore'` is an explicit opt-in.
  - Persist mapping artifacts on the explainer; attach mapping metadata to Explanation provenance.
  - Provide mapping persistence helpers:
    `Explainer.export_mapping() -> dict` and `Explainer.import_mapping(mapping: dict) -> None`.
    Mapping primitives must be JSON-safe for serialization.
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

- Automatic encoding path (`auto_encode='auto'`) and mapping persistence helpers
  for that mode with deterministic mapping storage on the wrapper.
- Unseen-category policy behavior and documentation (default `'error'`,
  configurable `'ignore'`).
- Mapping export/import helpers with JSON-safe primitives and documentation
  examples.

## Open Questions

- Where to store mappings (in-memory only vs. optional serialization helpers)?
  Start in-memory with API hooks for export/import.

## Implementation status (2026-06-02, v0.11.3 Task 9 Workstream B)

**Gap 2 — JSON-safe mapping export (closed):** `WrapCalibratedExplainer.export_preprocessor_mapping()` (in `src/calibrated_explanations/core/wrap_explainer.py`) already enforces JSON-safe output via `_validate_json_safe_mapping()` on both the `get_mapping_snapshot` and `mapping_` fallback paths. `import_preprocessor_mapping()` also validates JSON safety on import. Tests in `tests/unit/core/test_wrap_explainer_helpers.py` verify enforcement. The public helper names are `export_preprocessor_mapping` / `import_preprocessor_mapping` (not `export_mapping` / `import_mapping` as originally proposed — placement on wrapper is deliberate per ADR-001 boundary rules).

**Gap 3 — Helper-placement doc drift (closed):** The ADR-009 §Decision describes `Explainer.export_mapping()` and `Explainer.import_mapping()`. The actual implementation exposes these as `WrapCalibratedExplainer.export_preprocessor_mapping()` and `WrapCalibratedExplainer.import_preprocessor_mapping()`. This naming differs from the ADR proposal text but is intentional: the wrapper is the public preprocessing API surface (ADR-001 boundary), and the more descriptive names distinguish mapping persistence from other wrapper export operations. No code change is required; this note records the deliberate divergence so future contributors do not chase a naming drift as a bug.
