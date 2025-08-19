# ADR-002: Validation & Exception Design

Status: Proposed
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Validation logic currently scattered; errors raise generic `ValueError` / `RuntimeError` without actionable context. Consistent validation improves debuggability, facilitates golden tests for failure modes, and enables programmatic handling by downstream tooling.

## Decision

Introduce structured validation layer:

- Custom exception hierarchy rooted at `CalibratedError`:
  - `DataShapeError`, `SchemaMismatchError`, `ConfigurationError`, `StrategyError`, `ResourceLimitError`.
- Helper `validate(condition, exc_cls, message, *, details=None)` raising enriched error with `.details` dict.
- Standardized message guidelines (imperative, specify expected vs actual, reference parameter name).
- Central module `calibrated_explanations.core.validation` consumed by calibration & explanation code.
- Error codes (short string) inside `.details['code']` for machine consumption.

Logging & surfacing:

- Optionally attach last validation errors summary to a debug report generation utility.
- Provide `explain_exception(e)` helper to produce human-oriented multi-line message.

## Alternatives Considered

1. Keep ad-hoc exceptions (low effort, inconsistent quality).
2. Use dataclasses for error payloads only (adds ceremony, exceptions already carry data).
3. External validation frameworks (overkill, dependency gravity).

## Consequences

Positive:

- Faster debugging & clearer issue reports.
- Stable surface for downstream tools to branch on error codes.
- Encourages early coherent input checking.

Negative / Risks:

- Slight upfront verbosity adding validations.
- Need contributor discipline to follow guidelines.

## Adoption & Migration

Phase 1: Introduce hierarchy + helper; refactor most common hotspots.
Phase 2: Sweep remaining modules; add tests asserting representative errors.
Phase 3: Document patterns in contributing guide.

## Open Questions

- Should we namespace error codes (e.g., VAL001) or mnemonic strings? (Lean: mnemonic.)
- Provide optional warnings for soft issues (maybe via `warnings` module) alongside hard errors?
- Auto-collection of failed validation metrics? (Could integrate into metrics later.)

## Decision Notes

Review after first wave of refactors to adjust error taxonomy.
