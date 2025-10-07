# ADR-002: Validation & Exception Design

Status: Accepted
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Validation logic currently scattered; errors raise generic `ValueError` / `RuntimeError` without actionable context. Consistent validation improves debuggability, facilitates golden tests for failure modes, and enables programmatic handling by downstream tooling.

## Decision

Introduce a structured validation layer and exception taxonomy, aligned with Phase 1B plan.

Exception taxonomy (finalized for 1B):

- Base: `CalibratedError(Exception)`
- Validation & usage:
  - `ValidationError(CalibratedError)`
  - `DataShapeError(ValidationError)`
  - `NotFittedError(CalibratedError)`
  - `ModelNotSupportedError(CalibratedError)`
- Configuration & runtime:
  - `ConfigurationError(CalibratedError)`
  - `ConvergenceError(CalibratedError)`
  - `SerializationError(CalibratedError)`

Validation and helper contracts:

- `validate_inputs(x, y=None, task="auto", allow_nan=False, require_y=False, n_features=None, class_labels=None, check_finite=True) -> None`
- `validate_model(model) -> None`
- `validate_fit_state(explainer, require=True) -> None`
- `infer_task(x, y, model) -> Literal["classification","regression"]`
- (Optional) `validate(condition, exc_cls, message, *, details=None)` helper; details may include short error codes for machine consumption.

Messaging guidelines: imperative, specify expected vs actual, reference the parameter name; keep message substrings stable to avoid breaking tests.

Module paths (authoritative):

- Exceptions: `src/calibrated_explanations/core/exceptions.py`
- Validation: `src/calibrated_explanations/core/validation.py` (replaces `validation_stub.py`)
- Parameter canonicalization: `src/calibrated_explanations/api/params.py`

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

Phase 1B:

- Implement `core/exceptions.py` and `core/validation.py`, remove `validation_stub.py`.
- Introduce `api/params.py` with alias canonicalization and combination checks.
- Replace generic exceptions in core paths with the new classes (message text preserved where asserted by tests).
- Add unit tests for exceptions/validation/params; wire mypy in CI (strict on new modules).

Phase 2–4:

- Expand canonicalization, add deprecation warnings for old parameter names.
- Broaden validation coverage and documentation; finalize migration guidance.

## Open Questions

- Should we namespace error codes (e.g., VAL001) or mnemonic strings? (Lean: mnemonic.)
- Provide optional warnings for soft issues (maybe via `warnings` module) alongside hard errors?
- Auto-collection of failed validation metrics? (Could integrate into metrics later.)

## Decision Notes

Review after Phase 1B integration; adjust taxonomy only if gaps are observed. Update ADR if new classes are introduced.
