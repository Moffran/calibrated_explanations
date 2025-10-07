# ADR-020: Legacy User API Stability Guardrails

Status: Proposed
Date: 2025-10-07
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Public workflows rely on the README and accompanying notebooks to describe the
Calibrated Explanations API. Those materials exercise a stable set of entry
points on `WrapCalibratedExplainer`, `CalibratedExplainer`, and the explanation
collections, but we have no architectural record or regression tests pinning
that surface in place. As a result, seemingly innocuous refactors risk breaking
core capabilities such as probabilistic regression thresholds, interval tuning,
and conjunction tooling without warning.【F:docs/getting_started.md†L5-L413】【F:README.md†L150-L198】【F:improvement_docs/legacy_user_api_contract.md†L9-L93】

## Decision

Adopt a layered guardrail strategy that freezes the documented legacy API:

1. **Canonical contract.** The new `improvement_docs/legacy_user_api_contract.md`
   captures every method and parameter combination currently exercised in the
   README and notebooks. This document becomes the single source of truth for
   legacy-support decisions and must accompany any future user-facing change.【F:improvement_docs/legacy_user_api_contract.md†L1-L103】
2. **Signature regression tests.** Add unit tests that inspect the signatures of
   key methods on `WrapCalibratedExplainer`, `CalibratedExplainer`, and
   explanation collections to ensure required parameters (e.g.,
   `uq_interval`, `threshold`, `low_high_percentiles`, `filter_top`,
   `max_rule_size`, `n_top_features`) remain available. Tests also assert the
   presence of lifecycle helpers such as `.set_difficulty_estimator` and
   `.explore_alternatives` so accidental removals fail fast.【F:improvement_docs/legacy_user_api_contract.md†L15-L93】
3. **Release checklist hook.** During release planning, changelog reviewers must
   confirm whether the legacy contract is affected. Any intentional API change
   requires updating the contract document and the signature tests in the same
   pull request.

## Alternatives Considered

1. **Rely solely on documentation.** Rejected because prose alone does not fail
   builds when signatures drift.
2. **Record golden notebooks.** Rejected for now due to maintenance overhead and
   brittle outputs (plots, randomness). Signature tests provide lighter-weight
   protection while leaving room for future integration tests.
3. **Freeze the entire module namespace.** Rejected because some modules remain
   experimental; we only need to lock the user-facing workflow described in the
   public materials.

## Consequences

Positive:
- Contributors gain an explicit reference for what constitutes a breaking
  change in the legacy API.
- Continuous integration will fail if required parameters disappear or are
  renamed, surfacing issues early in review.
- Release engineers have a documented checklist tying intentional changes to
  updated tests and contract documentation.

Negative / Risks:
- Signature tests cover shape rather than behavior; subtle semantic changes may
  still slip through and require additional targeted tests.
- Keeping the contract document current adds review overhead, especially when
  notebooks evolve.

## Adoption & Migration

1. Land this ADR alongside the contract document and signature tests.
2. Update the contributor guide to reference the contract doc when proposing API
   changes (follow-up patch if needed).
3. Incorporate a legacy-API checkbox into the release checklist template so
   maintainers actively confirm whether updates are required.
4. Periodically audit notebooks against the contract doc to ensure new usage
   patterns are folded into the guardrails.

## Open Questions

- Should we add lightweight runtime smoke tests (e.g., executing the quick-start
  notebook) once CI resources allow?
- Do we need a deprecation policy for parameters that exist only for backward
  compatibility but are not exercised in current tutorials?

## Implementation Status

- 2025-10-07 – ADR proposed together with the initial contract document and
  signature-based regression tests.
