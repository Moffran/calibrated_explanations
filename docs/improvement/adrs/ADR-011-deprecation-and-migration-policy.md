> **Status note (2026-04-14):** Last edited 2026-04-14 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.9.0–v1.0.0 finalization window (with post-v1 maintenance).

# ADR-011: Deprecation & Migration Policy

Status: Accepted
Date: 2025-09-03
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

The project needs a deprecation policy that is predictable during normal development and still guarantees that the first stable major release is not burdened by known deprecated paths.

## Decision

Adopt a two-layer deprecation policy with an explicit precedence rule.

### 1) Default policy (normal operation)

- Warning semantics: emit `DeprecationWarning` once per session per symbol via the central `deprecate(msg, *, once_key)` helper.
- Timeline: deprecations remain for a minimum of two minor releases before removal.
- Scope: applies to public symbols, parameters (including aliases), module import paths, and serialized outputs.
- ADR-020 exception: symbols covered by ADR-020 legacy-stability commitments remain major-gated unless explicitly reclassified by release-plan governance.

### 2) v1.0 finalization exception (binding override)

During the final pre-v1.0 release line (`v0.11.x`), the default two-minor timeline is overridden for any deprecation that would otherwise survive into `v1.0.0`.

**Binding rule:** `v1.0.0` must ship with zero surviving deprecations.

To enforce that rule, maintainers must, within `v0.11.x`:

1. Remove deprecated paths outright, or
2. Complete migration and close the deprecation, or
3. Convert unresolved work into explicit pre-v1.0 blocking tasks assigned to a named `v0.11.x` milestone with verifiable release gates.

No deprecation-removal work may be deferred to `v1.0.0`, `v1.0.0-rc`, or any post-`v1.0.0` release.

### 3) Precedence rule

When default-policy timing conflicts with the v1.0 finalization exception, **the v1.0 finalization exception wins**.

## Alternatives Considered

1. Immediate breaking changes with major version bumps for every cleanup (too disruptive for existing users).
2. Keep strict two-minor timing even when it pushes deprecated surfaces into `v1.0.0` (rejected: violates the zero-deprecation major-release objective).
3. Silent aliasing without warnings (rejected: users cannot plan migration).

## Consequences

Positive:

- Maintainers and users get clear, enforceable timelines.
- The major release boundary is clean: no active deprecation debt in `v1.0.0`.
- Release-plan ownership becomes auditable because unresolved deprecations must be tracked as explicit `v0.11.x` blockers.

Negative/Risks:

- Some deprecations may be removed earlier than the normal two-minor cadence.
- Mitigation: mandatory migration docs, explicit release-plan scheduling, and CI checks (`CE_DEPRECATIONS=error`) while deprecations are still active.

## Adoption & Migration

- Keep the default two-minor policy for normal releases.
- For final pre-v1.0 cleanup, maintain a release-plan table that maps every active deprecation to a specific `v0.11.x` closure milestone.
- Block milestone closure if any scheduled deprecation closure item is incomplete.

## Open Questions

- Should strict mode add structured telemetry for deprecation invocations (`CE_DEPRECATIONS=error` + event stream)?
- Should a generated deprecation ledger be published automatically in release artifacts?
