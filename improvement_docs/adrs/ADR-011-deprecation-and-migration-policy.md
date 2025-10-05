# ADR-011: Deprecation & Migration Policy

Status: Accepted
Date: 2025-09-03
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Stabilizing public contracts while evolving internals requires a clear policy for deprecations and migrations to maintain user trust, especially around parameter aliases, module moves, and output schemas.

## Decision

Adopt a simple, predictable deprecation policy:

- Warning semantics: use `DeprecationWarning` emitted once per session per symbol via a central `deprecate(msg, *, once_key)` helper.
- Timeline: minimum of two minor releases before removal (e.g., introduced in v0.6.x; removed no earlier than v0.8.0).
- Scope: applies to public symbols, parameters (including aliases), module import paths, and serialized outputs.
- Communication: maintain a migration guide with side-by-side examples; keep a status table in `RELEASE_PLAN_v1.md`.
- Tooling: maintain an API snapshot diff tool and optional rewrite helpers for notebooks/scripts (parameter alias rewrites).

## Alternatives Considered

1. Immediate breaking changes with major version bumps (too disruptive for research users).
2. Silent aliasing without warnings (users unaware of upcoming removals).

## Consequences

Positive:

- Predictable upgrade path with early visibility of changes.
- Reduced breakage in downstream projects and notebooks.

Negative/Risks:

- Warning fatigue if not consolidated; mitigate via once-per-session helper and clear messages.

## Adoption & Migration

- Phase A–B: Turn on deprecation warnings for parameter aliases; document mapping table and examples.
- Phase G: Enforce policy gates in CI (no removal before two minor releases); publish migration notes per release.

## Open Questions

- Should we provide a `CE_DEPRECATIONS=error` env switch for strict users? (Proposed: yes.)
- Capture deprecation events in telemetry/metrics? (Out of scope for now.)
