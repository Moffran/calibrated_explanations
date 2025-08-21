# ADR-001: Core Decomposition Boundaries

Status: Accepted
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Current code interleaves calibration logic, explanation assembly, utilities, and I/O, making targeted performance tuning and API stabilization difficult. A clear modular boundary map enables parallel development, plugin points, and minimizes ripple effects of refactors.

## Decision

Define top-level internal packages (initial):

- `calibrated_explanations.core`: Shared domain models (PredictionSet, Interval, Explanation), strategy interfaces.
- `calibrated_explanations.calibration`: Calibration algorithms & conformal predictors.
- `calibrated_explanations.explanations`: Explanation strategies (attribution, intervals, diagnostics).
- `calibrated_explanations.cache`: Cache layer (see ADR-003) internal initially.
- `calibrated_explanations.parallel`: Parallel facade (see ADR-004) internal initially.
- `calibrated_explanations.schema`: JSON schema definitions & validation helpers (see ADR-005).
- `calibrated_explanations.plugins`: Registry & loading (to be defined in ADR-006).
- `calibrated_explanations.viz`: Visualization abstractions (future ADR-007).
- `calibrated_explanations.utils`: Non-domain helpers (logging, hashing, config parsing) — keep lean.

Public API consolidation via `calibrated_explanations.__init__` re-exporting stable entry points only (factories, dataclasses, high-level run functions). Everything else treated as private (leading underscore or omitted from `__all__`).

Rules:

- No cross-talk between siblings except through core domain models or explicitly defined interfaces.
- Calibration code must not import explanation modules (prevent circular conceptual dependency).
- Utilities layer must not depend on higher-level packages.
- Schema validation lives separately to avoid heavy imports where not needed.

## Alternatives Considered

1. Flat module namespace (simpler, but scaling pain and import cycles risk).
2. Split into multiple distrib packages (premature overhead now; maybe later for plugins).
3. Strict layering with facade pattern only (adds boilerplate without clear benefit yet).

## Consequences

Positive:

- Clear import graph enables mechanical refactors & golden tests.
- Simplifies future plugin surface delineation.
- Reduces accidental coupling and hidden dependencies.

Negative / Risks:

- Initial churn moving files & adjusting imports.
- Slight discoverability cost for new contributors (mitigate with README section + diagram).

## Adoption & Migration

Phase 1A: Create package skeletons + move code mechanically (no logic change) with compatibility shims.
Phase 1B: Update imports in tests & add golden serialization tests.
Phase 2+: Introduce cache/parallel modules behind feature flags.

## Open Questions

- Should `viz` be optional extra (`pip install calibrated-explanations[viz]`)?
- Do we need a `data` subpackage for dataset utilities or keep external?
- Where to host config system (utils vs dedicated `config`)?

## Decision Notes

Reassess after Phase 1 for any boundary adjustments.

Update 2025-08-21: Chosen filename for the wrapper explainer implementation is `wrap_explainer.py` (short form) instead of the earlier draft name `wrapper_explainer.py`. Action plan updated accordingly; no other impact.
