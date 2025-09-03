# ADR-012: Documentation & Gallery Build Policy

Status: Accepted
Date: 2025-09-03
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Documentation quality is central for research and community users. CI should build the API reference, render examples into HTML, and validate links to keep docs trustworthy.

## Decision

Establish doc build gates and example rendering policy:

- Build gates in CI: `sphinx-build -W` (treat warnings as errors) for API reference; `sphinx-build -b linkcheck` for external links.
- Examples: render notebooks via sphinx-gallery or nbconvert (MVP acceptable) into HTML; failures fail CI.
- Dependency handling: visualization backends kept as optional extras; gallery jobs install `[viz]` and `[notebooks]`.
- Contribution rules: examples must be runnable headlessly with seeded randomness and light datasets (<30s per example on CI hardware).
- Publishing: artifacts uploaded as workflow artifacts; optional GitHub Pages later.

## Alternatives Considered

1. Keep docs informal in README only (insufficient for scale and stability goals).
2. Build docs locally only (CI drift; broken links unnoticed).

## Consequences

Positive:

- Prevents doc rot; ensures examples remain first-class.
- Enables external users to explore examples as HTML without executing notebooks.

Negative/Risks:

- Slightly longer CI time; mitigate with a dedicated matrix job and caching.

## Adoption & Migration

- Phase F: Add docs CI job(s) with API ref build, example rendering, and linkcheck; fix initial warnings.
- Phase C: When PlotSpec lands, add examples illustrating adapter usage.

## Open Questions

- Choose between sphinx-gallery vs nbconvert for MVP; proposal: start with nbconvert then switch to gallery.
