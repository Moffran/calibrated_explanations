> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

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

Establish doc build expectations with right-sized gates for core development versus release
readiness:

- **OSS/mainline expectations:** `sphinx-build -W` and `sphinx-build -b linkcheck` should run
  in CI but are **advisory** for everyday OSS development. Warnings and linkcheck failures
  must be reported but do not block merges.
- **Release/stable branches:** `sphinx-build -W` and linkcheck are **blocking** gates, and
  docs artifacts must be reviewed before a release is cut.
- API reference scope: treat the WrapCalibratedExplainer contract (fit/calibrate/explain/predict
  flows, plotting, and uncertainty/threshold knobs) as immutable reference
  pages that must remain accurate and free of deprecation notices unless the
  contract itself evolves via a future ADR.
- Examples: render notebooks via sphinx-gallery or nbconvert (MVP acceptable) into HTML.
  Failures are advisory on OSS/mainline CI and blocking on release/stable branches.
- Dependency handling: visualization backends kept as optional extras; gallery jobs install
  `[viz]` and `[notebooks]`.
- Contribution rules: examples must be runnable headlessly with seeded randomness and light
  datasets (<30s per example on CI hardware).
- Publishing: artifacts uploaded as workflow artifacts; optional GitHub Pages later.

## Alternatives Considered

1. Keep docs informal in README only (insufficient for scale and stability goals).
2. Build docs locally only (CI drift; broken links unnoticed).

## Consequences

Positive:

- Prevents doc rot; ensures examples remain first-class.
- Enables external users to explore examples as HTML without executing notebooks.
- Keeps OSS contributions lightweight while preserving higher assurance gates for releases.

Negative/Risks:

- Slightly longer CI time; mitigate with a dedicated matrix job and caching.
- Advisory gates still require follow-up to avoid accumulating doc debt.

## Adoption & Migration

- Phase F: Add docs CI job(s) with API ref build, example rendering, and linkcheck.
- Phase C: When PlotSpec lands, add examples illustrating adapter usage.
- Before each release branch cut: resolve doc warnings and linkcheck failures so the release
  gate is green.

## Open Questions

- Choose between sphinx-gallery vs nbconvert for MVP; proposal: start with nbconvert then switch to gallery.

## Implementation status

- 2025-10-07 – Preparatory CI job landed in `main` behind an
  `allow_failure` flag to capture baseline warnings while documentation
  debt is resolved.
- v0.8.0 – Gallery prototypes render FAST plot walkthroughs and surface
  telemetry docs in the staging docs build, keeping the job optional
  while plugin routing stabilises.
- v0.9.0 – Release gate requires the docs build pipeline to pass
  blocking in CI (`sphinx-build -W`, linkcheck, gallery rendering) and
  publish artifacts for author review, matching the release plan
  milestone. Mainline CI continues to report warnings without blocking.
- v1.0.0-rc – Docs build treated as a release checklist item with
  nightly smoke runs; RC notes document the supported extras and
  gallery runtime expectations so adopters can mirror the setup.
- v1.0.0 – Stable branch inherits the blocking docs workflow and keeps
  linkcheck/gallery monitoring as part of post-release maintenance
  cadences.
