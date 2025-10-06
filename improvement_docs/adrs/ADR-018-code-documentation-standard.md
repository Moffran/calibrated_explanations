# ADR-018: Code Documentation Standardization

Status: Accepted
Date: 2025-10-06
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Docstring coverage and formatting diverge significantly across the package. A quarter of
functions lack docstrings, and modules mix differing documentation styles. The absence of a
shared standard complicates onboarding, impedes automated quality checks, and undermines
confidence in semi-public helper APIs.

## Decision

Adopt a unified documentation standard grounded in the numpydoc style guide with minimal
extensions:

- **Module docstrings**: Every importable module must begin with a high-level summary
  describing purpose, key abstractions, and notable usage constraints.
- **Callable docstrings**: Public functions, classes, and methods must use numpydoc section
  headings (`Parameters`, `Returns`, `Raises`, `Examples` where applicable). Internal helpers
  (prefixed `_`) require at least a one-line summary.
- **Deprecation notices**: Use `.. deprecated::` directives (rendered by Sphinx) or explicit
  `Warnings` sections to clarify timelines and alternatives.
- **Type information**: Document accepted value ranges and shapes even when type hints exist,
  ensuring parity with runtime expectations.
- **Cross-references**: Prefer ``:mod:`package.module``` or ``:class:`ClassName``` references
  for related APIs to aid Sphinx cross-linking.
- **Automation hooks**: Introduce docstring linting via `pydocstyle` configured for numpydoc
  conventions, complemented by custom checks for coverage thresholds.

## Alternatives Considered

1. **Ad-hoc cleanups per module**: rejected because the lack of guardrails would allow style
   regressions and produce inconsistent tone.
2. **Google-style docstrings**: rejected because the broader scientific Python ecosystem uses
   numpydoc, aligning with existing documentation tooling and contributor familiarity.

## Consequences

Positive:
- Consistent contributor expectations and faster code review for documentation.
- Unlocks automated docstring linting and potential doc coverage metrics in CI.
- Improves downstream library integrations relying on introspection.

Negative/Risks:
- Short-term documentation backlog as legacy modules are rewritten to the new format.
- Potential friction for contributors unfamiliar with numpydoc; mitigated through templates
  and CONTRIBUTING.md updates.

## Adoption & Migration

1. Ratify this ADR and announce in the next contributor sync / changelog.
2. Update `CONTRIBUTING.md` with numpydoc primer and linting instructions.
3. Roll out tooling (pydocstyle configuration and docstring coverage script) with CI gates
   after baseline remediation.
4. Track compliance via documentation debt checklist for each subpackage until coverage
   meets targets.

## Open Questions

- Should private helpers within notebooks/testing utilities require full numpydoc sections?
- Do we enforce automated coverage thresholds per module or package-wide?
- Can we auto-generate sections for dataclasses and simple containers via templates?

## Implementation status

- 2025-10-06 – ADR accepted with agreement to stage enforcement alongside the v0.7.0
  release cycle.
- Upcoming – Contributor documentation excerpts and initial tooling (pydocstyle
  configuration, docstring coverage reporting) to be added so contributors receive
  actionable feedback prior to strict CI enforcement.

