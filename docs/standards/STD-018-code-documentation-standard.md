> **Status note (2025-12-22):** Last edited 2025-12-24 · Archive after: Retain indefinitely as an engineering standard · Implementation window: Per Standard status (see Decision).

# Standard-018: Code Documentation Standardization

Formerly ADR-018. Reclassified as an engineering standard to keep ADRs scoped to
architectural or contract decisions.

Status: Active
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
extensions, while right-sizing enforcement for OSS development:

- **Module docstrings**: Every importable module must begin with a high-level summary
  describing purpose, key abstractions, and notable usage constraints.
- **Callable docstrings**: Public functions, classes, and methods must use numpydoc section
  headings (`Parameters`, `Returns`, `Raises`, `Examples` where applicable). Internal helpers
  (prefixed `_`) require at least a one-line summary.
- **Deprecation notices**: Use `.. deprecated::` directives (rendered by Sphinx) or explicit
  `Warnings` sections to clarify timelines and alternatives.
- **Public API guardrails**: WrapCalibratedExplainer contract members
  (fit/calibrate/explain/predict flows, plotting helpers, uncertainty/threshold
  options) MUST NOT acquire deprecation directives or altered signatures under
  this ADR; documentation should reinforce their stability and reference the
  release plan if evolution is ever proposed.
- **Type information**: Document accepted value ranges and shapes even when type hints exist,
  ensuring parity with runtime expectations.
- **Cross-references**: Prefer ``:mod:`package.module``` or ``:class:`ClassName``` references
  for related APIs to aid Sphinx cross-linking.
- **Automation hooks**: Introduce docstring linting via `pydocstyle` configured for numpydoc
  conventions, complemented by custom checks for coverage targets.
- **Enforcement posture**: On OSS/mainline CI, linting and coverage signals are blocking gates to prevent regressions. Release/stable branches continue to treat docstring linting and coverage targets as blocking gates, with the current threshold set at ≥94% overall coverage.

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
- Keeps OSS contribution flow lightweight while still protecting release readiness.

Negative/Risks:
- Short-term documentation backlog as legacy modules are rewritten to the new format.
- Potential friction for contributors unfamiliar with numpydoc; mitigated through templates
  and CONTRIBUTING.md updates.
- Advisory-only enforcement risks slower convergence without explicit backlog ownership.

## Adoption & Migration

1. Ratify this ADR and announce in the next contributor sync / changelog.
2. Update `CONTRIBUTING.md` with numpydoc primer and linting instructions.
3. Roll out tooling (pydocstyle configuration and docstring coverage script) with advisory
   CI reporting after baseline remediation.
4. Track compliance via documentation debt checklist for each subpackage until coverage
   meets targets, and flip release branch checks to blocking once targets are met.

## Open Questions

- Should private helpers within notebooks/testing utilities require full numpydoc sections?
- Do we enforce automated coverage thresholds per module or package-wide?
- Can we auto-generate sections for dataclasses and simple containers via templates?

## Implementation status

- 2025-10-06 – ADR accepted with agreement to stage enforcement alongside the
  v0.7.0 release cycle and seed contributor enablement material.
- v0.7.0 – CONTRIBUTING.md includes the numpydoc quick reference, CI runs
  pydocstyle in warning-only mode, and documentation debt trackers for
  batches A/B are published to guide cleanup work.
- v0.8.0 – Batches C (`explanations/`, `perf/`) and D (`plugins/`) reach
  compliance, CI starts reporting docstring coverage per module, and the
  documentation standardisation plan is updated with progress dashboards.
- v0.10.0 – Batches E (`viz/`, `viz/plots.py`, `legacy/plotting.py`) and F
  (`serialization.py`, `core.py`) are completed, docstring linting and coverage
  enforcement elevated to blocking on mainline CI with ≥94% threshold, and
  badges/reporting integrate with the docs build workflow per the release
  gate.
- v1.0.0-rc – Docstring coverage maintained at ≥94%, RC checklist calls
  out ongoing maintenance cadences, and regression alerts are wired into
  the release branch policies.
- v1.0.0 – Post-release audits ensure coverage remains above the gate and
  enumerate any deferred modules for the v1.0.x maintenance stream.
