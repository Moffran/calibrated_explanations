# Governance

This document outlines how the Calibrated Explanations project is stewarded,
how decisions are made, and how maintainers support contributors.

## Project stewardship

The project is stewarded by the maintainers listed below. Maintainers are
responsible for:

- Setting release priorities and approving ADRs.
- Reviewing and merging changes.
- Managing security disclosures and community health files.
- Ensuring the published documentation matches the released code.

Current maintainers:

- Helena Löfström (helena.lofstrom@ju.se)
- Tuwe Löfström (tuwe.lofstrom@ju.se)

## Decision-making

- **Small changes** (docs, tests, refactors) can be approved by any maintainer.
- **Significant changes** (public APIs, architecture, serialization contracts,
  plugin registry rules) require an ADR in `docs/improvement/adrs/` and at least
  one additional maintainer review.
- **Release gates** are tracked in `docs/improvement/RELEASE_PLAN_v1.md` and the
  matching `docs/improvement/vx.y.z_plan.md` implementation plans.

## Contributor expectations

- Follow the guidance in `.github/CONTRIBUTING.md`.
- Adhere to the Code of Conduct in `CODE_OF_CONDUCT.md`.
- Use the release plan to align PR scope with the current milestone.

## Escalation and conflict resolution

If there is a conflict that cannot be resolved in a PR or issue thread, the
maintainers will:

1. Summarize the positions and request a cooling-off period.
2. Convene a short review (async or meeting) to decide the next step.
3. Document the decision in the issue/PR or an ADR if needed.

## Changes to this document

Changes to governance require maintainer approval and should be proposed via a
pull request.
