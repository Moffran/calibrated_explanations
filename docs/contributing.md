# Contributing and Roadmap

Contributions are welcome. Please see the repository’s `CONTRIBUTING.md` for the full guide.

## Roadmap-driven development

We are updating the package according to a written release plan and ADRs:

- Release Plan: `improvement_docs/RELEASE_PLAN_v1.md` tracks milestone scope and readiness gates on the way to v1.0.0.
- ADRs: `improvement_docs/adrs/` contains accepted and proposed architectural decisions.

When opening a PR, please align with the active milestone in the release plan and reference the relevant ADRs.

## Quality gates (summary)

- Type checks: mypy for new/modified core modules.
- Linting: ruff and markdownlint.
- Tests: add unit tests for new behavior; keep runtime reasonable.
- Docs: update README/docs when public behavior changes.
