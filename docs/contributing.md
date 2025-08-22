# Contributing and Roadmap

Contributions are welcome. Please see the repository’s `CONTRIBUTING.md` for the full guide.

## Roadmap-driven development

We are updating the package according to a written Action Plan and ADRs:

- Action Plan: `improvement_docs/ACTION_PLAN.md` in the repo. Phases guide what kinds of changes are in-scope.
- ADRs: `improvement_docs/adrs/` contains accepted and proposed architectural decisions.

When opening a PR, please align with the current phase and reference the relevant Action Plan sections and ADRs.

## Quality gates (summary)

- Type checks: mypy for new/modified core modules.
- Linting: ruff and markdownlint.
- Tests: add unit tests for new behavior; keep runtime reasonable.
- Docs: update README/docs when public behavior changes.
