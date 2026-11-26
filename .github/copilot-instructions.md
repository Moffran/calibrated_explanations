# GitHub Copilot instructions

> [`CONTRIBUTOR_INSTRUCTIONS.md`](../CONTRIBUTOR_INSTRUCTIONS.md) is the single
> source of truth shared by all agent platforms. This file adds only GitHub
> Copilot-specific integration guidance.

## VS Code setup

- Setup and extension guidance:
  [`docs/get-started/copilot-setup.md`](../docs/get-started/copilot-setup.md).
- Durable corrections:
  [`.github/copilot-feedback-log.md`](copilot-feedback-log.md).

## Automatically applied instruction files

| File | Scope | Purpose |
|---|---|---|
| [`instructions/source-code.instructions.md`](instructions/source-code.instructions.md) | `src/**/*.py` | Python source boundaries and error-handling reminders |
| [`instructions/tests.instructions.md`](instructions/tests.instructions.md) | tests and test-like files | Route test work to `tests/README.md` |
| [`instructions/execution plan.instructions.md`](<instructions/execution plan.instructions.md>) | all files | Plan, ADR, release, and changelog authority |

These files are scoped reminders, not alternate owners of the repository rules.

## Prompt commands

Use the repository prompt files under [`.github/prompts/`](prompts/) for
`/generate-tests-strict`, `/implement-plugin`, `/fix-issue`, and
`/refresh-ce-context`.

## Workspace tips

- Use `@workspace` for CE-internal code discovery.
- Pin `#file:CONTRIBUTOR_INSTRUCTIONS.md` when repository-wide rules matter.
- Commit instruction changes together with the behavior or correction they govern.
