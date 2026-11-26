# AGENTS.md — Codex instructions

> [`CONTRIBUTOR_INSTRUCTIONS.md`](CONTRIBUTOR_INSTRUCTIONS.md) is the single
> source of truth for all shared CE rules. This file adds only Codex-specific
> workflow guidance.

## Task modes

Use one explicit mode per implementation turn:

- `quick-fix`: one small behavior correction with focused verification.
- `feature`: one release-scoped behavior change with tests and documentation.
- `hardening`: instruction, CI, governance, or test-policy work across files.

For every completed task, report changed files, validation commands and results,
and residual risk.

## Output shape

Keep implementation summaries reviewable:

```text
Task:
- goal
- scope and governing plan/ADR references
- changed files
- verification results
- migration note, if applicable
- residual risk
```

## Persistence and feedback

Codex has no reliable cross-session memory. Durable corrections belong in the
canonical source, a test, or the feedback log described by
`CONTRIBUTOR_INSTRUCTIONS.md`; do not make platform-specific copies of shared
engineering rules.
