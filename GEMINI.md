# GEMINI.md — Google Gemini instructions

> [`CONTRIBUTOR_INSTRUCTIONS.md`](CONTRIBUTOR_INSTRUCTIONS.md) is the single
> source of truth for all shared CE rules. This file adds only Gemini-specific
> workflow guidance.

## Context use

Use the larger context window to read complete relevant modules when an
architecture or behavior trace requires it, but prefer targeted reads and avoid
repeating the repository-wide rules in generated output.

## Tooling boundaries

- Do not modify files outside the repository root.
- Use the verification profiles and exact commands from
  `CONTRIBUTOR_INSTRUCTIONS.md`.
- Gemini CLI discovers CE skills through the generated `.agents/skills/`
  packages; edit skills only under `skills/` (index: `skills/README.md`).
- Preserve persisted feedback through the canonical feedback workflow instead
  of relying on cross-session memory.
