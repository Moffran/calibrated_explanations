# CLAUDE.md — Claude Code instructions

> [`CONTRIBUTOR_INSTRUCTIONS.md`](CONTRIBUTOR_INSTRUCTIONS.md) is the single
> source of truth for all shared CE rules. This file adds only Claude
> Code-specific tool guidance.

## Tool permissions

- The primary permission policy is `.claude/settings.json`.
- Local overrides belong in `.claude/settings.local.json`.
- Keep permission scopes minimal and repository-local.
- Do not bypass a permission constraint by switching toolchains.

## Skill discovery

Use the generated [`.claude/skills/README.md`](.claude/skills/README.md) to
discover the minimum skill set for the task. Canonical shared skill content lives
under `.claude/skills/`; compatibility trees contain shims only.

Durable feedback follows `CONTRIBUTOR_INSTRUCTIONS.md` rather than duplicating
shared rules in this file.
