---
name: ce-onboard
description: >
  Provide a short read-only CE session primer and route task work to the
  canonical instructions and generated discovery indexes.
model: haiku
---

# CE Onboard

Use this skill to orient a new session; do not reproduce the full policy.

## Required entry points

1. Read `CONTRIBUTOR_INSTRUCTIONS.md`.
2. Read `development/README.md`.
3. Read the sole active `development/current-work/vX.Y.Z_plan.md` before
   selecting release work.
4. Use the generated indexes to discover only relevant sources:
   - `development/adrs/README.md`
   - `development/standards/README.md`
   - `.claude/skills/README.md`

## CE-first reminder

Use `WrapCalibratedExplainer` from the public API. Instantiate it directly unless
an intentional, verified subclass preserves the wrapper contract. Follow
`fit -> calibrate -> explain/predict`, verify both lifecycle flags, and default
to calibrated outputs with uncertainty.

Do not use `calibrated_explanations.ce_agent_utils` as a shortcut in new code.

## Navigation

Select the minimum specialist skill for the task; task-specific workflows live
under `.claude/skills/`. Common choices include:

- `ce-adr-consult` for applicable architectural decisions.
- `ce-release-task` for one active-plan deliverable.
- `ce-test-author` for a specific test.
- `ce-code-review` for change risk and ADR conformance.
- `ce-docstring-author` or `ce-rtd-writer` for documentation work.

The full generated inventory is the only skill catalogue; do not maintain another
manual list in this primer.
