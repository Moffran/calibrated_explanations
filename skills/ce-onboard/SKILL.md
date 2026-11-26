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
   - `skills/README.md`

## CE-first reminder

Use `WrapCalibratedExplainer` from the public API. Instantiate it directly unless
an intentional, verified subclass preserves the wrapper contract. Follow
`fit -> calibrate -> explain/predict`, verify both lifecycle flags, and default
to calibrated outputs with uncertainty.

## Navigation

Select the minimum specialist skill for the task from the generated
`skills/README.md` catalogue; it is the only skill inventory.
