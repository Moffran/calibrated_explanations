# Improvement (Legacy Transition Area)

This folder contains existing improvement plans, ADRs, and work-in-progress
governance material created before the `development/` documentation map was
introduced.

New development planning, ADRs, Standards, capability claims, requirements,
verification-framework material, and curated evidence summaries belong under
`development/` according to `development/README.md`. Existing files in this
folder remain valid until migrated.

## Start here

- Development map: `development/README.md`
- Current release plan: `docs/improvement/RELEASE_PLAN_v1.md` until migrated
- ADRs: `docs/improvement/adrs/` until migrated
- Test quality method (ADR-030 tooling + roles): `docs/improvement/test-quality-method/README.md` until migrated

## Transition rules

- Do not add new planning or verification-governance files here.
- When active material here is substantially changed, move it to the appropriate
  `development/` location in the same change when that move is in scope.
- Keep existing references accurate while migration is incomplete.

## Related plans

- Coverage uplift plan (archived): `docs/improvement/archived/coverage_uplift_plan.md`
- Anti-pattern remediation plan (archived): `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`
- Release checklist: `docs/improvement/release_checklist.md`
- Modality extension rollout plan (draft): `docs/improvement/modality_extension_rollout_plan.md`

## Notes

- Over-testing analysis relies on per-test coverage contexts via `pytest --cov-context=test`.
- Prefer behavioral, public-contract tests over coverage padding (ADR-030).
