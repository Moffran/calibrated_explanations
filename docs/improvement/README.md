# Improvement (Maintainers & Contributors)

This folder collects improvement plans, ADRs, and work-in-progress governance
material used to steer releases.

## Start here

- Release plan: `docs/improvement/RELEASE_PLAN_v1.md`
- ADRs: `docs/improvement/adrs/`
- Test quality method (ADR-030 tooling + roles): `docs/improvement/test-quality-method/README.md`

## Related plans

- Coverage uplift plan (archived): `docs/improvement/archived/coverage_uplift_plan.md`
- Anti-pattern remediation plan (archived): `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`
- Release checklist: `docs/improvement/release_checklist.md`

## Notes

- Over-testing analysis relies on per-test coverage contexts via `pytest --cov-context=test`.
- Prefer behavioral, public-contract tests over coverage padding (ADR-030).
