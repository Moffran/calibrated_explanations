# Release checklist

Before publishing a release, review `ROADMAP.md` and the more detailed implementation plan in `docs/improvement/RELEASE_PLAN_v1.md` to ensure each pillar is on track. The checklist below keeps the release gate lightweight:

1. **Alignment check** – confirm every planned change is captured either through
   the selected pillar in the roadmap or through supporting ADRs, and note the
   corresponding verification checklist from `vx.y.z_plan.md` (e.g. v0.10.1_plan.md).
2. **Gating commands** – run or re-run the canonical suite using `make ci-local`
   to ensure CI compliance and capture the results. This includes documentation
   builds, link checks, tests, and coverage verification.

   Preserve the `pytest --cov` summary for the release notes and update the
   public coverage badge if the percentage changes.
3. **Ownership sign-off** – collect approvals from the section owners listed in
   `docs/foundations/governance/section_owners.md`, including the runtime tech
   verification steps.
4. **Release notes** – draft highlights, call out telemetry/plugin optionality,
   and link to the detailed implementation plan in `vx.y.z_plan.md`.

Document completion in the release issue template so future audits can trace
any regressions back to this checklist.
