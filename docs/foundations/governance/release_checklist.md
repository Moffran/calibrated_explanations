# Release checklist

Before publishing a release, review `ROADMAP.md` and the more detailed implementation plan in `development/current-work/RELEASE_PLAN_v1.md` to ensure each pillar is on track. The checklist below keeps the release gate lightweight:

1. **Alignment check** – confirm every planned change is captured either through
   the selected pillar in the roadmap or through supporting ADRs, and note the
   corresponding verification checklist from `vx.y.z_plan.md` (e.g. v0.10.1_plan.md).
2. **Gating commands** – run or re-run the canonical suite using `make ci-local`
   to ensure CI compliance and capture the results. This includes documentation
   builds, link checks, tests, and coverage verification.

   Preserve the `pytest --cov` summary for the release notes and update the
   public coverage badge if the percentage changes.
3. **Legacy API gate** – verify that the legacy user API contract documented in
   `development/finished-work/legacy_user_api_contract.md` is intact. Run
   `pytest tests/unit/api/test_legacy_user_api_contract.py -v` and confirm
   zero failures. If any legacy surface changed, confirm it was explicitly
   scheduled by the release plan and that the contract doc and parity tests
   were updated in the same PR (ADR-020).
4. **Ownership sign-off** – collect approvals from the section owners listed in
   `docs/foundations/governance/section_owners.md`, including the runtime tech
   verification steps.
5. **Release notes** – draft highlights, call out telemetry/plugin optionality,
   and link to the detailed implementation plan in `vx.y.z_plan.md`.

Document completion in the release issue template so future audits can trace
any regressions back to this checklist.
