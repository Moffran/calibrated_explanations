# /fix-issue

Diagnose and fix a bug or failing test in `calibrated_explanations`.

## Before you start

1. Read `docs/improvement/RELEASE_PLAN_v1.md` to understand what milestone is active.
2. Scan `docs/improvement/adrs/` for the ADR(s) most relevant to the affected module.

## Steps Copilot must follow

1. **Reproduce** – identify the minimal failing test or repro script.
2. **Locate root cause** – trace from the public API (`WrapCalibratedExplainer`) inward; check
   `src/calibrated_explanations/core/` first, then `plugins/`, then `calibration/`.
3. **Write a regression test (TDD red)** – follow `.github/tests-guidance.md`; add to the nearest
   existing test file.
4. **Fix (TDD green)** – make the minimal change required; do not refactor unrelated code.
5. **Verify no regressions** – run `make test` and confirm coverage remains ≥ 90 %.
6. **Fallback check** – if the fix touches a fallback path, assert `warnings.warn` is emitted;
   follow §7 of `copilot-instructions.md`.
7. **Update `CHANGELOG.md`** under `## [Unreleased]` with a one-line entry.

## Inputs (optional)

- `issue=<GitHub issue number or short description>`
- `module=<dotted.module.path>` (e.g. `calibrated_explanations.core.explainer`)
- `failing_test=<path/to/test_file.py::test_name>`

## Checklist on completion

- [ ] Regression test added and green.
- [ ] Root cause comment added inline if non-obvious.
- [ ] `CHANGELOG.md` entry added.
- [ ] `make ci-local` passes.
