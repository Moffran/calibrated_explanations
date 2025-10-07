# Documentation Standardization Plan

## Goal
Achieve consistent, numpydoc-compliant documentation across the `calibrated_explanations`
codebase with measurable coverage thresholds and automated enforcement.

## Phase 0 – Preparation (Week 1)
1. Ratify ADR-018 and circulate summary in maintainer channel.
2. Add a documentation style primer to `CONTRIBUTING.md` and link from the README.
3. Land shared tooling:
   - `pydocstyle` configuration targeting numpydoc rules (`convention = numpy`).
   - Python script that reports docstring coverage per module (baseline script already exists).
4. Define success metrics: ≥90% public callable coverage and zero undocumented modules by the
   end of Phase 2.

## Phase 1 – Baseline Remediation (Weeks 2-4)
1. Inventory undocumented callables per subpackage using the coverage script.
2. Prioritize user-facing areas (`explanations`, `utils`, `plugins`, `api`) for immediate cleanup.
3. Create parallel issues/checklists for each subpackage with assignees and review deadlines.
4. Add module summaries and upgrade docstrings to numpydoc format; capture tricky cases in a
   shared FAQ.
5. Track progress in a dashboard (GitHub project or spreadsheet) updated weekly.

## Phase 2 – Tooling Enforcement (Weeks 5-6)
1. Enable `pydocstyle` in CI as non-blocking (warning-only) to surface regressions.
2. Iterate on false positives; extend ignores only when accompanied by inline justification.
3. Once ≥85% coverage achieved, flip the CI check to blocking for touched files.

## Phase 3 – Continuous Improvement (Ongoing)
1. Add documentation coverage badge to README fed by scheduled job.
2. Extend linting to notebooks/examples via `nbdoclint` or custom hooks.
3. Review documentation debt quarterly; treat drops below 90% as release blockers.
4. Encourage contributors to add usage examples; integrate with existing documentation CI
   (ADR-012) for end-to-end validation.

