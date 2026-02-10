# Finalize Over-Testing Process

This document describes the restartable workflow used to raise baseline coverage
and safely prune auto-generated or otherwise redundant tests. Follow these
steps to reproduce or continue the work.

1. Ensure you have a fresh authoritative coverage artifact: `coverage.xml` in
   the repo root produced by running the usual pipeline (once). This file is the
   authoritative baseline used by the estimator.

2. Compute current line-rate from `coverage.xml` (attribute `lines-covered` /
   `lines-valid`). Target: `COVERAGE_TARGET = 0.905` (90.5%).

3. If coverage < target, add focused tests that exercise concrete public
   behaviors in the smallest number of lines possible. Prefer behavior-first
   assertions per ADR-030.

4. Re-run the pipeline once to validate coverage. If >= target, proceed to
   pruning. If not, repeat step 3 with more focused tests.

5. To prune generated/duplicative tests, run `scripts/over_testing/prune_generated_tests.py`
   (it requires either (a) per-test mapping from `pytest --cov-context=test` or
   (b) a conservative, manual allowlist). The script is conservative by
   default and will create a `reports/over_testing/prune_plan.json` before
   modifying test files.

6. Evaluate all removals using CI (full pipeline) — do not merge prunes without
   a full pipeline verification run.

Notes
- All decisions and remedy items must be recorded in `reports/over_testing/remedy_list.md`.
- Follow ADR-030 when evaluating and moving tests (assertions, markers, public API use).
