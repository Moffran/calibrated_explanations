# Test Quality Method

A repeatable method for enforcing high-quality tests (ADR-030) while keeping the
90% coverage gate intact.

This umbrella includes:

- Over-testing analysis using per-test coverage contexts (`--cov-context=test`)
- Under-testing remediation (targeted behavioral tests that close real gaps)
- Anti-pattern auditing (private-member usage, assertion-less tests, brittle patterns)
- Safe pruning (remove or consolidate low-value tests without breaking coverage)

## Prerequisites

```bash
pip install -e ".[dev]" -c constraints.txt
```

The dev dependencies include `pytest-cov` and `coverage` which provide the
per-test context tracking that this method relies on.

---

## Quick Start

Run the full pipeline from the repository root:

```bash
python scripts/over_testing/run_over_testing_pipeline.py
```

This single command:

1. Runs `pytest --cov-context=test` (records which test hits which line)
2. Runs `over_testing_report.py --require-multiple-contexts` (produces CSVs)
3. Runs `over_testing_triage.py --include-contexts` (produces triage reports)

All outputs land in `reports/over_testing/`.

> **Runtime**: Expect 3-10x slower than a normal `pytest` run due to per-test
> context recording. On this codebase, ~15-30 minutes depending on hardware.

---

## The Method (Step by Step)

### Step 1: Collect per-test coverage data

```bash
python scripts/over_testing/run_over_testing_pipeline.py
```

This is the **mandatory first step** every time you want fresh data. The
pipeline enforces `--cov-context=test` by default and the report step will
**refuse to proceed** if fewer than 2 contexts are detected (meaning per-test
tracking was not active).

**Outputs produced** (all in `reports/over_testing/`):

| File                         | Content                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `line_coverage_counts.csv`   | Per source line: file, line number, number of tests hitting it   |
| `block_coverage_counts.csv`  | Contiguous blocks with the same test count                       |
| `summary.json`               | Per-file summary: lines covered, mean/median/max test count      |
| `metadata.json`              | Run metadata: context count, warnings, threshold used            |
| `triage.json`                | Ranked hotspot files, lines, and blocks                          |
| `triage.md`                  | Human-readable triage report                                     |
| `hotspot_contexts.json`      | For each hotspot: which specific tests hit it                    |
| `hotspot_contexts.md`        | Human-readable hotspot-to-test mapping                           |

### Step 2: Extract per-test unique line counts and fingerprints

```bash
python scripts/over_testing/extract_per_test.py
python scripts/over_testing/detect_redundant_tests.py
```

IMPORTANT: `detect_redundant_tests.py` MUST be run as part of every
test-quality assessment. Preferred workflow to update `redundant_tests.csv`:

- Regenerate (preferred):

   1. Run the full pipeline: `python scripts/over_testing/run_over_testing_pipeline.py`
   2. Extract per-test summaries: `python scripts/over_testing/extract_per_test.py`
   3. Recreate redundancy report: `python scripts/over_testing/detect_redundant_tests.py`

- Manual exceptions (only when a generated entry is a false positive):

   1. Do not edit the generated `reports/over_testing/redundant_tests.csv` silently.
   2. Create `reports/over_testing/redundant_tests_review.csv` with these columns:
       `fingerprint,test_count,lines_covered,unique_lines_per_test,description,tests,status,reviewer,notes`
   3. Record `status` as `ACCEPTED`/`REJECTED`/`UNDER_REVIEW`, add a short `notes`, and commit
       `redundant_tests_review.csv` alongside your `reports/over_testing/final_remedy_plan.md`.
   4. If you must change the generated CSV, add an accompanying changelog entry in `CHANGELOG.md`.

This makes review decisions auditable and avoids losing the authoritative generated report.

Outputs:

| File                    | Content                                                                   |
| ----------------------- | ------------------------------------------------------------------------- |
| `per_test_summary.csv`  | Per test: name, unique lines (lines only this test covers), runtime       |
| `redundant_tests.csv`   | Per fingerprint: hash, test_count, unique_lines, list of test names       |

> **Primary Metric**: **Unique Lines**.
>
> A test with **0 unique lines** is a strong candidate for removal or consolidation.
> A test with **Identical Coverage Fingerprint** (same set of lines hit as another test)
> is a **Redundant Test** and must be removed or parameterized (per ADR-030).

### Step 3: Analyze the data

Use the triage reports and per-test summary to understand the current state.
Key questions to answer:

1. **How many tests have ZERO unique lines?** Count rows in `per_test_summary.csv` where `unique_lines=0`. These are the primary targets for pruning.
2. **Which tests have identical coverage fingerprints?** (Advanced) Identify groups of tests that hit exactly the same lines.
3. **What are the over-testing hotspots?** Read `triage.md` for the top files, lines, and blocks.
4. **Which tests hit each hotspot?** Read `hotspot_contexts.md`.

### Step 4: Simulate removals (optional, before acting)

Use the estimator to predict coverage impact before removing any tests:

```bash
# Get recommendations (low-value tests sorted by score)
python scripts/over_testing/estimator.py \
    --per-test reports/over_testing/per_test_summary.csv \
    --baseline reports/over_testing/baseline_summary.json \
    --recommend --budget 50

# Simulate removing a specific list of tests
python scripts/over_testing/estimator.py \
    --per-test reports/over_testing/per_test_summary.csv \
    --baseline reports/over_testing/baseline_summary.json \
    --remove-list candidates.txt
```

> **Safety rule**: Never apply removals whose estimated coverage is < 90%.

### Step 5: Act on findings

Apply changes (test removals, consolidations, new behavioral tests) and verify:

```bash
pytest --cov-fail-under=90
```

### Step 6: Re-run the pipeline to verify

After changes, re-run step 1 to get fresh data and confirm improvements:

```bash
python scripts/over_testing/run_over_testing_pipeline.py
```

---

## Agent Roles

The method is designed to be performed by specialist roles: read-only
perspectives that analyze in parallel and an implementer that consolidates
their findings and executes the cleanup. Each has a dedicated instruction file
in this directory. They can be run as AI agents or used as checklists for
manual analysis.

### [pruner.md](pruner.md) -- Test Removal Expert

Identifies overlapping and low-value tests for safe removal.

**Key tasks**:

- Classify generated/placeholder tests (keep, refactor, or remove)
- Analyze `per_test_summary.csv` for zero/low unique-line tests
- Identify overlapping hand-written tests across `tests/unit/`, `tests/focused/`, `tests/auto_approved/`
- Use `estimator.py --recommend` to rank removal candidates
- Produce prioritized removal batches with estimated coverage impact

**Key files to analyze**:

- `reports/over_testing/per_test_summary.csv`
- `reports/over_testing/baseline_summary.json`
- `reports/over_testing/prune_plan.json`
- `reports/over_testing/cov_fill_adr30_scan.csv`
- `scripts/over_testing/estimator.py`

**Key principles**:

- Prioritize safety over aggression: always simulate removals before applying them.
- Coordinate with `test-creator` so removals are sequenced with compensating tests where needed.
- Prefer public-contract coverage wins; avoid relying on private-member behavior.

### [deadcode_hunter.md](deadcode_hunter.md) -- Dead Source Code Expert

Identifies source code that is unreachable or only tested by low-value tests.

**Key tasks**:

- Run `scripts/anti-pattern-analysis/analyze_private_methods.py` to find dead private methods
- Analyze `reports/over_testing/gaps.csv` for large uncovered blocks
- Distinguish dead code from untested production code from conditionally reachable code
- Check lazy imports in `__init__.py` and plugin entry points for dynamic reachability
- Cross-reference with test coverage to find code only exercised by placeholder tests

**Key files to analyze**:

- `scripts/anti-pattern-analysis/analyze_private_methods.py`
- `reports/over_testing/gaps.csv`
- `reports/over_testing/line_coverage_counts.csv`
- `src/calibrated_explanations/__init__.py`
- `pyproject.toml` (entry points, plugin config)

**Key principles**:

- Verify runtime reachability (lazy imports, plugin entry points) before declaring code dead.
- Prefer conservative refactors that expose public seams rather than wholesale deletion.
- Document findings with concrete evidence (coverage hits, call graph snippets).

### [test_creator.md](test_creator.md) -- Coverage Gap Closer

Analyzes the coverage report to calculate the most efficient way to close
coverage gaps with new high-quality tests. Studies per-file and per-line
coverage data, ranks candidate files by the ratio of coverage gain to test
effort, and designs the smallest set of tests that produces the largest
coverage improvement — without introducing padding.

**Key tasks**:

- Analyze the pytest coverage report and sort files by missed statements to
   identify the highest-value gap-closing opportunities.
- Run gap analysis (`scripts/over_testing/gap_analyzer.py`) and check
   per-module gates (`scripts/quality/check_coverage_gates.py`) to find
   targets below threshold.
- Prioritize targets by **coverage gain per test line** (Tier 1/2/3) — this
   efficiency ranking is the core output of the analysis.
- Design concrete test sketches for Tier 1/2 targets (target, strategy,
   estimated gain, pseudocode, constraints).
- Verify each new test contributes > 0 unique lines via
   `scripts/over_testing/extract_per_test.py`; revise if redundant.
- Produce a test-creator proposal with a prioritized target table and
   specific test designs for the top targets.

**Key principles**:

- Use **public APIs only**; never access private members (`_name`) unless the
   allowlist explicitly permits it.
- Follow ADR-030 quality criteria: determinism, strong assertions,
   appropriate layering, and fixture discipline.
- Avoid import-only or assertion-less tests; every test must validate behavior.
- Prefer a few high-value tests over many low-value ones; aim for maximal
   coverage gain per test line.

**Key files**:

- `scripts/over_testing/gap_analyzer.py`
- `reports/over_testing/gaps.csv`
- `reports/over_testing/line_coverage_counts.csv`
- `reports/over_testing/per_test_summary.csv`
- `scripts/quality/check_coverage_gates.py`
- `scripts/over_testing/generate_test_templates.py`

**Key principles**:

- Tests must use public APIs and adhere to ADR-030 quality criteria (determinism, strong assertions, layering).
- Keep tests deterministic and fixture-driven; avoid randomness/time/network in unit tests.
- Provide clear assertions that fail for meaningful regressions; avoid vague existence checks.

### [anti_pattern_auditor.md](anti_pattern_auditor.md) -- Anti-Pattern Auditor

Audits the test suite for ADR-030 quality violations and anti-patterns (private
member usage, weak assertions, brittle coupling). Produces actionable
remediation recommendations and ensures any fallback-testing is visible
(warnings + logs) and justified.

**Key tasks**:

- Run the anti-pattern detector: `scripts/anti-pattern-analysis/detect_test_anti_patterns.py`.
- Audit private member usage with `scripts/anti-pattern-analysis/scan_private_usage.py` and
   cross-reference `.github/private_member_allowlist.json`.
- Analyze private method patterns via `analyze_private_methods.py` to find
   test-only helpers or dead private methods.
- Check assertion quality: locate tests without `assert`, weak `isinstance`
   checks, or bare truthiness assertions.
- Audit marker hygiene: ensure `slow`, `integration`, `viz`, and
   `platform_dependent` markers are applied where appropriate.
- Check fixture discipline: find duplicated setup, inappropriate autouse
   fixtures, and candidates for shared helpers in `tests/helpers/`.
- Produce an anti-pattern proposal with a summary table, private-member audit,
   assertion-quality report, marker hygiene fixes, and a prioritized remediation plan.

**Key principles**:

- Distinguish hard blockers (must-fix) from advisory findings and justify
   severity with examples and counts.
- Do not modify source or test files; this role is analysis-only.
- Provide reproducible evidence (file/line examples, grep/snippet commands,
   counts) so the `devils-advocate` and `implementer` can validate findings.
- Share findings with `pruner` and `test-creator` when anti-patterns overlap
   with removal or test-creation candidates.

**Key files**:

- `scripts/anti-pattern-analysis/detect_test_anti_patterns.py`
- `scripts/anti-pattern-analysis/scan_private_usage.py`
- `.github/private_member_allowlist.json`
- `reports/anti-pattern-analysis/test_anti_pattern_report.csv`
- `reports/over_testing/per_test_summary.csv`

**Key principles**:

- Distinguish hard blockers (must-fix) from advisory findings and justify
   severity with examples and counts.
- Do not modify code; provide reproducible evidence to enable safe remediation.
- Prioritize fixes that improve maintainability and reduce brittle coupling.

### [devils_advocate.md](devils_advocate.md) -- Critical Reviewer

Reviews proposals from all specialist agents and produces a consolidated,
cross-referenced risk assessment that reconciles conflicts across roles.

**Key tasks**:

- Challenge the `pruner`: Are "zero unique lines" classifications trustworthy?
- Challenge the `deadcode-hunter`: Is "dead" code truly dead or dynamically reachable?
- Challenge the `process-architect`: Are proposed quality gates realistic and feasible?
- Challenge the `test-creator`: Are proposed compensating tests realistic and deterministic?
- Challenge the `anti-pattern-auditor`: Are reported severities and remediation costs justified?
- Coordinate with the `implementer` to surface sequencing conflicts and execution constraints.
- Produce per-change risk ratings with mitigations and a reconciled execution order.
- Define a no-go list for changes that require prerequisites.

**Key principles**:

- Every criticism must include a concrete example or calculation.
- Consider interactions and conflicts across *all* specialist roles (pruner, deadcode-hunter, process-architect, test-creator, anti-pattern-auditor, implementer) when assessing risk.
- The `--cov-context=test` data quality is the foundation — verify it first.
- The 90% coverage floor must be maintained in every removal scenario.
- Reconcile conflicts across roles; where proposals contradict, surface sequencing and gating mitigations.
- Quantify net coverage impact for combined proposals (removals vs. added tests).
- Prefer conservative execution: require implementer sign-off before destructive changes.

**Key files**:

- `reports/over_testing/pruner_proposal.md`
- `reports/over_testing/deadcode_hunter_proposal.md`
- `reports/over_testing/process_architect_proposal.md`
- `reports/over_testing/test_creator_proposal.md`
- `reports/over_testing/anti_pattern_auditor_proposal.md`
- `reports/over_testing/devils_advocate_review.md`
- `reports/over_testing/baseline_summary.json`
- `reports/over_testing/per_test_summary.csv`

### [implementer.md](implementer.md) -- Consolidator and Executor

Merges the four expert proposals into a final remedy plan and carries out
the approved cleanup actions. This is the only role that modifies code.

**Key tasks**:

- Consolidate all proposals into `reports/over_testing/final_remedy_plan.md`
- Verify data freshness (check `metadata.json` for multiple contexts)
- Execute safe immediate actions (delete skipped/generated tests)
- Write compensating behavioral tests to close coverage gaps
- Remove coverage-padding tests after gap is closed
- Consolidate overlapping test pairs
- Re-run pipeline after changes and update reports

**Key constraint**: The 90% coverage gate is non-negotiable. Verify with
`pytest --cov-fail-under=90` after every batch of changes.

**Key files**:

- `reports/over_testing/final_remedy_plan.md` -- final consolidated plan
- `reports/over_testing/remedy_list.md` -- decision ledger
- `reports/over_testing/baseline_summary.json` -- current baseline metrics
- `scripts/over_testing/*` and `scripts/anti-pattern-analysis/*` -- tooling invoked during implementation

**Key principles**:

- Execute changes in phased batches and verify metrics after each phase.
- Maintain ADR conformance; consult `docs/improvement/adrs/` for relevant ADRs before major changes.
- Commit logically grouped changes (remove, then add compensating tests in separate commits).
-- Record rationale and results in `reports/over_testing/` and update `CHANGELOG.md` when milestones complete.

### [process_architect.md](process_architect.md) -- Process Optimization Expert

Designs and improves the test quality workflow and tooling.

**Key tasks**:

- Audit all scripts in `scripts/over_testing/` and `scripts/anti-pattern-analysis/`
- Assess current process gaps (documentation, automation, CI integration)
- Design quality gates with specific thresholds and ratcheting baselines
- Plan CI integration (per-PR checks vs periodic analysis)

**Key files to analyze**:

- All scripts in `scripts/over_testing/` and `scripts/anti-pattern-analysis/`
- `docs/improvement/test-quality-method/README.md`
- `docs/improvement/archived/over_testing_method.md` (historical notes)
- `docs/improvement/archived/over_testing_method.md` (archived stub)
- `scripts/over_testing/finalize_over_testing.md`
- `.github/workflows/` (CI definitions)
- `pyproject.toml` (pytest config, coverage config)

**Key principles**:

- Design gates to be actionable and measurable; avoid overly-brittle thresholds.
- Prioritize CI runtime and developer feedback loops when choosing what runs per-PR vs periodic.
- Ensure Windows compatibility and surface any platform-dependent caveats in the plan.

---

## Script Reference

### Pipeline scripts (`scripts/over_testing/`)

| Script                         | Purpose                                        | When to use                                |
| ------------------------------ | ---------------------------------------------- | ------------------------------------------ |
| `run_over_testing_pipeline.py` | End-to-end pipeline (pytest + report + triage) | **Start here** -- always run this first    |
| `over_testing_report.py`       | Generate per-line/block coverage counts        | Called by pipeline; can run standalone     |
| `over_testing_triage.py`       | Rank hotspots, produce triage reports          | Called by pipeline; can run standalone     |
| `extract_per_test.py`          | Extract per-test unique line counts            | Run after pipeline for per-test analysis   |
| `estimator.py`                 | Simulate coverage impact of test removals      | Run before removing tests                  |
| `gap_analyzer.py`              | Find contiguous untested code blocks           | After removals, to identify coverage gaps  |
| `generate_test_templates.py`   | Scaffold minimal test templates for gaps       | After gap analysis                         |
| `evaluate_cov_fill_adr030.py`  | Check generated tests for ADR-030 compliance   | During test classification                 |
| `prune_generated_tests.py`     | Classify and prune generated placeholder tests | During pruner analysis                     |
| `inspect_coverage_db.py`       | Inspect `.coverage` SQLite database            | Debugging coverage data issues             |
| `coverage_xml_gaps.py`         | Gap analysis from `coverage.xml`               | Alternative to `gap_analyzer.py`           |

### Anti-pattern scripts (`scripts/anti-pattern-analysis/`)

| Script                           | Purpose                                          | When to use                  |
| -------------------------------- | ------------------------------------------------ | ---------------------------- |
| `scripts/anti-pattern-analysis/detect_test_anti_patterns.py`   | Detect anti-patterns in test files               | CI + manual audits           |
| `analyze_private_methods.py`     | Find dead or test-only private methods in `src/` | Dead code analysis           |
| `scan_private_usage.py`          | Scan for private member usage across codebase    | Allowlist maintenance        |
| `generate_triage_report.py`      | Produce analysis triage report                   | After anti-pattern scan      |
| `summarize_analysis.py`          | Summarize anti-pattern findings                  | After triage                 |
| `find_shared_helpers.py`         | Find shared test helpers                         | Test consolidation           |
| `analyze_category_a.py`          | Analyze Category A anti-patterns                 | Detailed anti-pattern work   |
| `update_allowlist.py`            | Update private member allowlist                  | After allowlist changes      |

---

## Report Archive (`reports/over_testing/`)

After running the pipeline, the following reports are produced and can be
committed for future reference:

| File                       | Updated by                      | Content                        |
| -------------------------- | ------------------------------- | ------------------------------ |
| `line_coverage_counts.csv` | `over_testing_report.py`        | Per-line test counts           |
| `block_coverage_counts.csv`| `over_testing_report.py`        | Per-block test counts          |
| `summary.json`             | `over_testing_report.py`        | Per-file coverage summary      |
| `metadata.json`            | `over_testing_report.py`        | Run metadata and warnings      |
| `triage.json`              | `over_testing_triage.py`        | Ranked hotspots                |
| `triage.md`                | `over_testing_triage.py`        | Human-readable triage          |
| `hotspot_contexts.json`    | `over_testing_triage.py`        | Per-hotspot test lists         |
| `hotspot_contexts.md`      | `over_testing_triage.py`        | Human-readable hotspot contexts|
| `per_test_summary.csv`     | `extract_per_test.py`           | Per-test unique line counts    |
| `gaps.csv`                 | `gap_analyzer.py`               | Untested code blocks           |
| `baseline_summary.json`    | Manual / `extract_per_test.py`  | Coverage baseline              |

### Analysis reports (produced by agent analysis, not scripts)

| File                            | Content                                     |
| ------------------------------- | ------------------------------------------- |
| `pruner_proposal.md`            | Test removal analysis and recommendations   |
| `deadcode_hunter_proposal.md`   | Dead code analysis results                  |
| `process_architect_proposal.md` | Process redesign proposal                   |
| `devils_advocate_review.md`     | Consolidated risk assessment                |
| `final_remedy_plan.md`          | Merged action plan with verified data       |

---

## Safety Rules

1. **Always run with `--cov-context=test`**. The pipeline enforces this. Data
   collected without per-test contexts is unreliable for overlap analysis.

2. **Never remove tests whose estimated coverage drops below 90%**. Use
   `estimator.py` to check before acting.

3. **Re-run the pipeline after every batch of changes** to get fresh data.
   Coverage numbers shift as tests are added or removed.

4. **Distinguish "zero unique lines" from "useless"**. A test with zero unique
   lines still validates behavior -- it just shares code paths with other tests.
   Review what it actually asserts before removing.

5. **Check lazy imports and dynamic dispatch**. Source code that appears dead in
   static analysis may be reachable through `__init__.py` lazy imports, plugin
   entry points, or environment variable toggles.

6. **Maintain the remedy list**. Record all decisions in
   `reports/over_testing/remedy_list.md` and update the final remedy plan after
   each round.

7. **Protect docs + build-validation tests**. Avoid pruning tests under `tests/docs/` and tests marked `docs`/`rtd` unless you have an explicit docs CI strategy for replacements.

8. **Batch removals to reduce churn**. Prefer fewer, estimator-approved removal batches and re-run the pipeline after each batch; avoid micro-removals that require re-running the expensive pipeline repeatedly.

9. **Respect ADR-023 for viz coverage**. If coverage instrumentation breaks matplotlib adapter imports, follow ADR-023’s split strategy (run viz validation under `pytest --no-cov -m viz` and keep the coverage omit scoped to the documented adapter file).

---

## Example: Full Analysis Cycle

```bash
# 1. Collect fresh data (mandatory first step)
python scripts/over_testing/run_over_testing_pipeline.py

# 2. Extract per-test unique lines
python scripts/over_testing/extract_per_test.py

# 3. Review triage report
cat reports/over_testing/triage.md

# 4. Get removal recommendations
python scripts/over_testing/estimator.py \
    --per-test reports/over_testing/per_test_summary.csv \
    --baseline reports/over_testing/baseline_summary.json \
    --recommend --budget 50

# 5. Simulate a removal batch
python scripts/over_testing/estimator.py \
    --per-test reports/over_testing/per_test_summary.csv \
    --baseline reports/over_testing/baseline_summary.json \
    --remove-list my_removals.txt

# 6. Apply changes, then verify
pytest --cov-fail-under=90

# 7. Re-run pipeline to update reports
python scripts/over_testing/run_over_testing_pipeline.py
```

---

## History

| Date | Action | Result |
| --- | --- | --- |
| 2026-02-12 | First reliable `--cov-context=test` run | 2,967 contexts, 90.39% baseline |
| 2026-02-12 | Agent analysis (pruner, deadcode-hunter, process-architect, devils-advocate) | Final remedy plan produced |
| 2026-02-13 | Phase 1 executed: deleted 3 skipped tests, 42 generated tests, force_mark test; wrote 17 behavioral tests | Coverage at 90.01% without any padding |
