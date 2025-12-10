> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# Code Documentation Uplift (ADR-018)

This consolidated roadmap merges the ADR-018 docstring strategy with the `pydocstyle` execution guide so contributors have a single reference for planning and day-to-day execution.

## Goal
Achieve consistent, numpydoc-compliant code documentation across the `calibrated_explanations` codebase with measurable coverage thresholds and automated enforcement. Each phase below maps to the release train so stakeholders can see when gates turn on: Phase 0 in v0.8.0, Phase 1 in v0.9.1, Phase 2 in v0.10.0, and Phase 3 carried forward as the v1.0.0 sustainability bar. The uplift depends on the Documentation Overhaul blueprint to keep terminology and examples synchronized across prose and code (apply the same language for calibrated explanations, probabilistic regression, and optional telemetry callouts). Terminology follows hard guardrails without accidentally breaking published workflows. Terminology follows [terminology](RELEASE_PLAN_v1.md#terminology-for-improvement-plans): phases are numbered plan segments, and release milestones remain the versioned gates.

## Phase 0 – Preparation (Week 1)
1. Ratify ADR-018 and circulate a short style primer in `CONTRIBUTING.md` and the README.
2. Land shared tooling:
   - `pydocstyle` configuration targeting numpydoc rules (`convention = numpy`).
   - Python script that reports docstring coverage per module (baseline script already exists).
3. Define success metrics: ≥90% public callable coverage and zero undocumented modules by the end of Phase 2.

## Phase 1 – Baseline Remediation (Weeks 2-4)
1. Inventory undocumented callables per subpackage using the coverage script.
2. Prioritize user-facing areas (`explanations`, `utils`, `plugins`, `api`) for immediate cleanup.
3. Create parallel issues/checklists for each subpackage with assignees and review deadlines.
4. Add module summaries and upgrade docstrings to numpydoc format; capture tricky cases in a shared FAQ. Reinforce calibrated explanations/probabilistic regression as the primary narrative in examples and parameter descriptions, pair alternative explanations with triangular plot context, and mark telemetry/performance hooks (including fast explanation plugins) as optional references only.
5. Track progress in a dashboard (GitHub project or spreadsheet) updated weekly.
6. Target success metrics per batch before moving to Phase 2: Batch A–C reach ≥90% public callable coverage, Batch D reaches ≥88% (acknowledging plugin sprawl), Batch E ≥85% with an explicit waiver expiry by the next release, and Batch F aligns to the package-wide ≥90% goal.

## Phase 2 – Tooling Enforcement
1. Enable `pydocstyle` in CI as non-blocking (warning-only) to surface regressions.
2. Iterate on false positives; extend ignores only when accompanied by inline justification.
3. Once ≥85% coverage achieved, flip the CI check to blocking for touched files.
4. Capture and commit the initial failure report before blocking enforcement so future regressions reference a known baseline.
5. Release cadence alignment: v0.10.0 branch cut flips the blocking check for touched files; v0.10.1 raises the default to package-wide ≥90% and makes waivers expire within one iteration unless renewed with a documented follow-up issue.

## Phase 3 – Continuous Improvement (Ongoing)
1. Add a documentation coverage badge to README fed by scheduled job.
2. Extend linting to notebooks/examples via `nbdoclint` or custom hooks.
3. Review documentation debt quarterly; treat drops below 90% as release blockers.
4. Encourage contributors to add usage examples that highlight calibrated explanations and probabilistic regression outcomes; integrate with existing documentation CI (ADR-012) for end-to-end validation.

## `pydocstyle --convention=numpy` execution guide

### 1. Establish the baseline
1. Install `pydocstyle` into the active environment if it is not already available:
   ```bash
   pip install pydocstyle
   ```
2. Capture the current failure list for later comparison:
   ```bash
   mkdir -p reports
   pydocstyle --convention=numpy src tests | tee reports/pydocstyle-baseline.txt
   ```
   - Commit the report so future contributors can quickly spot regressions.
   - The command output identifies the exact rule (e.g., `D100`) and file/line triggering each violation.

### 2. Prioritize the work
Group files into batches that are small enough to review easily but large enough to deliver visible progress.

#### 2.1 Production Packages (src/calibrated_explanations)
| Batch | Scope | Rationale |
|-------|-------|-----------|
| A | `utils/`, `core/interval_regressor.py`, `core/venn_abers.py` | Low file count, foundational utilities likely reused by many modules. Fixing them early reduces duplicate effort later. |
| B | `api/` and `core/` package | Public-facing entry points; improving these files ensures that the surface area users see first meets style expectations. |
| C | `explanations/` and `perf/` packages | Mid-sized groups that are conceptually cohesive; tackle them after the high-visibility API pieces. |
| D | `plugins/` package | Largest cluster (9 files); split further per plugin if needed. |
| E | Visualization modules (`viz/`, `viz/plots.py`, `legacy/plotting.py`) | Visualization code often has long docstrings—tackle after gaining momentum with earlier batches. |
| F | Serialization and compatibility shims (`serialization.py`, `core.py`) | Final cleanup once the main logic is compliant. |

For each batch:
1. Run `pydocstyle --convention=numpy <paths>`.
2. Address one rule category at a time (e.g., first ensure every module has a docstring, then fix parameter sections, etc.).
3. Add regression tests or doctests where docstrings reveal missing behavioral details.

#### 2.2 Test Suite (tests/)
| Batch | Scope | Suggested Strategy |
|-------|-------|--------------------|
| G | Top-level fixtures (`_fixtures.py`, `_helpers.py`, `conftest.py`) | Small set of helpers used by many tests; clean documentation here clarifies test utilities. |
| H | `tests/plugins/` | Aligns with production plugins; can be handled in parallel once the related source files are compliant. |
| I | `tests/integration/` | Many short modules—tackle in groups of 5–6 files to keep reviews manageable. |
| J | `tests/unit/` | Highest file count (36). Address per logical sub-module (e.g., group by feature under test). |

### 3. Apply Numpy docstring patterns
While editing, consistently enforce the following conventions:
- Begin every public module, class, and function with a one-line summary.
- Include `Parameters`, `Returns`, and `Raises` sections when applicable.
- Document default values, shapes, and expected types for arrays or tensors.
- Use imperative mood for summary lines and keep them under 79 characters.
- When documenting generators or context managers, prefer `Yields` or `Receives` sections.

### 4. Automate enforcement
1. Add a `pydocstyle` invocation to CI (e.g., in `tox.ini`, `noxfile.py`, or the GitHub Actions workflow) once violations in the monitored paths are eliminated.
2. Update `CONTRIBUTING.md` to instruct contributors to run `pydocstyle --convention=numpy src tests` before submitting pull requests.

### 5. Track progress
- Keep the baseline report updated after each merged batch.
- Celebrate milestones (e.g., “`utils/` package fully compliant”) to maintain team momentum.
- Consider enabling branch protection that requires the pydocstyle check to pass once the codebase is clean.
