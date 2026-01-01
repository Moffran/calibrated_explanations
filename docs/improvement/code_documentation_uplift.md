> **Status note (2025-12-23):** Last edited 2025-12-23 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# Code Documentation Uplift (Standard-018)

This consolidated roadmap merges the Standard-018 docstring strategy with the `pydocstyle` execution guide so contributors have a single reference for planning and day-to-day execution.

## Goal
Achieve consistent, numpydoc-compliant code documentation across the `calibrated_explanations` codebase with measurable coverage thresholds and automated enforcement. The uplift has progressed to sustainment mode, with current baseline at 94.45% overall coverage and blocking enforcement at ≥94% on mainline CI. Notebook linting is already enforced via `nbqa ruff` in CI. The focus shifts to targeted cleanup of remaining gaps and regression prevention rather than broad phased remediation.

## Sustainment Actions

1. **Monitor coverage and enforce threshold:** Use `scripts/check_docstring_coverage.py` to ensure overall coverage remains ≥94% (current baseline: 94.45%). CI blocks on regressions below this threshold.
2. **Target remaining docstring gaps:** Run `pydocstyle` on affected packages to identify missing class/method docstrings. Prioritize fixes in modules with gaps (e.g., classes at 98.53%, methods at 92.24%).
3. **Refresh baseline reports:** Update and re-commit coverage snapshots and pydocstyle baselines in `reports/` after gap closures to maintain accurate known baselines.
4. **Notebook linting:** Already enforced via `nbqa ruff` in CI; maintain as part of standard linting cadence.
5. **Test documentation scope:** Confirmed out of scope per `pyproject.toml` `match-dir` exclusion; CI runs `pydocstyle src tests` but config excludes tests, so tests are not enforced. If inclusion desired, update config and plan accordingly.

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
Test documentation is out of scope per `pyproject.toml` `match-dir` exclusion, despite CI running `pydocstyle src tests`. No enforcement or batches planned for tests.

### 3. Apply Numpy docstring patterns
While editing, consistently enforce the following conventions:
- Begin every public module, class, and function with a one-line summary.
- Include `Parameters`, `Returns`, and `Raises` sections when applicable.
- Document default values, shapes, and expected types for arrays or tensors.
- Use imperative mood for summary lines and keep them under 79 characters.
- When documenting generators or context managers, prefer `Yields` or `Receives` sections.

### 4. Automate enforcement
1. `pydocstyle` is invoked in CI on `src tests` (though config excludes tests, so effectively only src).
2. Docstring coverage enforced at ≥94% via `scripts/check_docstring_coverage.py --fail-under 94.0`.
3. Update `CONTRIBUTING.md` to instruct contributors to run `pydocstyle --convention=numpy src` and check coverage before submitting pull requests.

### 5. Track progress
- Keep the baseline report updated after each merged batch.
- Celebrate milestones (e.g., “`utils/` package fully compliant”) to maintain team momentum.
- Consider enabling branch protection that requires the pydocstyle check to pass once the codebase is clean.
