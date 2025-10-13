# Roadmap for Achieving `pydocstyle --convention=numpy` Compliance

This document breaks the repository-wide docstring overhaul into focused, repeatable workstreams. Each stage can be executed in isolation and merged independently, gradually driving the codebase toward full compliance.

## 1. Establish the Baseline
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

## 2. Prioritize the Work
Group files into batches that are small enough to review easily but large enough to deliver visible progress.

### 2.1 Production Packages (src/calibrated_explanations)
| Batch | Scope | Rationale |
|-------|-------|-----------|
| A | `utils/`, `core/interval_regressor.py`, `core/venn_abers.py` | Low file count, foundational utilities likely reused by many modules. Fixing them early reduces duplicate effort later. |
| B | `api/` and `core/` package | Public-facing entry points; improving these files ensures that the surface area users see first meets style expectations. |
| C | `explanations/` and `perf/` packages | Mid-sized groups that are conceptually cohesive; tackle them after the high-visibility API pieces. |
| D | `plugins/` package | Largest cluster (9 files); split further per plugin if needed. |
| E | Visualization modules (`viz/`, `viz/plots.py`, `legacy/_plots_legacy.py`) | Visualization code often has long docstrings—tackle after gaining momentum with earlier batches. |
| F | Serialization and compatibility shims (`serialization.py`, `core.py`) | Final cleanup once the main logic is compliant. |

For each batch:
1. Run `pydocstyle --convention=numpy <paths>`.
2. Address one rule category at a time (e.g., first ensure every module has a docstring, then fix parameter sections, etc.).
3. Add regression tests or doctests where docstrings reveal missing behavioral details.

### 2.2 Test Suite (tests/)
| Batch | Scope | Suggested Strategy |
|-------|-------|--------------------|
| G | Top-level fixtures (`_fixtures.py`, `_helpers.py`, `conftest.py`) | Small set of helpers used by many tests; clean documentation here clarifies test utilities. |
| H | `tests/plugins/` | Aligns with production plugins; can be handled in parallel once the related source files are compliant. |
| I | `tests/integration/` | Many short modules—tackle in groups of 5–6 files to keep reviews manageable. |
| J | `tests/unit/` | Highest file count (36). Address per logical sub-module (e.g., group by feature under test). |

## 3. Apply Numpy Docstring Patterns
While editing, consistently enforce the following conventions:
- Begin every public module, class, and function with a one-line summary.
- Include `Parameters`, `Returns`, and `Raises` sections when applicable.
- Document default values, shapes, and expected types for arrays or tensors.
- Use imperative mood for summary lines and keep them under 79 characters.
- When documenting generators or context managers, prefer `Yields` or `Receives` sections.

## 4. Automate Enforcement
1. Add a `pydocstyle` invocation to CI (e.g., in `tox.ini`, `noxfile.py`, or the GitHub Actions workflow) once violations in the monitored paths are eliminated.
2. Update `CONTRIBUTING.md` to instruct contributors to run `pydocstyle --convention=numpy src tests` before submitting pull requests.

## 5. Track Progress
- Keep the baseline report updated after each merged batch.
- Celebrate milestones (e.g., “`utils/` package fully compliant”) to maintain team momentum.
- Consider enabling branch protection that requires the pydocstyle check to pass once the codebase is clean.

Following these staged batches prevents large, risky docstring refactors and makes it practical to reach full Numpy-style compliance over time.
