> **Status note (2026-01-28):** Last edited 2026-01-28 · Archive after: Retain indefinitely as architectural record · Implementation window: Completed in v0.11.0.

# ADR-010: Core vs Evaluation Split & Distribution Strategy

Status: Accepted (completed in v0.11.0)
Date: 2025-09-02
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

The repository contains both the installable core library (`src/calibrated_explanations`) and a significant amount of evaluation, notebooks, and research/development artifacts (`evaluation/`, `debug/`, `notebooks/`, `scripts/`). While these folders are not included in the distributed package (setuptools finds packages only under `src`), current default dependencies in the core package include heavier, user-facing tools (e.g., `ipython`, `matplotlib`, `lime`). This increases install footprint and can blur boundaries between the core runtime and auxiliary tooling.

Additionally, plot code is part of the package and tests exercise plotting. We want to keep plotting supported, but make visualization and evaluation dependencies clearly optional to streamline the core and improve comprehension for new users.

## Decision

Adopt a monorepo with a strict “core vs. evaluation” separation, and define clear distribution boundaries for optional capabilities:

- Keep a single repository. Maintain the installable library confined to `src/calibrated_explanations`.
- Keep research/evaluation assets in top-level folders (`evaluation/`, `debug/`, `notebooks/`, `scripts/`) that are excluded from packaging.
- Introduce optional dependency groups in `pyproject.toml`:
  - `viz`: plotting stack (e.g., `matplotlib`);
  - `notebooks`: Jupyter authoring helpers (e.g., `ipython`, `jupyter`, `nbformat`);
  - `dev`: `viz` + testing/linting/type tools;
  - `eval`: adds packages used by the evaluation harness.
- Keep default/core runtime dependencies minimal (e.g., `numpy`, `pandas`, `scikit-learn`, `crepes`, `venn-abers`), moving `ipython`, `lime`, and `matplotlib` to optional extras where feasible.
- Mark plotting code as an optional feature at runtime: attempt lazy imports and raise a clear, actionable error if the `viz` extra is not installed (implementation scheduled after plan approval).
- Document the split in README and CONTRIBUTING, with a quickstart that works without `viz` when plotting is not requested.
- Define extension boundaries: optional packages or plugins that add features must be opt-in and interact only through public extension points. Core APIs and numerical behavior must remain unchanged when optional packages are not enabled.
- Add compatibility checks that assert core results are unchanged when optional packages are installed but disabled.

Future consideration:

- If evaluation grows further, consider moving it to a sibling repo (e.g., `calibrated-explanations-eval`) with its own environment lock and CI. Start with monorepo; revisit post v0.7.0.

## Consequences

Positive:

- Smaller default install, faster import for core usage.
- Clearer mental model: core library vs. evaluation/research assets.
- Easier to maintain and communicate optional features.
- Stronger guarantees that optional capabilities do not alter core behavior by default.

Negative / Risks:

- Test suite must adapt to optional visualization (skip or mark when extras not installed).
- Some users may need to change installation commands (e.g., `pip install calibrated-explanations[viz]`).
- Additional compatibility tests increase CI time.

## Adoption & Migration

Phase 2S (see release plan):

1. Add `[project.optional-dependencies]` groups in `pyproject.toml` for `viz`, `notebooks`, `dev`, `eval` (no behavior changes yet).
2. Update README with install matrix and examples for extras.
3. Mark plotting imports as lazy and raise a helpful error if missing (guarded by tests marked `viz`).
4. Tag or mark tests requiring visualization with `@pytest.mark.viz` and conditionally skip when extras are absent.
5. Add an `evaluation/README.md` describing its scope and a `conda`/`pip` environment file for evaluation runs.
6. Optional: create a GitHub Actions workflow that installs `[eval]` and runs evaluation jobs separately from core CI.

### Adoption Progress (2025-09-02)

Done/Partial:

- Optional-dependency groups `viz` and `lime` added to `pyproject.toml`.
- Plotting code performs lazy import and raises clear error with install hint for `viz`.
- Getting Started docs reference the `viz` extra.

Pending:

- Add remaining extras: `notebooks`, `dev`, `eval` with finalized package lists.
- Update README installation section with extras matrix and examples.
- Mark/skip viz-dependent tests and add a CI job without viz extras to enforce core independence.
- Optional evaluation workflow that installs `[eval]` and runs benchmarks.
- Add compatibility checks confirming optional packages do not change core outputs unless explicitly enabled.

## Alternatives Considered

1. Split into multiple pip distributions immediately (core, viz, eval). Deferred to reduce maintenance overhead; extras provide a simpler first step.
2. Status quo. Retains heavier default deps and less clear boundaries.

## Addendum (2026-01-28): Core-only Import Verification for v0.11.0

### Decision
Implement verification that core-only installs do not require matplotlib at import time, as targeted for v0.11.0.

### Rationale
The ADR specified that core-only installs should not require matplotlib at import time. Lazy imports in `__init__.py` prevent eager loading of heavy dependencies like matplotlib, ensuring core functionality remains lightweight.

### Implementation
- **Lazy Imports:** Confirmed that `calibrated_explanations.__init__.py` uses lazy loading for optional dependencies, preventing matplotlib from being imported on core package import.
- **Test Verification:** Added `test_import_package_does_not_eagerly_import_matplotlib()` in `tests/test_core_import_no_matplotlib.py` to assert that `matplotlib` is not in `sys.modules` after importing `calibrated_explanations`.
- **CI Enforcement:** The test ensures ongoing compliance in CI pipelines.

### Testing
- Unit test passes when matplotlib is not installed or not eagerly imported.
- Test simulates clean import environment by clearing `sys.modules` of matplotlib-related entries.

## Open Questions

- Minimum set of default dependencies for a smooth quickstart. Start conservative; iterate based on user feedback.
- Exact package list for `eval` extra depends on evaluation scripts; define during implementation.

## Decision Notes

This ADR complements ADR-001 (boundaries), ADR-036 (PlotSpec canonical contract), and ADR-037 (visualization extension governance). The optional extras approach prepares for ADR-036/ADR-037 by making visualization an explicit, optional capability without blocking core usage.
