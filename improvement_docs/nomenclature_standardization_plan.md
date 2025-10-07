# Nomenclature remediation roadmap

This plan outlines sequential changes for aligning the internal codebase with ADR-017 while keeping the public API untouched.

## Phase 0 – Preparatory work (week 0)
1. Circulate ADR-017 for acceptance and add a shortstyle excerpt to `CONTRIBUTING.md`.
2. Enable repository tooling (pre-commit hooks or CI) that flags non-snake_case filenames and new double-underscore attributes.
3. Inventory external entry-points (e.g. notebooks, docs snippets) that import internal modules so that rename impacts are known ahead of time.

## Phase 1 – Infrastructure & documentation (weeks 1–2)
1. Split `utils/helper.py` into topical modules (`fs_utils`, `validation_utils`, etc.) without renaming the functions yet to minimize diff churn. 【F:src/calibrated_explanations/utils/helper.py†L1-L160】
2. Deprecate `core.py` by renaming the shim to `deprecated_core_module.py` and updating imports inside the package to target `calibrated_explanations.core`. Keep the old name as a thin import wrapper that emits a warning. 【F:src/calibrated_explanations/core.py†L1-L14】
3. Document the new naming policy inside `docs/architecture.md` and the developer guides so contributors can reference expectations.

## Phase 2 – Module renames (weeks 3–5)
1. Rename CamelCase and underscored private modules (`_VennAbers.py`, `_interval_regressor.py`, `_plots.py`, `_plots_legacy.py`) to descriptive snake_case equivalents (e.g. `venn_abers.py`, `interval_regressor.py`, `plot_builders.py`, `plot_legacy_adapter.py`). Provide import shims that proxy to the new names and emit `DeprecationWarning`. 【F:src/calibrated_explanations/_VennAbers.py†L1-L144】【F:src/calibrated_explanations/_interval_regressor.py†L1-L40】【F:src/calibrated_explanations/_plots.py†L1-L40】【F:src/calibrated_explanations/_plots_legacy.py†L1-L40】
2. Move shim modules and other legacy helpers into a dedicated `legacy/` namespace to visually distinguish transitional code.
3. Update internal imports, tests, and docs to reference the canonical module names and rely on the shims only for backwards compatibility.

## Phase 3 – Attribute and helper cleanup (weeks 6–8)
1. Replace direct access to mangled attributes (e.g. `_CalibratedExplainer__initialized`) with public or protected accessors so that helper modules stop propagating mangled names. 【F:src/calibrated_explanations/core/calibration_helpers.py†L19-L115】
2. Rename helper functions to follow verb-first snake_case conventions, removing ambiguous prefixes such as `safe_` unless they convey a real semantic guarantee.
3. Update plugin identifiers and schema keys to follow the dot-delimited lowercase format described in ADR-017. Audit docs and code comments for alignment.

## Phase 4 – Enforcement & cleanup (weeks 9–10)
1. Remove deprecated shims once warning periods expire and downstream projects have migrated.
2. Lock linting rules in CI so regressions fail fast; add tests ensuring new modules adhere to the naming guide.
3. Retrospectively review the directory tree to ensure no new kitchen-sink helpers have formed; adjust module boundaries where needed.

## Success criteria
- Every module and package uses descriptive snake_case names with legacy shims isolated under a `legacy/` namespace.
- Helper modules are scoped by responsibility, and no new double-underscore attributes are introduced outside legacy compatibility layers.
- Contributor docs and automated tooling reinforce the conventions, preventing future drift.
