> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# ADR-017 Nomenclature Remediation

This consolidated plan combines the nomenclature review and standardization roadmap so the problem statement, phases, and success criteria live in one place.

## Severity assessment

**Rating: High** – Inconsistent naming conventions appear throughout multiple core modules, affecting classes, helper functions, and file names. These inconsistencies make it harder to navigate the package, complicate refactors, and increase the cognitive load for new contributors. Key pain points include:

- **Mixed module naming styles.** Core implementation files historically mixed CamelCase, snake_case, and legacy-style prefixes (for example, `core/venn_abers.py`, `core/interval_regressor.py`, `viz/plots.py`, and the now-removed `legacy/_plots_legacy.py` lived side-by-side), making it unclear which modules were public, private, or transitional. With the shim retired, `legacy/plotting.py` remains the sole legacy module bearing the plotting helpers.【F:src/calibrated_explanations/core/venn_abers.py†L1-L120】【F:src/calibrated_explanations/core/interval_regressor.py†L1-L40】【F:src/calibrated_explanations/viz/plots.py†L1-L40】【F:src/calibrated_explanations/legacy/plotting.py†L1-L120】
- **Ambiguous package vs. module naming.** The repository exposes both a `core` package and a `core.py` module as a deprecation shim, encouraging imports that look almost identical but carry different semantics. This duality raises the risk of developers importing the wrong target or duplicating logic during migrations.【F:src/calibrated_explanations/core/__init__.py†L1-L60】【F:src/calibrated_explanations/core.py†L1-L14】
- **Leaky private state references.** Helpers reach into double-underscore attributes such as `_CalibratedExplainer__initialized` and `_CalibratedExplainer__noise_type`, which bypass Python’s intended encapsulation and propagate mangled names into additional modules.【F:src/calibrated_explanations/core/calibration_helpers.py†L19-L115】
- **Kitchen-sink helper modules.** Broad utility modules like `utils/helper.py` accumulate unrelated responsibilities (filesystem, type checks, numerical transforms), mixing snake_case verbs with legacy names like `safe_isinstance` and `immutable_array`. Without a convention for grouping or prefixing, it becomes difficult to predict where a helper lives or how it should be named.【F:src/calibrated_explanations/utils/helper.py†L1-L160】

Collectively these issues indicate that nomenclature drift is systemic rather than localized, warranting a focused remediation plan.

## Phase plan

### Phase 0 – Preparatory work (week 0)
1. Circulate ADR-017 for acceptance and add a shortstyle excerpt to `CONTRIBUTING.md`.
2. Enable repository tooling (pre-commit hooks or CI) that flags non-snake_case filenames and new double-underscore attributes.
3. Inventory external entry-points (e.g. notebooks, docs snippets) that import internal modules so that rename impacts are known ahead of time.

### Phase 1 – Infrastructure & documentation (weeks 1–2)
1. Split `utils/helper.py` into topical modules (`fs_utils`, `validation_utils`, etc.) without renaming the functions yet to minimize diff churn.【F:src/calibrated_explanations/utils/helper.py†L1-L160】
2. Deprecate `core.py` by renaming the shim to `deprecated_core_module.py` and updating imports inside the package to target `calibrated_explanations.core`. Keep the old name as a thin import wrapper that emits a warning.【F:src/calibrated_explanations/core.py†L1-L14】
3. Document the new naming policy inside `docs/architecture.md` and the developer guides so contributors can reference expectations.

### Phase 2 – Module renames (weeks 3–5)
1. Rename CamelCase and underscored private modules (`core/venn_abers.py`, `_interval_regressor.py`, `viz/plots.py`) to descriptive snake_case equivalents. The canonical implementations now live in `core/venn_abers.py`, `core/interval_regressor.py`, and `viz/plots.py`; temporary import shims lived under `legacy/` until their scheduled v0.9.0 removal.【F:src/calibrated_explanations/core/venn_abers.py†L1-L120】【F:src/calibrated_explanations/core/interval_regressor.py†L1-L120】【F:src/calibrated_explanations/viz/plots.py†L1-L20】【F:src/calibrated_explanations/legacy/__init__.py†L1-L6】
2. Move shim modules and other legacy helpers into a dedicated `legacy/` namespace to visually distinguish transitional code.
3. Update internal imports, tests, and docs to reference the canonical module names and rely on the shims only for backwards compatibility.

### Phase 3 – Attribute and helper cleanup (weeks 6–8)
1. Replace direct access to mangled attributes (e.g. `_CalibratedExplainer__initialized`) with public or protected accessors so that helper modules stop propagating mangled names.【F:src/calibrated_explanations/core/calibration_helpers.py†L19-L115】
2. Rename helper functions to follow verb-first snake_case conventions, removing ambiguous prefixes such as `safe_` unless they convey a real semantic guarantee.
3. Update plugin identifiers and schema keys to follow the dot-delimited lowercase format described in ADR-017. Audit docs and code comments for alignment.

### Phase 4 – Enforcement & cleanup (weeks 9–10)
1. Remove deprecated shims once warning periods expire and downstream projects have migrated.
2. Lock linting rules in CI so regressions fail fast; add tests ensuring new modules adhere to the naming guide.
3. Retrospectively review the directory tree to ensure no new kitchen-sink helpers have formed; adjust module boundaries where needed.

## Success criteria
- Every module and package uses descriptive snake_case names with legacy shims isolated under a `legacy/` namespace.
- Helper modules are scoped by responsibility, and no new double-underscore attributes are introduced outside legacy compatibility layers.
- Contributor docs and automated tooling reinforce the conventions, preventing future drift.
- ADR-021 references link back to the terminology note relocated to the concepts library so interval semantics remain consistent.
