# Nomenclature Review

## Severity assessment

**Rating: High** – Inconsistent naming conventions appear throughout multiple core modules, affecting classes, helper functions, and even file names. These inconsistencies make it harder to navigate the package, complicate refactors, and increase the cognitive load for new contributors. Key pain points include:

- **Mixed module naming styles.** Core implementation files mix CamelCase, snake_case, and legacy-style prefixes (for example, `_VennAbers.py`, `_interval_regressor.py`, `_plots.py`, and `_plots_legacy.py` live side-by-side), making it unclear which modules are public, private, or transitional. 【F:src/calibrated_explanations/_VennAbers.py†L1-L144】【F:src/calibrated_explanations/_interval_regressor.py†L1-L40】【F:src/calibrated_explanations/_plots.py†L1-L40】【F:src/calibrated_explanations/_plots_legacy.py†L1-L40】
- **Ambiguous package vs. module naming.** The repository exposes both a `core` package and a `core.py` module as a deprecation shim, encouraging imports that look almost identical but carry different semantics. This duality raises the risk of developers importing the wrong target or duplicating logic during migrations. 【F:src/calibrated_explanations/core/__init__.py†L1-L60】【F:src/calibrated_explanations/core.py†L1-L14】
- **Leaky private state references.** Helpers reach into double-underscore attributes such as `_CalibratedExplainer__initialized` and `_CalibratedExplainer__noise_type`, which bypass Python’s intended encapsulation and propagate mangled names into additional modules. This pattern entrenches inconsistent attribute prefixes and encourages further ad-hoc naming to work around access barriers. 【F:src/calibrated_explanations/core/calibration_helpers.py†L19-L115】
- **Kitchen-sink helper modules.** Broad utility modules like `utils/helper.py` accumulate unrelated responsibilities (filesystem, type checks, numerical transforms), mixing snake_case verbs with legacy names like `safe_isinstance` and `immutable_array`. Without a convention for grouping or prefixing, it becomes difficult to predict where a helper lives or how it should be named. 【F:src/calibrated_explanations/utils/helper.py†L1-L160】

Collectively these issues indicate that nomenclature drift is systemic rather than localized, warranting a focused remediation plan.

## Recommended focus areas

1. Establish a canonical naming style guide covering files, modules, classes, functions, and private attributes.
2. Clarify how transitional shims (such as `core.py`) should be labeled, documented, and eventually retired.
3. Consolidate helper utilities into purpose-specific modules so that names communicate intent without redundant prefixes.
4. Introduce linting or CI checks (e.g., custom Ruff rules) to prevent regressions once conventions are adopted.
