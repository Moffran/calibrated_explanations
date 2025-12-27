# Anti-Pattern Remediation Plan: Private Helper Calls

## Executive Summary
The repository currently contains >1200 instances of tests accessing private members (methods/attributes starting with `_`). This creates high coupling between tests and implementation details, making refactoring difficult and tests brittle.

This document outlines a systematic approach to eliminating these anti-patterns by categorizing them, applying specific refactoring strategies, and establishing prevention mechanisms.

## 1. Analysis & Categorization

We have identified four primary categories of private member usage in tests:

### Category A: Internal Logic Testing (High Severity)
Tests that directly call private methods to verify internal logic, bypassing the public API.
*   **Examples:** `explainer._get_sigma_test()`, `explainer.__set_mode()`, `explainer._check_explanation_runtime_metadata()`.
*   **Risk:** Prevents internal refactoring; tests implementation details rather than behavior.
*   **Remediation:** Rewrite tests to verify the *effect* of the logic via public methods (`explain()`, `predict()`, `__init__`).

### Category B: Test Utilities (Medium Severity)
Helper functions defined in `conftest.py` or `_fixtures.py` that are named with a leading underscore but used widely across tests.
*   **Examples:** `_make_binary_dataset()`, `_FixtureLearner`, `_run_quickstart_classification()`.
*   **Risk:** Ambiguous scope. `_` implies internal/private, but they are effectively public test APIs.
*   **Remediation:** Rename to public (remove `_`) and move to a dedicated `tests.helpers` module.

### Category C: Deprecated/Legacy Access (Medium Severity)
Tests accessing private members that correspond to deprecated features or have been replaced by new public APIs.
*   **Examples:** `explainer._is_lime_enabled()` (deprecated in favor of `LimePipeline`).
*   **Risk:** Tests keep deprecated code alive and don't verify the new recommended usage.
*   **Remediation:** Update tests to use the new public API.

### Category D: Factory/Setup Bypass (Low Severity)
Tests using private factory methods to instantiate objects in specific states, bypassing standard initialization.
*   **Examples:** `WrapCalibratedExplainer._from_config()`.
*   **Risk:** Tests invalid states that cannot be reached by users.
*   **Remediation:** Use public constructors or factories. If the state is valid but hard to reach, expose a public factory method or `from_config` (without `_`).

## 2. Remediation Strategy

### Phase 1: Triage & Grouping (Automated)
We have established an **Analysis Toolbox** in the `scripts/` directory:
*   [analyze_private_methods.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/analyze_private_methods.py): Scans `src/` for definitions and tracks usages across the project. Identifies Pattern 3 candidates.
*   [scan_private_usage.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/scan_private_usage.py): Scans tests for private usages and categorizes them using definition data from the analysis.
*   [summarize_analysis.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/summarize_analysis.py): Provides a high-level summary of findings.

### Refined Category Data (Initial Scan)
Initial analysis shows approximately 1426 private usages in tests:
- **Category A (Internal Logic):** ~1043 usages (e.g. `_plugin_manager`, `_interval_context_metadata`).
- **Category B (Test Utilities):** ~350 usages (helpers defined in `tests/`).
- **Category D (Factory Bypass):** ~14 usages (e.g. `_from_config`).
- **Pattern 3 (Dead Code):** 8 unique candidates (defined in `src/`, only called from tests).

## 2. Remediation Strategy

#### Pattern 1: The "Internal Logic" Fix
**Before:**
```python
explainer._CalibratedExplainer__set_mode("classification", initialize=False)
assert explainer.num_classes == 3
```
**After:**
```python
# Test via public interface (init)
explainer = CalibratedExplainer(model, cal_data, mode="classification")
assert explainer.num_classes == 3
```

#### Pattern 2: The "Test Utility" Fix
**Before:**
```python
from ._fixtures import _make_binary_dataset
X, y = _make_binary_dataset()
```
**After:**
```python
# Move to tests/helpers/dataset_utils.py
from tests.helpers.dataset_utils import make_binary_dataset
X, y = make_binary_dataset()
```

#### Pattern 3: The "Dead Code" Fix
If a private method is *only* called by tests and not by any library code:
1.  Remove the test case.
2.  Remove the private method from the library.

### Phase 3: Prevention (CI/Linting)
Implement a CI check that fails if new private member accesses are introduced in tests.
*   **Tool:** Custom AST-based linter or `flake8` plugin.
*   **Policy:** Zero new violations. Existing violations are whitelisted until fixed.

## 3. Execution Plan

1.  **Tooling:** Maintain the **Analysis Toolbox** (`scripts/analyze_private_methods.py`, etc.) to track progress.
2.  **Batch 0 (Dead Code):** Remove the 8 identified Pattern 3 candidates and their associated tests.
3.  **Batch 1 (Test Utilities):** Rename and move `_make_binary_dataset`, `_run_quickstart_*`, etc. (~25% of cases).
4.  **Batch 2 (Core Internals):** Refactor `CalibratedExplainer` and `orchestrator` internal tests (Category A).
5.  **Batch 3 (Cleanup):** Final sweep for remaining Category D and deprecated members.
6.  **CI Enforcement:** Enable a linter check.


## 4. Coverage Maintenance
Refactoring must not decrease coverage.
*   If a private method was tested directly, ensure the new public-API test covers the same code paths.
*   Use `pytest --cov` to verify coverage before and after each batch.
