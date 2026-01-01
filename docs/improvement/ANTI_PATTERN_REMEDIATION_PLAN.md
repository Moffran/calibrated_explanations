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
*   [analyze_private_methods.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/anti-pattern-analysis/analyze_private_methods.py): Scans `src/` for definitions and tracks usages across the project. Identifies Pattern 3 candidates.
*   [scan_private_usage.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/anti-pattern-analysis/scan_private_usage.py): Scans tests for private usages and categorizes them using definition data from the analysis.
*   [summarize_analysis.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/anti-pattern-analysis/summarize_analysis.py): Provides a high-level summary of findings.
*   [analyze_category_a.py](file:///c:/Users/loftuw/Documents/Github/kristinebergs-calibrated_explanations/scripts/anti-pattern-analysis/analyze_category_a.py): Performs deep analysis on Category A (Internal Logic) methods to identify allow-list candidates.

### Refined Category Data (Current Status)
Analysis shows approximately 1074 private usages in tests:
- **Category A (Internal Logic):** ~1008 usages (e.g. `_plugin_manager`, `_interval_context_metadata`).
- **Category B (Test Utilities):** ~52 usages (helpers defined in `tests/`).
- **Category D (Factory Bypass):** ~14 usages (e.g. `_from_config`).
- **Pattern 3 (Dead Code):** 0 unique candidates (Remediation Complete).

## 3. Remediation Patterns

### Pattern 1: The "Internal Logic" Fix
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

See the detailed [Pattern 1 Remediation Plan](PATTERN_1_REMEDIATION_PLAN.md) for the phased execution strategy.

### Pattern 2: The "Test Utility" Fix
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

### Pattern 3: The "Dead Code" Fix
If a private method is *only* called by tests and not by any library code:
1.  Remove the test case.
2.  Remove the private method from the library.

## 4. Prevention (CI/Linting)
Implement a CI check that fails if new private member accesses are introduced in tests.
*   **Tool:** Custom AST-based linter or `flake8` plugin.
*   **Policy:** Zero new violations. Existing violations are whitelisted in `.github/private_member_allowlist.json` until fixed.

### Allow-list Policy
Methods may be added to the allow-list if they meet one of the following criteria:
1.  **Name-mangled internals**: Essential for verifying initialization state in unit tests (e.g., `_CalibratedExplainer__initialized`).
2.  **Internal Factory/Setup**: Methods like `_from_config` used to bypass complex setup in integration tests.
3.  **High Refactor Risk**: Methods with >20 test usages where refactoring would be extremely high effort with low immediate benefit.
4.  **Legacy Maintenance**: Private methods in legacy modules (e.g., `legacy/plotting.py`) that are only tested by legacy tests.

All allow-list entries must have an **expiry version** (defaulting to the next major release gate, e.g., `v0.11.0`).

## 5. Execution Plan

1.  **Tooling:** Maintain the **Analysis Toolbox** (`scripts/analyze_private_methods.py`, etc.) to track progress.
2.  **Batch 0 (Dead Code):** Remove the 8 identified Pattern 3 candidates and their associated tests.
3.  **Batch 1 (Test Utilities):** Rename and move `_make_binary_dataset`, `_run_quickstart_*`, etc. (~25% of cases).
4.  **Batch 2 (Core Internals):** Refactor `CalibratedExplainer` and `orchestrator` internal tests (Category A).
5.  **Batch 3 (Cleanup):** Final sweep for remaining Category D and deprecated members.
6.  **CI Enforcement:** Enable a linter check.


## 6. Coverage Maintenance
Refactoring must not decrease coverage.
*   If a private method was tested directly, ensure the new public-API test covers the same code paths.
*   Use `pytest --cov` to verify coverage before and after each batch.
