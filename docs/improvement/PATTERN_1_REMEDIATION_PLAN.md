# Pattern 1 Remediation Plan: Internal Logic Testing

## Overview
Pattern 1 (Category A) represents tests that directly access private members of the library to verify internal logic. This is the most significant anti-pattern in the codebase, with ~1000 occurrences across 163 unique methods.

This plan outlines a phased approach to refactoring these tests to use public APIs or justifying their inclusion in the allow-list.

## 1. Allow-list Strategy
We have established a baseline allow-list in `.github/private_member_allowlist.json` containing 93 entries. These entries are granted a temporary waiver until v0.11.0.

### Criteria for Allow-listing
1.  **Name-mangled internals**: Essential for verifying initialization state (e.g., `_CalibratedExplainer__initialized`).
2.  **Internal Factory/Setup**: Methods like `_from_config` used to bypass complex setup.
3.  **High Refactor Risk**: Methods with >20 test usages where refactoring is high effort.
4.  **Legacy Maintenance**: Private methods in legacy modules (e.g., `legacy/plotting.py`).

## 2. Execution Phases

### Phase 1: Low-Hanging Fruit (Refactor to Public API) - [COMPLETED]
**Target:** Methods with < 5 test usages that are not name-mangled or legacy.
*   **Goal:** Reduce the number of unique private methods being accessed.
*   **Strategy:** Identify the public effect of the private method and assert on that instead.
*   **Status:** Completed in v0.10.1. Refactored `_get_sigma_test`, `_feature_names`, and `_preprocessor_metadata`.

### Phase 2: Orchestrator & Plugin Manager Refactoring
**Target:** Core internal components like `orchestrator.py` and `manager.py`.
*   **Goal:** Use existing public accessors or introduce new ones where appropriate.
*   **Strategy:**
    *   Replace `explainer._plugin_manager` with `explainer.require_plugin_manager()`.
    *   Refactor orchestrator tests to use the public `explain()` or `predict_intervals()` methods with appropriate mocks.

### Phase 3: Integration & Parallel Subsystems
**Target:** `shap` integrations and `parallel` execution logic.
*   **Goal:** Decouple tests from internal state of these subsystems.
*   **Strategy:** Verify behavior via the public `WrapCalibratedExplainer` or `ParallelExecutor` interfaces.

### Phase 4: Final Review & Hardening
**Target:** Remaining allow-listed items.
*   **Goal:** Eliminate as many allow-list entries as possible.
*   **Strategy:**
    *   Convert essential internal factories to public `classmethod`s (e.g., `from_config`).
    *   Rewrite legacy tests to use the new plugin-based plotting architecture.

## 3. Progress Tracking
Progress will be tracked using the **Analysis Toolbox**:
1.  Run `scripts/anti-pattern-analysis/analyze_private_methods.py` to update definition data.
2.  Run `scripts/anti-pattern-analysis/scan_private_usage.py` to update usage data.
3.  Run `scripts/anti-pattern-analysis/summarize_analysis.py` for a high-level overview.

## 4. CI Enforcement
The CI guard will be updated to fail on any private member access in tests that is **not** in the allow-list.

---
*Last Updated: 2025-12-28*
