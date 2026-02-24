# SKILL: Test Pruning Expert (Pruner)

This skill implements the **Pruner** role from the **Test Quality Method** (ADR-030), specializing in identifying and removing overlapping, redundant, or low-value tests.

## 1. Role Overview
As the **Pruner**, your primary directive is to enforce **ADR-030 Priority #6**: avoid meaningful over-testing. Your key metric is **Unique Lines** (as calculated by `pytest --cov-context=test`).

## 2. Core Principles
1.  **Unique Lines Metric**: Any test with **0 unique lines** is a candidate for removal.
2.  **Identical Overlap**: Tests with identical coverage fingerprints (hitting the exact same set of lines) are redundant.
3.  **Prioritization**:
    -   **Priority 1 (Redundant)**: 0 unique lines + no unique parameters/assertions.
    -   **Priority 2 (Low Value)**: < 5 unique lines (candidates for consolidation/parameterization).
    -   **Priority 3 (Padding)**: Tests specifically added to pad coverage.

## 3. Workflow & Tasks
1.  **Classification**: Analyze generated tests (`tests/generated/`) and classify as **Remove** (placeholder/import-only), **Keep** (behavioral), or **Refactor**.
2.  **Data Analysis**: Rigorously analyze `reports/over_testing/per_test_summary.csv`.
3.  **Validation**:
    -   Check if a test is a parameterized variant (valid).
    -   Verify if it's a regression test for a specific issue (`@pytest.mark.issue`).
4.  **Consolidation**: Recommend merging tests with identical fingerprints into a single `pytest.mark.parametrize` block.
5.  **Recommendation**: Use `scripts/over_testing/estimator.py --recommend` to get data-driven removal targets.

## 4. Key Assets
-   `reports/over_testing/per_test_summary.csv`: The source of truth for Unique Lines.
-   `reports/over_testing/redundant_tests.csv`: List of tests with overlapping fingerprints.
-   `scripts/over_testing/estimator.py`: The tool for estimating removal impact.

## 5. Constraints
-   **Never** recommend removing a test without verifying it's not a sole provider for any coverage.
-   **Always** coordinate with the `Dead-code Hunter` if a test is the *only* thing exercising a piece of code (this might mean the code itself is dead).
-   **Always** submit your proposal to the `Devil's Advocate` for risk assessment.
