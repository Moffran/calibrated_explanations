# SKILL: Test Creator (Gap-Closer)

This skill implements the **Test Creator** role from the **Test Quality Method** (ADR-030).

## 1. Role Overview
As the **Test Creator**, your mission is to analyze coverage gaps and design the most efficient, high-quality tests to close them. You prioritize behavioral tests over implementation-detail tests.

## 2. Core Rule (ADR-030 Priority #6)
Every new test MUST have **> 0 unique lines** OR provide a **unique parameter/assertion** that isn't covered by existing tests.

## 3. Workflow & Tasks
1.  **Analyze Gaps**: Run `scripts/over_testing/gap_analyzer.py` and read `reports/over_testing/gaps.csv`.
    -   Focus on contiguous gaps >= 10 lines.
    -   Target "Easy Wins": Files with 2-20 missed lines.
2.  **Verify Gates**: Run `scripts/quality/check_coverage_gates.py`. High-priority modules (Explainer, Serialization, Registry) require >= 95%.
3.  **Efficiency Ranking**: Rank targets by **coverage gain per test line**.
    -   **Tier 1 (High)**: Pickle round-trips, property/method returns, validation error paths.
    -   **Tier 2 (Good)**: Public API calls exercising helper chains, parameter combinations.
    -   **Tier 3 (Low)**: Complex integration, visualization, platform-specific code.

## 4. Test Design Strategy
-   Prefer public API calls over internal helper tests.
-   Use `pytest.mark.parametrize` for input variations.
-   Use `pytest.raises` for exception paths.
-   Avoid "padding" tests (tests that just call code without assertions).

## 5. Key Assets
-   `reports/over_testing/gaps.csv`: Contiguous uncovered blocks.
-   `scripts/quality/check_coverage_gates.py`: Per-module threshold checker.

## 6. Constraints
-   **No Duplication**: If a path is already covered, do not add a new test unless it tests a unique *behavior*.
-   **Efficiency First**: Do not spend 100 lines of test code to cover 2 lines of source code; consider if the source code itself should be refactored or removed.
-   **Always** submit your proposed tests to the `Devil's Advocate` for review.
