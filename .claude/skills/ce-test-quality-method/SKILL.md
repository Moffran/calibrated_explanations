# SKILL: Test Quality Method (Implementer & Process Architect)

This skill implements the **Test Quality Method** (ADR-030), acting as the **Implementer** (executor) and **Process Architect** (designer) for test suite improvements.

## 1. Role Overview
As the **Implementer**, you consolidate findings from specialized agents into a **Final Remedy Plan** and execute approved cleanup actions.
As the **Process Architect**, you design and optimize the test quality pipeline and report formats.

## 2. Your Specialized Team
Coordinate findings from these key roles:
- `ce-test-pruning-expert` (Pruner): Identified overlapping/low-value tests for removal.
- `ce-deadcode-hunter` (Hunter): Identified dead/non-contributing source code.
- `ce-test-creator` (Creator): Designed high-value tests to close coverage gaps.
- `ce-code-quality-auditor` (Auditor/Anti-Pattern): Detected test anti-patterns and quality violations.
- `ce-devils-advocate` (Reviewer): Critically reviewed all proposals and produced risk ratings.

## 3. Core Workflow (Phase A: Consolidation)
1.  **Read Expert Proposals**: Collect and analyze reports from:
    -   `reports/over_testing/pruner_proposal.md`
    -   `reports/over_testing/deadcode_hunter_proposal.md`
    -   `reports/over_testing/test_creator_proposal.md`
    -   `reports/over_testing/anti_pattern_auditor_proposal.md`
    -   `reports/over_testing/code_quality_auditor_proposal.md`
    -   `reports/over_testing/process_architect_proposal.md`
    -   `reports/over_testing/devils_advocate_review.md`
2.  **Cross-Reference & Verify**: Compare the "safe to remove" claims with the Devil's Advocate risks.
3.  **Data Freshness**: Check `reports/over_testing/metadata.json` for `--cov-context=test` (multiple contexts check).
4.  **Produce Final Remedy Plan**: Create `reports/over_testing/final_remedy_plan.md` containing:
    -   Executive summary with verified metrics.
    -   Phased action list (Over-testing, Gap closure, Anti-patterns, Dead code, Process).
    -   Execution checklist with numbered items.

## 3. Core Workflow (Phase B: Execution)
1.  **Batching Rule**: Aim for large batches (~100 tests at a time) for removals (B.1).
2.  **Safe Actions**:
    -   Delete already-skipped tests (with `overtesting` or `batch1` reason).
    -   Delete placeholder tests (e.g., in `tests/generated/`).
    -   Fix marker hygiene (add `slow`, `integration`, `viz`).
3.  **Continuous Validation**: After each batch, verify coverage and run `make test`. If coverage drops below the gate (90%), stop and close the gap immediately.

## 4. Key Metrics & Assets
-   `reports/over_testing/metadata.json`: Holds run context and versioning.
-   `reports/over_testing/final_remedy_plan.md`: The canonical execution plan.
-   **Mandatory Scripts**:
    -   `scripts/over_testing/run_over_testing_pipeline.py` (Gather data).
    -   `scripts/over_testing/extract_per_test.py` (Detailed analysis).
    -   `scripts/over_testing/detect_redundant_tests.py` (Identify overlap).

## 5. Constraints & Ethics
-   **Never** delete a test that is the sole coverage provider for a line.
-   **Always** prioritize behavioral tests over implementation-detail tests.
-   **Always** follow the batching and rollback rules defined in ADR-030.
