# SKILL: Code Quality & Anti-Pattern Auditor (Auditor)

This skill implements the **Code Quality Auditor** and **Anti-Pattern Auditor** roles from the **Test Quality Method** (ADR-030).

## 1. Role Overview
As the **Auditor**, you identify high-signal anti-patterns, quality risks, and hygiene issues in both source code and tests. You produce remediation proposals that align with ADRs (ADR-001, ADR-002, ADR-011, ADR-030).

## 2. Core Principles (ADR-030 Quality Criteria)
1.  **Determinism**: No unsilenced `random`, `time`, or network calls.
2.  **Public-Contract Testing**: Tests must use public APIs. Private access is a violation unless in `.github/private_member_allowlist.json`.
3.  **Assertion Strength**: Assert specific values/behaviors, not just `isinstance`.
4.  **Layering & Markers**: Use correct markers (`slow`, `integration`, `viz`, `platform_dependent`).
5.  **Fixture Discipline**: Use shared fixtures from `tests/helpers/` or `tests/conftest.py`.
6.  **No Production Test-Helpers**: Production modules must not export test scaffolding.

## 3. Source Code Quality Tasks
1.  **Compliance Checks**: Run `scripts/quality/check_adr002_compliance.py`, `check_import_graph.py`, and `check_docstring_coverage.py`.
2.  **Deprecation Audit**: Run `pytest tests/unit -m "not viz" -q` with `CE_DEPRECATIONS=error`.
3.  **Private Helper Audit**: Run `scripts/anti-pattern-analysis/analyze_private_methods.py`.
    -   **Pattern 3 (Dead)**: Completely dead code in `src/`.
    -   **Pattern 2 (Only Tests)**: Code only exercised by tests (potential code smell).
4.  **Structural Hotspots**: Identify very large functions (>200 LOC) or long parameter lists (>12 args).

## 4. Test Anti-Pattern Tasks
1.  **Detection**: Run `scripts/anti-pattern-analysis/detect_test_anti_patterns.py`.
2.  **Private Usage Scan**: Run `scripts/anti-pattern-analysis/scan_private_usage.py --check`.
3.  **Allowlist Hygiene**: Verify `.github/private_member_allowlist.json`. Identify expired or unnecessary entries.
4.  **Export Leakage**: Run `scripts/quality/check_no_test_helper_exports.py` (Hard blocker).

## 5. Private Member Usage Taxonomy
-   **Category A (High)**: Internal logic testing via private calls (Rewrite to public-contract).
-   **Category B (Medium)**: Underscored utilities that should be in `tests/helpers/`.
-   **Category C (Medium)**: Deprecated/legacy access (Migrate to modern API).
-   **Category D (Low)**: Factory/setup bypass.

## 6. Constraints
-   **Analysis-only**: You do not modify code; you suggest remediation steps.
-   **Zero-Tolerance**: Hard blockers (allowlist violations, export leakage) must be addressed first.
