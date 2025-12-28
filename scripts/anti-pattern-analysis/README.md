# Anti-Pattern Analysis Tools

This directory contains scripts for identifying and analyzing anti-patterns in the codebase, specifically focusing on private member access in tests.

## Scripts

- `analyze_private_methods.py`: Scans the library (`src/`) for private member definitions and tracks their usage across the project. Categorizes findings into patterns (e.g., Pattern 1: Leaked internals, Pattern 2: Test utilities, Pattern 3: Dead code).
- `scan_private_usage.py`: Scans the test suite for private member calls and categorizes them using the data from `analyze_private_methods.py`.
- `summarize_analysis.py`: Provides a high-level summary of the anti-pattern status, highlighting top targets for remediation.
- `generate_triage_report.py`: Generates a prioritized triage report (`test_only_private_refs.csv` and `triage_next_actions.md`) by combining definition and usage data.
- `detect_test_anti_patterns.py`: A general-purpose AST-based scanner for various test anti-patterns (private calls, exact path comparisons, etc.).
- `find_shared_helpers.py`: Specifically identifies private test helpers that are defined in multiple files (Pattern 2 candidates).

## Workflow

1. Run `analyze_private_methods.py` to generate the base report:
   ```bash
   python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv
   ```
2. Run `summarize_analysis.py` to see the current status:
   ```bash
   python scripts/anti-pattern-analysis/summarize_analysis.py
   ```
3. Use `scan_private_usage.py` for a detailed per-file report of violations:
   ```bash
   python scripts/anti-pattern-analysis/scan_private_usage.py tests
   ```
4. Generate the triage report for manual remediation:
   ```bash
   python scripts/anti-pattern-analysis/generate_triage_report.py
   ```
