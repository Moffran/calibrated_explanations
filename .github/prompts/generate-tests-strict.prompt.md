# /generate-tests-strict

You are generating tests under the repository’s "Test File Creation & Grouping Policy". Follow these rules strictly:
- First, identify the SUT and locate the nearest existing test file.
- Extend an existing file unless *all* new-file criteria are met.
- Use the correct directory/name mapping for the language.
- Reuse fixtures; keep tests deterministic; use AAA structure.
- Output a short summary of where tests were placed and why.

Inputs (optional):
- `target=<path/to/source_file>`
- `scope=unit|integration|e2e`
- `framework=pytest|jest|vitest|...`
