---
name: ce-evidence-linker
description: >
  Curate CE verification evidence linking claims, requirements, TIFs, and
  concrete test results.
---

# CE Evidence Linker

## Use this skill when

- Generating a curated evidence summary after a verification run
- Diagnosing which CE-CAP / CE-REQ claims lack passing evidence
- Updating an evidence file after a patch or new test run
- Producing a human-readable chain status report (claim → req → TIF → test → evidence)

## Inputs

- Raw evidence output from `reports/verification/` or `make capability-evidence-refresh`
- TIF register from `development/capabilities/verification/tif/README.md`
- Requirements from `development/capabilities/requirements/`
- Claims from `development/capabilities/claims/`

## Workflow

1. **Parse test results.** Run or read the capability test suite:
   ```bash
   pytest tests/capabilities/ -v --tb=short 2>&1
   ```
   Identify each test ID, its pass/fail status, and the TIF or requirement it covers
   (from test naming convention `test_should_<behavior>_when_<condition>`).

2. **Map tests to TIF IDs.** Use the TIF README table in
   `development/capabilities/verification/tif/README.md` to resolve
   `test_id → TIF ID → requirement IDs → claim IDs`.

3. **Compute coverage.** For each claim:
   - List all child requirements
   - For each requirement, list all TIF IDs
   - For each TIF, list tests with their pass/fail status
   - A requirement is evidenced only when ALL its TIFs have at least one passing test

4. **Produce a curated evidence summary** in
   `development/capabilities/evidence/evidence_<area>_<version>.md`.
   Follow the curated evidence format from `development/capabilities/evidence/README.md`.
   Minimum fields per evidence record:
   - `evidence_id`: `CE-EVID-<AREA>-<NNN>-<YYYYMMDD>`
   - `claim_ids`: list of CE-CAP-... IDs
   - `requirement_ids`: list of CE-REQ-... IDs
   - `tif_ids`: list of CE-TIF-... IDs (empty for TIF-exempt requirements)
   - `verification_type`: api_contract | behavioral_contract | numerical_behavior | empirical_smoke
   - `test_id`: full pytest node ID
   - `commit_sha`: git HEAD SHA
   - `package_version`: calibrated_explanations.__version__
   - `result`: pass | fail

5. **Identify open gaps.** List requirements with no passing evidence in priority order:
   - No test exists for the TIF (untested)
   - Test exists but is failing (failing)
   - Test passes but no TIF link (orphan — zero traceability credit)

6. **Flag regressions.** Compare against previous evidence summary. Flag any
   TIF that previously had passing evidence but now shows a failing test.

7. **Update requirement `verification_status` fields** where newly passing evidence
   closes a previously `unverified` requirement. Set to `verified`.

## Verification

```bash
make capability-chain-check
python scripts/quality/validate_capability_chain.py
```

## Output contract

Return:

1. Curated evidence summary file contents for `development/capabilities/evidence/`.
2. Full chain-status table: claim → requirement → TIF → test → status.
3. Gap list: requirements with no passing evidence (categorised by gap type).
4. Regression list: TIFs that regressed since last evidence record.
5. List of `verification_status` fields updated to `verified`.

## Constraints

- Only passing tests provide positive evidence — a failing test does NOT close a requirement.
- Orphan tests (no TIF link) contribute zero traceability credit. List them in the gap report.
- Evidence records must include `commit_sha` and `package_version` — a "pass" without
  these fields is not usable as formal evidence.
- `verification_status` in a requirement file must only be set to `verified` when:
  (a) all TIF IDs for that requirement have a passing test, and
  (b) the evidence record has been written with the required fields.
- Do not suppress failing tests from the gap list. Every failure must be visible.
- Curated evidence summaries go to `development/capabilities/evidence/`. Raw JSON/JSONL
  outputs from CI go to `reports/verification/` — do not mix these directories.
