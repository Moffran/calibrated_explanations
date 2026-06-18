# Capability Evidence — Curated Summaries

This directory contains human-curated evidence summaries for CE capability
verification runs. Evidence records here are produced after a verification
run in `reports/verification/` and capture a human-reviewed snapshot of
what was verified, which test IDs passed, and against which package version.

## Role in the verification chain

```text
Capability claim        -> development/capabilities/claims/
    -> requirement      -> development/capabilities/requirements/
    -> verification case
          -> scenarios and helpers (development/capabilities/verification/)
          -> pytest tests          (tests/capabilities/)
    -> evidence record  -> reports/verification/      (generated run output)
                        -> development/capabilities/evidence/ (this directory — curated)
```

## When to add an evidence record

Add a curated summary here when:
- A capability release milestone is reached (e.g., v0.11.4 Task 9 sign-off).
- A failing requirement was fixed and a re-run confirms it now passes.
- A regression was investigated and confirmed to be absent.

Do NOT add raw pytest output here. Raw output belongs in `reports/verification/`.

## File naming

```
evidence_<area>_<version>.md
```

Examples: `evidence_expl_conj_v0.11.4.md`, `evidence_filter_ops_v0.11.4.md`

## Required content per evidence record

Each file must state:

- **requirement_ids**: which CE-REQ-... IDs are covered
- **package_version**: the calibrated_explanations version under test
- **commit_sha**: the commit at which the run was executed
- **test_ids**: named test functions that passed
- **dataset_id**: dataset used (e.g., sklearn make_classification, random_seed=42)
- **result**: PASS / FAIL / PARTIAL
- **notes**: any reviewer observations (optional)

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated (raw) run outputs | `reports/verification/` |
