# Generated Verification Run Outputs

This directory contains machine-generated outputs from capability verification
runs: raw pytest output, coverage reports, and structured result files produced
by automated verification scripts.

## Role in the verification chain

```text
Capability claim        -> development/capabilities/claims/
    -> requirement      -> development/capabilities/requirements/
    -> verification case
          -> scenarios  (development/capabilities/verification/)
          -> pytest      (tests/capabilities/)
    -> evidence record  -> reports/verification/      (this directory — generated)
                        -> development/capabilities/evidence/ (curated summaries)
```

## What belongs here

- Raw `pytest --tb=short -v` output captured to file
- `coverage.xml` or `coverage.json` from capability test runs
- Structured JSON/YAML result files produced by verification scripts
- CI artifact outputs from capability gates

## What does NOT belong here

- Human-written evidence summaries → `development/capabilities/evidence/`
- Claim or requirement files → `development/capabilities/claims/` or `requirements/`
- Verification scripts → `development/capabilities/verification/`

## File naming

```
run_<area>_<date>.<ext>
```

Examples: `run_expl_conj_2026-06-17.txt`, `run_filter_ops_2026-06-17.json`
