# Capability Tests

This directory contains pytest tests that verify CE capability requirements.

Each test here is explicitly linked to one or more requirement IDs from
`development/capabilities/requirements/` and provides machine-executable
verification of the acceptance criteria stated in those requirements.

## Role in the verification chain

```text
Capability claim        -> development/capabilities/claims/
    -> requirement      -> development/capabilities/requirements/
    -> verification case
          -> helpers     (development/capabilities/verification/)
          -> tests       (this directory)
    -> evidence record  -> reports/verification/
                        -> development/capabilities/evidence/ (curated summaries)
```

## File naming

Test files should name the requirement area they cover:

```
test_<area>_contracts.py
```

Example: `test_explanation_contracts.py`, `test_prediction_contracts.py`

## Test rules (ADR-030)

Tests in this directory must follow all standard CE test rules:

1. **Determinism** — no wall-clock time, nondeterministic RNG, network I/O.
   Seed RNG explicitly.
2. **Public-contract testing** — test observable public behavior; do not access
   `_private` members.
3. **Strong assertions** — assert specific values or invariants that would fail
   for plausible regressions.
4. **Requirement link** — every test must reference its requirement ID in a
   docstring, comment, or `pytest.mark` decoration.
5. **Acceptance criteria visible** — criteria must be stated in the requirement
   file, not hidden only inside test code.

## Running

```bash
pytest tests/capabilities/ -v
```

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Generated run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
