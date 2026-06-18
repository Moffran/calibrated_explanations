# Verification Scenarios and Helpers — Capabilities

This directory contains executable verification scenarios and helpers for CE
capability verification.

Verification scenarios are distinct from pytest tests. They may be runnable
standalone scripts, parameterized helpers, or scenario libraries that back
multiple tests. They belong here because they are part of the verification
infrastructure, not test assertions.

Pytest test files belong in `tests/capabilities/`.

## Role in the verification chain

```text
Capability claim        -> development/capabilities/claims/
    -> requirement      -> development/capabilities/requirements/
    -> verification case
          -> scenarios and helpers (this directory)
          -> pytest tests in tests/capabilities/
    -> evidence record  -> reports/verification/
                        -> development/capabilities/evidence/ (curated summaries)
```

## Naming

Scenario files should name the capability area they serve:

```
scenario_<area>.py
helpers_<area>.py
```

## Rules

1. Scenarios implement requirements, not claims directly.
2. Each scenario or helper should reference the requirement ID(s) it serves
   in its module or function docstring.
3. Scenarios must be runnable in isolation and must not depend on test fixtures
   from `tests/conftest.py`.
4. Do not encode acceptance criteria only in scenario code — the criteria must
   be visible in the requirements files in `development/capabilities/requirements/`.

## Related locations

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
