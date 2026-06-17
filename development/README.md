# Development Documentation Map

This map defines the internal development documentation surface for
Calibrated Explanations.

`docs/` is for user-facing and contributor-facing documentation. `development/`
is for maintainer planning, engineering governance, capability claim and
requirement catalogs, and curated closure evidence summaries.

This file defines the target layout. Some existing material still lives in
`docs/improvement/` during migration.

---

## Canonical Locations

| Material | Location |
|---|---|
| Active development planning | `development/current-work/` |
| Forward-looking planning | `development/future-work/` |
| Closed planning and curated closure evidence summaries | `development/finished-work/` |
| Architectural Decision Records | `development/adrs/` |
| Engineering Standards | `development/standards/` |
| Capability claims | `development/capabilities/claims/` |
| Requirements derived from capability claims | `development/capabilities/requirements/` |
| Verification scenarios and helpers | `verification/capabilities/` |
| Pytest capability verification | `tests/capabilities/` |
| Generated verification run outputs | `reports/verification/` |

These locations are authoritative even when a directory has not yet been
created. Do not create additional locations for the same material.

---

## Verification Layout Rule

Capability verification follows this chain:

```text
Capability claim
    -> requirement
    -> verification case
    -> evidence record
```

Use the location map above for each layer:

- Claims belong in `development/capabilities/claims/`.
- Requirements belong in `development/capabilities/requirements/`.
- Executable verification scenarios and helpers belong in
  `verification/capabilities/`.
- Pytest checks belong in `tests/capabilities/` for new capability-contract
  tests. Existing nearby unit or integration tests may be linked from
  requirements when they already verify the required public behavior.
- Generated run outputs belong in `reports/verification/`.
- Human-curated release or closure evidence summaries belong in
  `development/finished-work/`.

Do not add claim catalogs, requirement catalogs, schemas, verification code,
tests, or evidence records unless a task explicitly asks for that work.

---

## Migration Rule

`docs/improvement/` is a legacy planning area.

- Existing files in `docs/improvement/` remain valid until migrated.
- Do not add new planning, ADR, Standard, claim, requirement, verification
  framework, or curated evidence files to `docs/improvement/`.
- If active material in `docs/improvement/` is being substantially changed,
  move it to the appropriate `development/` location in the same change when
  that move is within scope.
- Keep path references accurate while migration is incomplete; do not update a
  reference to `development/` until the referenced file has actually moved.

---

## Navigation

Start here, then follow the current locations:

1. Active release plan: `development/current-work/` after migration; currently
   existing plans may still be in `docs/improvement/`.
2. ADRs: `development/adrs/` after migration; currently existing ADRs may still
   be in `docs/improvement/adrs/`.
3. Standards: `development/standards/` after migration; currently existing
   Standards may still be in `docs/standards/`.
4. Test guidance: `tests/README.md`.
5. Repository-wide agent rules: `CONTRIBUTOR_INSTRUCTIONS.md`.
